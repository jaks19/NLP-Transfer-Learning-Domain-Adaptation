from parameters import *

from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

# Processes all sentences in out datasets to give useful containers of data concerning the corpus:
# word2id vocab
# dict of question id to list of words in the question
def process_whole_corpuses():
    list_dataset_paths = [ubuntu_corpus_path, android_corpus_path]
    all_txt = []
    ubuntu_id_to_data = {}
    android_id_to_data = {}

    for dataset_path in list_dataset_paths:
        lines = open(dataset_path).readlines()
        for line in lines:

            id_title_body_list = line.split('\t')
            idx = int(id_title_body_list[0])
            title_plus_body = id_title_body_list[1] + ' ' + id_title_body_list[2][:-1]
            all_txt.append(title_plus_body)

            if dataset_path == ubuntu_corpus_path:
                ubuntu_id_to_data[idx] = title_plus_body.split()
            else:
                android_id_to_data[idx] = title_plus_body.split()

    vectorizer = CountVectorizer(binary=True, analyzer='word', token_pattern='[^\s]+[a-z]*[0-9]*')
    vectorizer.fit(all_txt)

    return {'word_to_id': vectorizer.vocabulary_, 'ubuntu_id_to_data': ubuntu_id_to_data, 'android_id_to_data': android_id_to_data}


# Get glove embeddings matrix only for words in our corpus (++Gain of gigabytes of memory)
# Matrix [num_words_with_embeddings x word_dim] is be fed to pytorch nn.Embedding module without gradient
# Function returns this nn.Embedding Object
def load_glove_embeddings(glove_path, word_to_id_vocab, embedding_dim=300):
    with open(glove_path) as f:
        glove_matrix = np.zeros((len(word_to_id_vocab), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word_to_id_vocab.get(word)
            if index:
                try:
                    vector = np.array(values[1:], dtype='float32')
                    glove_matrix[index] = vector
                except:
                    pass

    glove_matrix = torch.from_numpy(glove_matrix).float()
    torch_embedding = nn.Embedding(glove_matrix.size(0), glove_matrix.size(1), padding_idx=padding_idx)
    torch_embedding.weight = nn.Parameter(glove_matrix)
    torch_embedding.weight.requires_grad = False

    return torch_embedding


# Takes a question id and the corresponding dict of question_id_to_words
# Builds a matrix of [1 x num_words x input_size] where first dim is for concatenation in future
# Use up to TRUNCATE_LENGTH number of words and pad if needed
def get_question_matrix(question_id, dict_qid_to_words, words_to_id_vocabulary, pytorch_embeddings):
    question_data = dict_qid_to_words[question_id]
    word_ids = []

    # Build list of ids of words in that question
    for word in question_data:
        if len(word_ids) == 100: break

        word_ids.append(int(words_to_id_vocabulary[word.lower()]))

    # Pad if need more rows
    number_words_before_padding = len(word_ids)
    if number_words_before_padding < truncate_length: word_ids += [padding_idx] * (truncate_length - len(word_ids))

    question_in_embedded_form = pytorch_embeddings(torch.LongTensor(word_ids)).data
    return question_in_embedded_form.unsqueeze(0), number_words_before_padding


# Given ids of main qs in this batch
# Returns:
# 1. ids in ordered list as:
# [ q_1+, q_1-, q_1--,..., q_1++, q_1-, q_1--,...,
# q_2+, q_2-, q_2--,..., q_2++, q_2-, q_2--,...,]
# All n main questions have their pos,neg,neg,neg,... interleaved
# 2. A dict mapping main question id --> its interleaved sequence length
def organize_ids_training(q_ids, data, num_differing_questions):
    sequence_ids = []
    dict_sequence_lengths = {}

    for q_main in q_ids:
        p_pluses = data[q_main][0]
        p_minuses = list(np.random.choice(data[q_main][1], num_differing_questions, replace=False))
        sequence_length = len(p_pluses) * num_differing_questions + len(p_pluses)
        dict_sequence_lengths[q_main] = sequence_length
        for p_plus in p_pluses:
            sequence_ids += [p_plus] + p_minuses

    return sequence_ids, dict_sequence_lengths


# Given ids of main qs in this android batch
# Returns:
# 1. list of ids of all the questions if candidates
# 2. list of tuples, (q_main_id, num_candidates)
# 3. list of 1,0... 1 for pos, 0 for neg (wrt. candidates) to be used in AUC metric
def organize_test_ids(q_ids, data):
    processed_ids = []
    target_labels = []
    q_main_pattern = []

    for q_main in q_ids:
        all_p = data[q_main][1]
        p_pluses = data[q_main][0]
        for p in all_p:
            if p in p_pluses:
                target_labels.append(1)
            else:
                target_labels.append(0)
        processed_ids += all_p
        q_main_pattern.append((q_main, len(all_p)))

    return processed_ids, q_main_pattern, target_labels


# A tuple is (q+, q-, q--, q--- ...)
# Let all main questions be set Q
# Each q in Q has a number of tuples equal to number of positives |q+, q++, ...|
# Each q in Q will have a 2D matrix of: num_tuples x num_candidates_in_tuple
# Concatenate this matrix for all q in Q and you get a matrix of: |Q| x num_tuples x num_candidates_in_tuple

# The above is for candidates
# To do cosine_similarity, need same structure with q's
# Basically each q will be a matrix of repeated q's: num_tuples x num_candidates_in_tuple, all elts are q (repeated)

# This method constructs those matrices, use candidates=True for candidates matrix
def construct_qs_matrix_training(q_ids_sequential, lstm, h0, c0, word2vec, id2Data, dict_sequence_lengths,
                                 num_differing_questions, word_to_id_vocab, candidates=False):
    if not candidates:
        q_ids_complete = []
        for q in q_ids_sequential:
            q_ids_complete += [q] * dict_sequence_lengths[q]

    else:
        q_ids_complete = q_ids_sequential

    qs_matrix_list = []
    qs_seq_length = []

    for q in q_ids_complete:
        q_matrix_3d, q_num_words = get_question_matrix(q, id2Data, word_to_id_vocab, word2vec)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0))
    qs_hidden = lstm(qs_padded, (h0, c0))
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))
    qs_tuples = mean_pooled_h_qs.split(1 + num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)

    return final_matrix_tuples_by_constituent_qs_by_hidden_size


def construct_qs_matrix_testing_candidates(q_ids_in_order, lstm, h0, c0, word2vec, id2Data, word_to_id_vocab):
    qs_matrix_list = []
    qs_seq_length = []

    for q in q_ids_in_order:
        q_matrix_3d, q_num_words = get_question_matrix(q, id2Data, word_to_id_vocab, word2vec)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0), requires_grad=False)
    qs_hidden = lstm(qs_padded, (h0, c0))
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))

    return mean_pooled_h_qs


def construct_qs_matrix_testing_main(q_main_ids, lstm, h0, c0, word2vec, id2Data, word_to_id_vocab):
    all_mean_pooled_hiddens = []
    resulting_hidden_size = hidden_size * 2 if bidirectional else hidden_size
    initializing = True
    built_tensor_of_all_qs = None

    for (q, num_repetitions) in q_main_ids:
        q_matrix_3d, q_num_words = get_question_matrix(q, id2Data, word_to_id_vocab, word2vec)
        q_hidden = lstm(Variable(q_matrix_3d), (h0, c0))
        summed_h_q = torch.sum(q_hidden[0], dim=1)
        mean_pooled_h_q = torch.div(summed_h_q,
                                    torch.autograd.Variable(torch.FloatTensor([q_num_words] * resulting_hidden_size)))

        if initializing:
            built_tensor_of_all_qs = torch.cat([mean_pooled_h_q] * num_repetitions)
            initializing = False
        else:
            built_tensor_of_all_qs = torch.cat([built_tensor_of_all_qs] + [mean_pooled_h_q] * num_repetitions)

    return built_tensor_of_all_qs


# For categorization of questions by neural net, build a matrix of numq * lstm_hidden_layer_size
# Takes in list of q ids
# Matrix is to be fed as a batch to a neural network after being stacked with a similar matrix for another domain and compared to target
def construct_qs_matrix_domain_classification(domain_ids, lstm, h0, c0, word2vec, domain_specific_id_to_data, word_to_id_vocab):
    qs_matrix_list = []
    qs_seq_length = []

    for q in domain_ids:
        q_matrix_3d, q_num_words = get_question_matrix(q, domain_specific_id_to_data, word_to_id_vocab, word2vec)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0))
    # [ [num_q, num_word_per_q, hidden_size] i.e. all hidden, [1, num_q, hidden_size]  i.e. final hidden]:
    qs_hidden = lstm(qs_padded, (h0, c0))
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))

    return mean_pooled_h_qs