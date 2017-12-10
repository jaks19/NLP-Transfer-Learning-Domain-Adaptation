from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from scoring_metrics import *

import torch
from torch.autograd import Variable
import time

saved_model_name = "transfer"
log_results = False
save_model = False


'''Hyperparams dashboard'''
dropout = 0.2
margin = 0.4
lr = 10**-3


''' Data Prep '''
glove_path = '../glove.840B.300d.txt'
android_corpus_path = '../android_dataset/corpus.tsv'
ubuntu_corpus_path = '../ubuntu_dataset/text_tokenized.txt'
WORD_TO_ID = 'word_to_id'
U_ID2DATA = 'ubuntu_id_to_data'
A_ID2DATA = 'android_id_to_data'

processed_corpus = process_whole_corpuses()
word_to_id_vocab = processed_corpus[WORD_TO_ID]
word2vec = load_glove_embeddings(glove_path, word_to_id_vocab)

ubuntu_id_to_data = processed_corpus[U_ID2DATA]
android_id_to_data = processed_corpus[A_ID2DATA]


'''Data Set'''
training_data_ubuntu = ubuntu_id_to_similar_different()
training_question_ids_ubuntu = list(training_data_ubuntu.keys())[:10]
dev_data_android = android_id_to_similar_different(dev=True)
dev_question_ids_android = list(dev_data_android.keys())[:2]
test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())[:10]


''' Model Specs '''
input_size = 300
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = True


lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

first_dim = num_layers * 2 if bidirectional else num_layers
h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Procedural parameters '''
batch_size = 2
num_differing_questions = 20
num_epochs = 10
num_batches = round(len(training_question_ids_ubuntu) / batch_size)


def train_model(lstm, optimizer, batch_ids, batch_data, word2vec, id2Data, word_to_id_vocab):
    lstm.train()
    optimizer.zero_grad()

    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, h0, c0, word2vec, id2Data,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, candidates=True)

    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, lstm, h0, c0, word2vec, id2Data,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix,
        dim=2, eps=1e-08)

    target = Variable(torch.LongTensor([0] * int(len(sequence_ids) / (1 + num_differing_questions))))
    loss_batch = loss_function(similarity_matrix, target)

    loss_batch.backward()
    optimizer.step()

    print("loss_on_batch:", loss_batch.data[0], " time_on_batch:", time.time() - start)
    return


def eval_model(lstm, ids, data, word2vec, id2Data, word_to_id_vocab):
    lstm.eval()
    sequence_ids, p_pluses_indices_dict = organize_test_ids(ids, data)

    candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, lstm, h0, c0, word2vec, id2Data,
                                                              num_differing_questions, word_to_id_vocab,
                                                              candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_testing(ids, lstm, h0, c0, word2vec, id2Data, num_differing_questions,
                                                        word_to_id_vocab, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2,
                                                              eps=1e-08)
    MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)
    return MRR_score


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches + 1):
        start = time.time()
        ids_this_batch_for_lstm = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        print("Working on batch #: ", batch)
        train_model(lstm, optimizer, ids_this_batch_for_lstm, training_data_ubuntu, word2vec, ubuntu_id_to_data,
                    word_to_id_vocab)

    # Evaluate on dev and test sets for MRR score
    dev_MRR_score = eval_model(lstm, dev_question_ids_android, dev_data_android, word2vec, android_id_to_data, word_to_id_vocab)
    test_MRR_score = eval_model(lstm, test_question_ids_android, test_data_android, word2vec, android_id_to_data, word_to_id_vocab)
    print("MRR score on dev set:", dev_MRR_score)

    # Log results to local logs.txt file
    if log_results:
        with open('logs.txt', 'a') as log_file:
            log_file.write('epoch: ' + str(epoch) + '\n')
            log_file.write('lr: ' + str(lr) +  ' marg: ' + str(margin) + ' drop: ' + str(dropout) + '\n' )
            log_file.write('dev_MRR: ' +  str(dev_MRR_score) + '\n')

    #Save model for this epoch
    if save_model:
        torch.save(lstm, '../Pickle/' + saved_model_name + '_epoch' + str(epoch) + '.pt')