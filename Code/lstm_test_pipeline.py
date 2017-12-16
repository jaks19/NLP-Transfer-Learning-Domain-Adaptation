from parameters import *

from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from domain_classifier_model import *
from meter import *

import torch
from torch.autograd import Variable


# Initialize the data sets
processed_corpus = process_whole_corpuses()
word_to_id_vocab = processed_corpus['word_to_id']
word2vec = load_glove_embeddings(glove_path, word_to_id_vocab)
android_id_to_data = processed_corpus['android_id_to_data']


''' Data Sets '''
test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())


''' Encoder (LSTM) '''
lstm = lstm = torch.load('../Pickle/adapt_first_attempt_epoch5.pt')

h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Procedural parameters '''
auc_scorer = AUCMeter()


def eval_model(lstm, ids, data, word2vec, id2Data, word_to_id_vocab):
    lstm.eval()
    auc_scorer.reset()

    candidate_ids, q_main_ids, labels = organize_test_ids(ids, data)
    num_q_main = len(q_main_ids)

    for i in range(num_q_main):
        q_main_id_num_repl_tuple = q_main_ids.pop(0)
        candidates = candidate_ids.pop(0)

        candidates_qs_matrix = construct_qs_matrix_testing(candidates, lstm, h0, c0, word2vec, id2Data,
        word_to_id_vocab, main=False)
        main_qs_matrix = construct_qs_matrix_testing([q_main_id_num_repl_tuple], lstm, h0, c0, word2vec, id2Data,
        word_to_id_vocab, main=True)

        similarity_matrix_this_batch = torch.nn.functional.cosine_similarity(candidates_qs_matrix, main_qs_matrix, eps=1e-08).data
        target_this_batch = torch.FloatTensor(labels.pop(0))
        auc_scorer.add(similarity_matrix_this_batch, target_this_batch)

    auc_score = auc_scorer.value()

    return auc_score


# Evaluate on dev set for AUC score
test_AUC_score = eval_model(lstm, test_question_ids_android, test_data_android, word2vec, android_id_to_data,
                            word_to_id_vocab)

print("Test AUC score:", test_AUC_score)
