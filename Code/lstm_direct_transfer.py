from parameters import *

from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from domain_classifier_model import *
from meter import *

import torch
from torch.autograd import Variable
import time


# Initialize the data sets
processed_corpus = process_whole_corpuses()
word_to_id_vocab = processed_corpus['word_to_id']
word2vec = load_glove_embeddings(glove_path, word_to_id_vocab)
ubuntu_id_to_data = processed_corpus['ubuntu_id_to_data']
android_id_to_data = processed_corpus['android_id_to_data']


''' Data Sets '''
training_data_ubuntu = ubuntu_id_to_similar_different()
training_question_ids_ubuntu = list(training_data_ubuntu.keys())
dev_data_android = android_id_to_similar_different(dev=True)
dev_question_ids_android = list(dev_data_android.keys())
# Note: Remember to edit batch_size accordingly if testing on smaller size data sets


''' Encoder (LSTM) '''
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function_lstm = torch.nn.MultiMarginLoss(margin=margin)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr_lstm)

h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Procedural parameters '''
num_batches = round(len(training_question_ids_ubuntu) / batch_size)
auc_scorer = AUCMeter()


def train_lstm_question_similarity(lstm, batch_ids, batch_data, word2vec, id2Data, word_to_id_vocab):
    lstm.train()
    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, h0, c0, word2vec, id2Data,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, lstm, h0, c0, word2vec, id2Data,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)
    target = Variable(torch.LongTensor([0] * int(len(sequence_ids) / (1 + num_differing_questions))))
    loss_batch = loss_function_lstm(similarity_matrix, target)

    print("lstm multi-margin loss on batch:", loss_batch.data[0])
    return loss_batch


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

    print("Dev AUC score:", auc_score)
    return auc_score


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches + 1):
        start = time.time()
        optimizer_lstm.zero_grad()
        print("Working on batch #: ", batch)

        # Train on ubuntu similar question retrieval
        ids_this_batch_for_lstm = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        loss_batch_similarity = train_lstm_question_similarity(lstm, ids_this_batch_for_lstm,
        training_data_ubuntu, word2vec, ubuntu_id_to_data, word_to_id_vocab)

        overall_loss = loss_batch_similarity
        overall_loss.backward()
        optimizer_lstm.step()

        print("Time on batch:", time.time() - start)

    # Evaluate on dev set for AUC score
    dev_AUC_score = eval_model(lstm, dev_question_ids_android, dev_data_android, word2vec, android_id_to_data,
                               word_to_id_vocab)