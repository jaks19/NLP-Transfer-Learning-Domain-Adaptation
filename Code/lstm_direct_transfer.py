from parameters import *

from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from domain_classifier import *
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
training_question_ids_ubuntu = list(training_data_ubuntu.keys())[:20]
dev_data_android = android_id_to_similar_different(dev=True)
dev_question_ids_android = list(dev_data_android.keys())[:20]
test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())[:10]
# Note: Remember to edit batch_size accordingly if testing on smaller size data sets


''' Encoder (LSTM) '''
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function_lstm = torch.nn.MultiMarginLoss(margin=margin)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr_lstm)

h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Domain Classifier (None because this model does Direct Transfer) '''
neural_net = DomainClassifier(input_size_nn, first_hidden_size_nn, second_hidden_size_nn)


''' Procedural parameters '''
batch_size = 2
num_epochs = 10
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
    candidate_ids, q_main_ids, labels = organize_test_ids(ids, data)

    candidates_qs_matrix = construct_qs_matrix_testing_candidates(candidate_ids, lstm, h0, c0, word2vec, id2Data,
        word_to_id_vocab)
    main_qs_matrix = construct_qs_matrix_testing_main(q_main_ids, lstm, h0, c0, word2vec, id2Data,
        word_to_id_vocab)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_matrix, main_qs_matrix, eps=1e-08)
    target = torch.FloatTensor(labels)
    auc_scorer.reset()
    auc_scorer.add(similarity_matrix.data, target)
    auc_score = auc_scorer.value()

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

        print("Time_on_batch:", time.time() - start)

    # Evaluate on dev set for AUC score
    dev_AUC_score = eval_model(lstm, dev_question_ids_android, dev_data_android, word2vec, android_id_to_data,
        word_to_id_vocab)
    test_AUC_score = eval_model(lstm, test_question_ids_android, test_data_android, word2vec, android_id_to_data,
        word_to_id_vocab)
    print("Dev AUC Score:", dev_AUC_score)
    print("Test AUC Score:", test_AUC_score)


    # Log results to local logs.txt file
    if log_results:
        with open('logs.txt', 'a') as log_file:
            log_file.write('epoch: ' + str(epoch) + '\n')
            log_file.write('lstm lr: ' + str(lr_lstm) + ' marg: ' + str(margin) + ' drop: ' + str(
                dropout) + '\n' + 'nn lr: ' + str(lr_nn) + 'lambda: ' + lamb)
            log_file.write('dev_MRR: ' + str(dev_MRR_score) + '\n')

    # Save model for this epoch
    if save_model:
        torch.save(lstm, '../Pickle/' + saved_model_name + '_epoch' + str(epoch) + '.pt')