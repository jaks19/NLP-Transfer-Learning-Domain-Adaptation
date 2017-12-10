from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from scoring_metrics import *
from domain_classifier import *

import torch
from torch.autograd import Variable
import time

saved_model_name = "domain_adaptlr3d2m4lr3lam4"
log_results = False
save_model = False


'''Hyperparams dashboard'''
dropout = 0.2
margin = 0.4
lr_lstm = 10**-3

lr_nn = 10**-3
lamb = 10**-4

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
training_question_ids_ubuntu = list(training_data_ubuntu.keys())[:40]
dev_data_android = android_id_to_similar_different(dev=True)
dev_question_ids_android = list(dev_data_android.keys())[:40]
test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())[:10]


'''Model Specs LSTM'''
input_size = 300
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = True


lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function_lstm = torch.nn.MultiMarginLoss(margin=margin)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr_lstm)

first_dim = num_layers * 2 if bidirectional else num_layers
h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Model Specs NN'''
input_size_nn = 2*hidden_size if bidirectional else hidden_size
first_hidden_size_nn = 300
second_hidden_size_nn = 150

neural_net = DomainClassifier(input_size_nn, first_hidden_size_nn, second_hidden_size_nn)
loss_function_nn = nn.CrossEntropyLoss()
optimizer_nn = torch.optim.Adam(neural_net.parameters(), lr=lr_nn)


''' Procedural parameters '''
batch_size = 2
num_differing_questions = 20
num_epochs = 10
num_batches = round(len(training_question_ids_ubuntu) / batch_size)


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


def train_nn_domain_classification(neural_net, lstm, h0, c0, ids_ubuntu, ids_android, word2vec,
    ubuntu_id_to_data, android_id_to_data):
    neural_net.train()
    lstm.train()

    qs_matrix_ubuntu = construct_qs_matrix_domain_classification(ids_ubuntu, lstm, h0, c0, word2vec,
        ubuntu_id_to_data, word_to_id_vocab)
    qs_matrix_android = construct_qs_matrix_domain_classification(ids_android, lstm, h0, c0, word2vec,
        android_id_to_data, word_to_id_vocab)
    overall_qs_matrix = torch.cat([qs_matrix_ubuntu, qs_matrix_android])

    out = neural_net.forward(overall_qs_matrix)
    target_vector = Variable(torch.LongTensor(torch.cat([torch.zeros(20), torch.ones(20)]).numpy()))
    loss_batch = loss_function_nn(out, target_vector)

    print("Neural net cross-entropy loss on batch:", loss_batch.data[0])
    return loss_batch


def eval_model(lstm, ids, data, word2vec, id2Data, word_to_id_vocab):
    lstm.eval()
    sequence_ids, p_pluses_indices_dict = organize_test_ids(ids, data)

    candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, lstm, h0, c0, word2vec, id2Data,
        num_differing_questions, word_to_id_vocab, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_testing(ids, lstm, h0, c0, word2vec, id2Data, num_differing_questions,
        word_to_id_vocab, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix,
        dim=2, eps=1e-08)
    MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)
    return MRR_score


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches + 1):
        start = time.time()
        optimizer_lstm.zero_grad()
        optimizer_nn.zero_grad()
        print("Working on batch #: ", batch)

        # Train on ubuntu similar question retrieval
        ids_this_batch_for_lstm = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        loss_batch_similarity = train_lstm_question_similarity(lstm, ids_this_batch_for_lstm,
            training_data_ubuntu, word2vec, ubuntu_id_to_data, word_to_id_vocab)

        # Train on ubuntu-android domain classification task
        ids_randomized_ubuntu = get_20_random_ids(training_question_ids_ubuntu)
        ids_randomized_android = get_20_random_ids(dev_question_ids_android)
        loss_batch_domain_classification = train_nn_domain_classification(neural_net, lstm, h0, c0,
            ids_randomized_ubuntu, ids_randomized_android, word2vec, ubuntu_id_to_data, android_id_to_data)

        # Overall loss = multi-margin loss - LAMBDA * cross entropy loss
        overall_loss = loss_batch_similarity - (lamb * loss_batch_domain_classification)
        overall_loss.backward()
        optimizer_lstm.step()
        optimizer_nn.step()

        print("Time_on_batch:", time.time() - start)

    # Evaluate on dev set for MRR score
    dev_MRR_score = eval_model(lstm, dev_question_ids_android, dev_data_android, word2vec, android_id_to_data,
        word_to_id_vocab)
    test_MRR_score = eval_model(lstm, test_question_ids_android, test_data_android, word2vec, android_id_to_data,
        word_to_id_vocab)
    print("MRR score on dev set:", dev_MRR_score)

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