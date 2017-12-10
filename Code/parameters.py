''' Params Dashboard '''

''' Procedural parameters '''
batch_size = 2
num_differing_questions = 20
num_epochs = 10

saved_model_name = "domain_adaptlr3d2m4lr3lam4"
log_results = False
save_model = False


''' Model specs LSTM '''
dropout = 0.2
margin = 0.4
lr_lstm = 10**-3

input_size = 300
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = True
first_dim = num_layers * 2 if bidirectional else num_layers


''' Model specs NN '''
lr_nn = 10**-3
lamb = 10**-4

input_size_nn = 2*hidden_size if bidirectional else hidden_size
first_hidden_size_nn = 300
second_hidden_size_nn = 150


''' Data processing specs '''
truncate_length = 100
padding_idx = 0

glove_path = '../glove.840B.300d.txt'
android_corpus_path = '../android_dataset/corpus.tsv'
ubuntu_corpus_path = '../ubuntu_dataset/text_tokenized.txt'