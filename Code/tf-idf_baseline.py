from parameters import *

from preprocess_datapoints import *
from preprocess_text_to_tensors import *
from meter import *

import torch
from torch.autograd import Variable

''' Data processing helpers only for TF-IDF implementation '''


def process_only_andoid_corpus():
    dataset_path = android_corpus_path
    all_txt = []
    android_id_to_data = {}

    lines = open(dataset_path).readlines()
    for line in lines:
        id_title_body_list = line.split('\t')
        idx = int(id_title_body_list[0])
        title_plus_body = id_title_body_list[1] + ' ' + id_title_body_list[2][:-1]
        all_txt.append(title_plus_body)
        android_id_to_data[idx] = title_plus_body

    vectorizer = CountVectorizer(binary=True, analyzer='word', max_df=0.1)
    vectorizer.fit(all_txt)

    return {
        'android_id_to_data': android_id_to_data,
        'vectorizer': vectorizer
    }


''' Data Sets '''
processed_corpus = process_only_andoid_corpus()
android_id_to_data = processed_corpus['android_id_to_data']
vectorizer = processed_corpus['vectorizer']

test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())

auc_scorer = AUCMeter()

''' Begin Evaluation'''
candidate_ids, q_main_ids, labels = organize_test_ids(test_question_ids_android, test_data_android)
list_of_scores = []

index_into_list_all_all_candidate_ids = 0
for q_main_id, num_associated_questions in q_main_ids:

    q_main_sentence = android_id_to_data[q_main_id]
    q_main_vector = torch.FloatTensor(vectorizer.transform([q_main_sentence]).toarray()[0]).unsqueeze(0)

    for k in range(num_associated_questions):
        q_candidate_id = candidate_ids[index_into_list_all_all_candidate_ids]
        q_candidate_sentence = android_id_to_data[q_candidate_id]
        q_candidate_vector = torch.FloatTensor(vectorizer.transform([q_candidate_sentence]).toarray()[0]).unsqueeze(0)

        score_cos_sim = torch.nn.functional.cosine_similarity(q_candidate_vector, q_main_vector)[0]
        list_of_scores.append(score_cos_sim)

        index_into_list_all_all_candidate_ids += 1

# print(list_of_scores)
target = torch.FloatTensor(labels)
auc_scorer.reset()
auc_scorer.add(torch.FloatTensor(list_of_scores), target)
auc_score = auc_scorer.value()

print("AUC Score using TF-IDF:", auc_score)