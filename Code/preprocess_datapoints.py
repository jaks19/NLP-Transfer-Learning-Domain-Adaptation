from collections import defaultdict

# For Direct Transfer, training:
# Ubuntu main question id mapped to [[similar_questions_ids], [different_questions_ids]]
def ubuntu_id_to_similar_different():
    filepath = "../ubuntu_dataset/train_random.txt"
    lines = open(filepath, encoding = 'utf8').readlines()

    training_data = {}
    for line in lines:
        id_similarids_diffids = line.split('\t')
        question_id = int(id_similarids_diffids[0])
        similar_ids = id_similarids_diffids[1].split(" ")
        different_ids = id_similarids_diffids[2].split(" ")

        for i in range(len(similar_ids)): similar_ids[i] = int(similar_ids[i])
        for j in range(len(different_ids)): different_ids[j] = int(different_ids[j])

        training_data[question_id] = [ similar_ids, different_ids ]
    return training_data


# For Direct Transfer eval, and testing:
# Android main question id mapped to [[similar_questions_ids], [different_questions_ids]]
# For dev set, use dev=True, for test set use dev=False
# Note: May have more than one similar and [similar_questions_ids] is a subset of [all_questions_ids]
def android_id_to_similar_different(dev=True):
    if dev:
        pos_filepath = '../android_dataset/dev.pos.txt'
        neg_filepath = '../android_dataset/dev.neg.txt'
    else:
        pos_filepath = '../android_dataset/test.pos.txt'
        neg_filepath = '../android_dataset/test.neg.txt'
        
    pos_lines = open(pos_filepath, encoding = 'utf8').readlines()
    neg_lines = open(neg_filepath, encoding = 'utf8').readlines()    
    
    evaluation_data = defaultdict(lambda: [[],[]])
    for lines_set in [pos_lines, neg_lines]:
        if lines_set == pos_lines: pos = True
        else: pos = False
            
        for line in lines_set:
            main_q_id = int(line.split()[0])
            if len(evaluation_data[main_q_id][1]) == 20: continue
                
            associated = int(line.split()[1])
            if pos: 
                evaluation_data[main_q_id][0].append(associated)
                evaluation_data[main_q_id][1].append(associated)
            else: 
                evaluation_data[main_q_id][1].append(associated)

    return evaluation_data

# training_data = ubuntu_id_to_similar_different():
# dev_data = android_id_to_similar_different(dev=True)
# test_data = android_id_to_similar_different(dev=False)