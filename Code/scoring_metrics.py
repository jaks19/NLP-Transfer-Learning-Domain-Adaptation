# Given a matrix of similarities, one row for each q_main
# Returns the MRR score for this set
# Note: each q_main may have anywhere from 0 to 20 positives, not including 20
def get_MRR_score(similarity_matrix, dict_pos):
    rows = similarity_matrix.split(1)
    reciprocal_ranks = []

    for row_index, r in enumerate(rows):
        pos_indices_this_row = dict_pos[row_index]
        lst_scores = list(r[0].data)
        rank = None
        lst_sorted_scores = sorted(lst_scores, reverse=True)

        for rk, score in enumerate(lst_sorted_scores):
            index_original = lst_scores.index(score)
            if index_original in pos_indices_this_row:
                rank = rk + 1
                reciprocal_ranks.append(1.0 / rank)
                break

    return sum(reciprocal_ranks) / len(reciprocal_ranks)