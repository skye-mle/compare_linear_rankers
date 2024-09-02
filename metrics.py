import numpy as np


def precision_at_k(predicted_scores, true_labels, k):
    indices = np.argsort(predicted_scores)[::-1][:k]
    sorted_true_labels = np.array(true_labels)[indices]
    relevant_items = np.sum(sorted_true_labels)
    precision = relevant_items / k
    return precision


def recall_at_k(predicted_scores, true_labels, k):
    indices = np.argsort(predicted_scores)[::-1][:k]
    sorted_true_labels = np.array(true_labels)[indices]
    relevant_items = np.sum(sorted_true_labels)
    total_relevant_items = np.sum(true_labels)
    recall = relevant_items / total_relevant_items
    return recall


def dcg_at_k(scores, k):
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def ndcg_at_k(predicted_scores, true_labels, k):
    indices = np.argsort(predicted_scores)[::-1][:k]
    sorted_true_labels = np.array(true_labels)[indices]
    dcg_max = dcg_at_k(sorted_true_labels, k)
    sorted_ideal_labels = np.sort(true_labels)[::-1][:k]
    idcg = dcg_at_k(sorted_ideal_labels, k)
    if not idcg:
        return 0.0
    return dcg_max / idcg


def evaluate_ranking_metrics(predicted_scores, true_labels, ks, precision=2):
    results = {}
    for k in ks:
        results[f"P@{k}"] = np.round(
            precision_at_k(predicted_scores, true_labels, k) * 100, precision
        )
        results[f"R@{k}"] = np.round(
            recall_at_k(predicted_scores, true_labels, k) * 100, precision
        )
        results[f"N@{k}"] = np.round(
            ndcg_at_k(predicted_scores, true_labels, k) * 100, precision
        )
    return results
