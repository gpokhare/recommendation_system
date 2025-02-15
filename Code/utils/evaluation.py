import numpy as np
import math
import random
import torch

def precision_at_k(recs, gt, k):
    """
    Compute Precision@K for one user.
    
    Args:
        recs (list or array): List of recommended movieIds.
        gt (set or list): Ground truth set of relevant movieIds.
        k (int): Number of recommendations to consider.
    
    Returns:
        Precision value.
    """
    recs = np.array(recs)[:k]
    # Create a boolean array indicating hits.
    hits = np.isin(recs, list(gt))
    return np.sum(hits) / k

def recall_at_k(recs, gt, k):
    """
    Compute Recall@K for one user.
    
    Args:
        recs (list or array): List of recommended movieIds.
        gt (set or list): Ground truth set of relevant movieIds.
        k (int): Number of recommendations.
    
    Returns:
        Recall value.
    """
    recs = np.array(recs)[:k]
    hits = np.isin(recs, list(gt))
    return np.sum(hits) / len(gt) if len(gt) > 0 else 0.0

def ndcg_at_k(recs, gt, k):
    """
    Compute NDCG@K for one user.
    
    Args:
        recs (list or array): List of recommended movieIds.
        gt (set or list): Ground truth set of relevant movieIds.
        k (int): Number of recommendations.
    
    Returns:
        NDCG value.
    """
    recs = np.array(recs)[:k]
    hits = np.isin(recs, list(gt)).astype(float)
    if hits.size == 0:
        return 0.0
    # Compute DCG: note ranks start at 1.
    ranks = np.arange(1, hits.size + 1)
    dcg = np.sum(hits / np.log2(ranks + 1))
    # Compute ideal DCG: assume ideal ranking of all 1's for min(|gt|, k) hits.
    ideal_hits = min(len(gt), k)
    idcg = np.sum(1.0 / np.log2(np.arange(1, ideal_hits + 1) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommender_metrics(recommend_fn, user_ids, test_df, k=10, threshold=4.0, sample_size=500):
    """
    Evaluate a recommender by computing Precision@K, Recall@K, and NDCG@K.
    This function samples a subset of users to speed up evaluation.
    
    Args:
        recommend_fn: Function taking a user_id and returning a list of recommended movieIds.
        user_ids: Iterable of user_ids to evaluate.
        test_df: Test DataFrame with columns: userId, movieId, rating.
        k (int): Number of recommendations.
        threshold (float): Minimum rating to consider a movie relevant.
        sample_size (int): Number of users to sample.
    
    Returns:
        Tuple of (avg_precision, avg_recall, avg_ndcg)
    """
    user_ids = list(user_ids)
    if sample_size < len(user_ids):
        user_ids = random.sample(user_ids, sample_size)
    
    # Build ground truth: user_id -> set of movieIds with rating >= threshold.
    ground_truth = {}
    for _, row in test_df.iterrows():
        if row['rating'] >= threshold:
            ground_truth.setdefault(row['userId'], set()).add(row['movieId'])
    
    precision_list = []
    recall_list = []
    ndcg_list = []
    
    for u in user_ids:
        recs = recommend_fn(u)
        if not recs:
            continue
        gt = ground_truth.get(u, set())
        if len(gt) == 0:
            continue
        precision_list.append(precision_at_k(recs, gt, k))
        recall_list.append(recall_at_k(recs, gt, k))
        ndcg_list.append(ndcg_at_k(recs, gt, k))
    
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    print(f"Avg Precision@{k}: {avg_precision:.4f}")
    print(f"Avg Recall@{k}: {avg_recall:.4f}")
    print(f"Avg NDCG@{k}: {avg_ndcg:.4f}")
    return avg_precision, avg_recall, avg_ndcg

def evaluate_rmse(predict_fn, test_df):
    """
    Evaluate RMSE for a prediction function on the test set.
    
    Args:
        predict_fn: Function that takes (user_id, movie_id) and returns a predicted rating.
        test_df: Test DataFrame with columns: userId, movieId, rating.
    
    Returns:
        RMSE value.
    """
    errors = []
    # Optionally, you can vectorize this if your predict_fn supports batch processing.
    for _, row in test_df.iterrows():
        pred = predict_fn(row['userId'], row['movieId'])
        errors.append((pred - row['rating']) ** 2)
    mse = np.mean(errors)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")
    return rmse
