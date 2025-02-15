import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np 

import math


class NeuMFHybrid(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim, mlp_layers=[64, 32, 16, 8]):
        super(NeuMFHybrid, self).__init__()
        # GMF Branch
        self.gmf_user = nn.Embedding(num_users, embedding_dim)
        self.gmf_movie = nn.Embedding(num_movies, embedding_dim)
        self.gmf_genre = nn.Linear(num_genres, embedding_dim)
        # MLP Branch
        self.mlp_user = nn.Embedding(num_users, embedding_dim)
        self.mlp_movie = nn.Embedding(num_movies, embedding_dim)
        self.mlp_genre = nn.Linear(num_genres, embedding_dim)
        mlp_input_dim = 3 * embedding_dim
        self.mlp_fc_layers = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_layers[0]),
            nn.ReLU(),
            nn.Linear(mlp_layers[0], mlp_layers[1]),
            nn.ReLU(),
            nn.Linear(mlp_layers[1], mlp_layers[2]),
            nn.ReLU(),
            nn.Linear(mlp_layers[2], mlp_layers[3]),
            nn.ReLU()
        )
        final_input_dim = embedding_dim + mlp_layers[-1]
        self.final_layer = nn.Linear(final_input_dim, 1)

    def forward(self, user_idx, movie_idx, genre_vector):
        # GMF branch
        gmf_u = self.gmf_user(user_idx)
        gmf_m = self.gmf_movie(movie_idx)
        gmf_g = F.relu(self.gmf_genre(genre_vector))
        gmf_output = gmf_u * gmf_m * gmf_g
        # MLP branch
        mlp_u = self.mlp_user(user_idx)
        mlp_m = self.mlp_movie(movie_idx)
        mlp_g = F.relu(self.mlp_genre(genre_vector))
        mlp_input = torch.cat([mlp_u, mlp_m, mlp_g], dim=1)
        mlp_output = self.mlp_fc_layers(mlp_input)
        # Combine branches
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        rating = self.final_layer(combined)
        return rating.squeeze()

    def fit(self, train_loader, optimizer, criterion, num_epochs, device):
        """
        Standard training loop using MSE loss.
        """
        self.to(device)
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for user, movie, target, genre_vec in train_loader:
                user = user.to(device)
                movie = movie.to(device)
                target = target.to(device)
                genre_vec = genre_vec.to(device)
                optimizer.zero_grad()
                output = self(user, movie, genre_vec)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * user.size(0)
            print(f"NeuMFHybrid Epoch {epoch+1}/{num_epochs}: Loss {total_loss/len(train_loader.dataset):.4f}")
        return self

    def fit_bpr(self, train_loader, positive_user_items, all_movie_indices, inv_movie2index, movieid_to_index,
                optimizer, num_epochs, device, genre_features):
        """
        Training loop using Bayesian Personalized Ranking (BPR) loss.
        For each positive sample, a negative movie (one that the user did not rate positively)
        is sampled. Loss is computed as:
            loss = -log(sigmoid(pos_score - neg_score))
        
        Args:
            train_loader: DataLoader providing batches of (user, pos_movie, target, genre_vector).
            positive_user_items (dict): Mapping from user index to set of positive movie indices.
            all_movie_indices (list): List of all movie indices.
            inv_movie2index (dict): Inverse mapping from movie index to original movie ID.
            movieid_to_index (dict): Mapping from original movie ID to row index in movies_df.
            optimizer: Optimizer.
            num_epochs: Number of epochs.
            device: torch.device.
            genre_features: NumPy array of genre features (with same ordering as in movies_df).
        
        Returns:
            self
        """
        self.to(device)
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for user, pos_movie, target, genre_vec in train_loader:
                user = user.to(device)
                pos_movie = pos_movie.to(device)
                genre_vec = genre_vec.to(device)
                batch_size = user.size(0)
                optimizer.zero_grad()
                # Sample one negative movie per sample in batch
                neg_movies = []
                neg_genre_vectors = []
                for i in range(batch_size):
                    u = user[i].item()
                    pos_set = positive_user_items.get(u, set())
                    candidates = [m for m in all_movie_indices if m not in pos_set]
                    if len(candidates) == 0:
                        cand = random.choice(all_movie_indices)
                    else:
                        cand = random.choice(candidates)
                    neg_movies.append(cand)
                    neg_movie_id = inv_movie2index[cand]
                    neg_genre_vectors.append(genre_features[movieid_to_index[neg_movie_id]])
                neg_movie_tensor = torch.tensor(neg_movies, dtype=torch.long, device=device)
                neg_genre_tensor = torch.tensor(np.array(neg_genre_vectors), dtype=torch.float, device=device)
                # Get predictions for positive and negative samples.
                pos_scores = self(user, pos_movie, genre_vec)
                neg_scores = self(user, neg_movie_tensor, neg_genre_tensor)
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"NeuMFHybrid BPR Epoch {epoch+1}/{num_epochs}: Loss {avg_loss:.4f}")
        return self

    def predict(self, user_tensor, movie_tensor, genre_tensor):
        self.eval()
        with torch.no_grad():
            return self(user_tensor, movie_tensor, genre_tensor)



def evaluate_ranking_neumf(model, test_loader, k, device):
    """
    Evaluate the NeuMFHybrid model using precision@k, recall@k, and NDCG@k.
    
    Args:
        model (NeuMFHybrid): The trained NeuMFHybrid model.
        test_loader (DataLoader): DataLoader yielding (user, movie, label, genre_vector).
        k (int): The cutoff for precision@k, recall@k, and NDCG@k.
        device (torch.device): The torch device.
    
    Returns:
        avg_precision (float): Average precision@k over all users.
        avg_recall (float): Average recall@k over all users.
        avg_ndcg (float): Average NDCG@k over all users.
    """
    # Dictionaries to store predictions and true labels per user.
    user_scores = {}
    user_labels = {}

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            user, movie, label, genre_vec = batch
            user = user.to(device)
            movie = movie.to(device)
            genre_vec = genre_vec.to(device)
            
            # Get predictions from the model.
            preds = model(user, movie, genre_vec)
            
            # Move tensors to CPU.
            users_np = user.cpu().numpy()
            preds_np = preds.cpu().numpy()
            labels_np = label.cpu().numpy()  # binary labels
            
            # Group predictions and labels by user.
            for u, score, true_label in zip(users_np, preds_np, labels_np):
                if u not in user_scores:
                    user_scores[u] = []
                    user_labels[u] = []
                user_scores[u].append(score)
                user_labels[u].append(true_label)
    
    precisions, recalls, ndcgs = [], [], []
    
    # Function to compute DCG for a list of binary relevances.
    def dcg_at_k(relevances, k):
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            # If the item is relevant, contribute 1/log2(i+2)
            if relevances[i] == 1:
                dcg += 1.0 / math.log2(i + 2)
        return dcg

    # Compute metrics for each user.
    for u in user_scores:
        scores = np.array(user_scores[u])
        labels = np.array(user_labels[u])
        
        # Skip users with no positive samples.
        if labels.sum() == 0:
            continue
        
        # Sort predictions in descending order.
        sorted_indices = np.argsort(-scores)
        top_k_labels = labels[sorted_indices][:k]
        
        # Precision@k: relevant items in top-k divided by k.
        precision = np.sum(top_k_labels) / k
        
        # Recall@k: relevant items in top-k divided by total number of relevant items.
        recall = np.sum(top_k_labels) / np.sum(labels)
        
        
        # NDCG@k:
        dcg = dcg_at_k(top_k_labels, k)
        # Ideal DCG: sort labels in descending order (all positives at the top)
        ideal_labels = np.sort(labels)[::-1]
        idcg = dcg_at_k(ideal_labels, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    
    return avg_precision, avg_recall, avg_ndcg

