import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np 
import math

class HybridRecSys(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim):
        """
        A simple hybrid recommender that integrates collaborative filtering and
        content-based (genre) information.
        
        Args:
            num_users (int): Number of users.
            num_movies (int): Number of movies.
            num_genres (int): Dimension of the multi-hot genre vector.
            embedding_dim (int): Embedding dimension.
        """
        super(HybridRecSys, self).__init__()
        # Embeddings for users and movies (collaborative filtering)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        # Linear layer to project genre features into embedding space
        self.genre_fc = nn.Linear(num_genres, embedding_dim)
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(3 * embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, user_idx, movie_idx, genre_vector):
        """
        Forward pass.
        
        Args:
            user_idx (Tensor): Tensor of user indices.
            movie_idx (Tensor): Tensor of movie indices.
            genre_vector (Tensor): Tensor of genre multi-hot vectors.
            
        Returns:
            Tensor: Predicted (normalized) rating.
        """
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        genre_emb = F.relu(self.genre_fc(genre_vector))
        augmented_movie = torch.cat([movie_emb, genre_emb], dim=1)
        x = torch.cat([user_emb, augmented_movie], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        rating = self.fc3(x)
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
            print(f"HybridRecSys Epoch {epoch+1}/{num_epochs}: Loss {total_loss/len(train_loader.dataset):.4f}")
        return self

    def fit_bpr(self, train_loader, positive_user_items, all_movie_indices, inv_movie2index, movieid_to_index, optimizer, num_epochs, device, genre_features):
        """
        Training loop that optimizes for ranking using Bayesian Personalized Ranking (BPR) loss.
        
        For each positive sample in the batch, a negative movie (one that the user did not rate positively)
        is sampled. Then, the loss is computed as:
            loss = -log(sigmoid(pos_score - neg_score))
        
        Args:
            train_loader: DataLoader providing batches of (user, pos_movie, target, genre_vector).
            positive_user_items (dict): Mapping from user index to set of positive movie indices.
            all_movie_indices (list): List of all movie indices.
            inv_movie2index (dict): Inverse mapping from movie index to original movie ID.
            movieid_to_index (dict): Mapping from original movie ID to row index in movies_df (for genre features).
            optimizer: Optimizer for training.
            num_epochs (int): Number of epochs.
            device: torch.device (e.g., "mps").
            genre_features: NumPy array of genre features (same ordering as in movies_df).
            
        Returns:
            The trained model.
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
                # For each sample in the batch, sample one negative movie.
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
                    # Get original movie ID for candidate negative.
                    neg_movie_id = inv_movie2index[cand]
                    neg_genre_vectors.append(genre_features[movieid_to_index[neg_movie_id]])
                neg_movie_tensor = torch.tensor(neg_movies, dtype=torch.long, device=device)
                neg_genre_tensor = torch.tensor(np.array(neg_genre_vectors), dtype=torch.float, device=device)
                # Get predictions for positive and negative items.
                pos_scores = self(user, pos_movie, genre_vec)  # shape: (batch_size,)
                neg_scores = self(user, neg_movie_tensor, neg_genre_tensor)  # shape: (batch_size,)
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"HybridRecSys BPR Epoch {epoch+1}/{num_epochs}: Loss {avg_loss:.4f}")
        return self

    def predict(self, user_tensor, movie_tensor, genre_tensor):
        self.eval()
        with torch.no_grad():
            return self(user_tensor, movie_tensor, genre_tensor)


def evaluate_hybrid_ranking(model, test_loader, k, device):
    """
    Evaluate the HybridRecSys model with precision@k, recall@k, and NDCG@k.
    
    Args:
        model: The trained HybridRecSys model.
        test_loader: DataLoader yielding (user, movie, label, genre_vector).
        k (int): The cutoff for precision@k, recall@k, and NDCG@k.
        device: The torch device (e.g., "cuda" or "cpu").
        
    Returns:
        avg_precision (float): Average precision@k over all users.
        avg_recall (float): Average recall@k over all users.
        avg_ndcg (float): Average NDCG@k over all users.
    """
    
    # Dictionaries to accumulate predictions and true labels per user.
    user_scores = {}
    user_labels = {}

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            user, movie, label, genre_vec = batch
            user = user.to(device)
            movie = movie.to(device)
            genre_vec = genre_vec.to(device)
            
            # Get model predictions for this batch.
            preds = model(user, movie, genre_vec)  # shape: (batch_size,)
            
            # Move tensors to CPU and convert to numpy arrays.
            users_np = user.cpu().numpy()
            preds_np = preds.cpu().numpy()
            labels_np = label.cpu().numpy()  # assuming label is a tensor of 0/1
            
            # Group the predictions and labels by user.
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
            if relevances[i] == 1:
                dcg += 1.0 / math.log2(i + 2)
        return dcg

    # Compute metrics per user.
    for u in user_scores:
        scores = np.array(user_scores[u])
        labels = np.array(user_labels[u])
        
        # Skip users with no positive items to avoid division by zero.
        if labels.sum() == 0:
            continue
        
        # Get indices that would sort the scores in descending order.
        sorted_indices = np.argsort(-scores)
        top_k_labels = labels[sorted_indices][:k]
        
        # Precision@k: # of relevant items in top-k divided by k.
        precision = np.sum(top_k_labels) / k
        
        # Recall@k: # of relevant items in top-k divided by total # of relevant items.
        recall = np.sum(top_k_labels) / np.sum(labels)
        
 
        
        # Compute NDCG@k.
        dcg = dcg_at_k(top_k_labels, k)
        # Ideal DCG: the best possible ranking where all positives are at the top.
        ideal_labels = np.sort(labels)[::-1]
        idcg = dcg_at_k(ideal_labels, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    # Average metrics over all users.
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    
    return avg_precision, avg_recall, avg_ndcg
