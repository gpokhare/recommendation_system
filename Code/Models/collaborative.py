import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.ratings_pivot = None
        self.user_similarity = None
        self.user_ids = None
        self.movie_ids = None

    def fit(self, train_df):
        """
        Build the user–item matrix and compute user–user cosine similarity.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame with columns: userId, movieId, rating.
        """
        self.ratings_pivot = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_similarity = cosine_similarity(self.ratings_pivot.values)
        self.user_ids = self.ratings_pivot.index.tolist()
        self.movie_ids = self.ratings_pivot.columns.tolist()
        return self

    def recommend(self, user_id, k=10):
        """
        Recommend top-k movies for the given user based on weighted average of ratings from similar users.
        """
        if user_id not in self.user_ids:
            return []
        user_idx = self.user_ids.index(user_id)
        sim_scores = self.user_similarity[user_idx]
        weighted_ratings = np.dot(sim_scores, self.ratings_pivot.values)
        norm_factor = np.sum(sim_scores) if np.sum(sim_scores) != 0 else 1
        user_pred = weighted_ratings / norm_factor
        # Exclude movies already rated by the user.
        rated = set(self.ratings_pivot.columns[self.ratings_pivot.iloc[user_idx] > 0])
        recs = []
        for i, movie in enumerate(self.movie_ids):
            if movie not in rated:
                recs.append((movie, user_pred[i]))
        recs = sorted(recs, key=lambda x: -x[1])
        return [movie for movie, _ in recs[:k]]

    def predict(self, user_id, movie_id):
        """
        Predict a rating for a given user and movie.
        """
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return 0.0
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        sim_scores = self.user_similarity[user_idx]
        weighted_ratings = np.dot(sim_scores, self.ratings_pivot.values)
        norm_factor = np.sum(sim_scores) if np.sum(sim_scores) != 0 else 1
        user_pred = weighted_ratings / norm_factor
        return user_pred[movie_idx]
