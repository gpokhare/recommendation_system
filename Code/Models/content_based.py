import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_df, genre_features):
        """
        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information; must have 'movieId'.
            genre_features (np.array): NumPy array of shape (num_movies, num_genres) with multi‑hot genre features.
        """
        self.movies_df = movies_df.copy()
        self.genre_features = genre_features
        self.user_profiles = {}  # user_id -> averaged genre vector for positive movies
        self.user_rated = {}     # user_id -> set of movieIds already seen

    def fit(self, train_df, threshold=4.0):
        """
        Build user content profiles from training data.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame with columns: userId, movieId, rating.
            threshold (float): Minimum rating to consider a movie positive.
        """
        # Build mapping from movieId to index in movies_df / genre_features
        movieid_to_index = {mid: idx for idx, mid in enumerate(self.movies_df['movieId'].values)}
        for _, row in train_df.iterrows():
            user = row['userId']
            movie = row['movieId']
            rating = row['rating']
            # Save all rated movies to later exclude them.
            self.user_rated.setdefault(user, set()).add(movie)
            # Only consider positive interactions.
            if rating >= threshold:
                self.user_profiles.setdefault(user, []).append(self.genre_features[movieid_to_index[movie]])
        # Average the positive vectors to create the user profile.
        for user, vectors in self.user_profiles.items():
            if vectors:
                self.user_profiles[user] = np.mean(vectors, axis=0).reshape(1, -1)
            else:
                self.user_profiles[user] = None
        return self

    def recommend(self, user_id, k=10):
        """
        Recommend top-k movies for the given user based on cosine similarity between the
        user's content profile and movie genre vectors.
        """
        # If no profile, recommend random movies.
        if user_id not in self.user_profiles or self.user_profiles[user_id] is None:
            candidate_movies = self.movies_df['movieId'].values.tolist()
            return list(np.random.choice(candidate_movies, k, replace=False))
        # Compute similarity between user profile and all movie genre vectors.
        user_profile = self.user_profiles[user_id]
        sims = cosine_similarity(user_profile, self.genre_features)[0]
        movie_ids = self.movies_df['movieId'].values
        # Exclude movies the user has already rated.
        rated = self.user_rated.get(user_id, set())
        candidates = [(movie_ids[i], sims[i]) for i in range(len(movie_ids)) if movie_ids[i] not in rated]
        candidates = sorted(candidates, key=lambda x: -x[1])
        recs = [movie for movie, _ in candidates[:k]]
        return recs

    def predict(self, user_id, movie_id):
        """
        Return the cosine similarity between the user's content profile and the movie's genre vector.
        """
        if user_id not in self.user_profiles or self.user_profiles[user_id] is None:
            return 0.0
        movieid_to_index = {mid: idx for idx, mid in enumerate(self.movies_df['movieId'].values)}
        if movie_id not in movieid_to_index:
            return 0.0
        idx = movieid_to_index[movie_id]
        movie_vector = self.genre_features[idx].reshape(1, -1)
        sim = cosine_similarity(self.user_profiles[user_id], movie_vector)[0][0]
        return sim
