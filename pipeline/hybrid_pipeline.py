import pandas as pd
import numpy as np
import mlflow

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def run_hybrid_experiment():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("movie_recommendation_hybrid")

    # ---------- Load data ----------
    ratings = pd.read_csv(
        "data/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )

    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin1"
    )

    # ---------- Collaborative filtering ----------
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["UserID", "MovieID", "Rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD(n_factors=50, n_epochs=20)
    model.fit(trainset)

    predictions = model.test(testset)

    svd_rmse = accuracy.rmse(predictions)

    # ---------- Content based ----------
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"))
    genre_matrix = vectorizer.fit_transform(movies["Genres"])

    similarity = cosine_similarity(genre_matrix)

    # ---------- Hybrid experiments ----------
    weights = [
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3)
    ]

    for w1, w2 in weights:

        hybrid_scores = []

        for pred in predictions:

            user = pred.uid
            movie = pred.iid
            svd_score = pred.est

            movie_index = movies[movies["MovieID"] == movie].index

            if len(movie_index) == 0:
                continue

            movie_index = movie_index[0]

            content_score = similarity[movie_index].mean()

            hybrid_score = w1 * svd_score + w2 * content_score

            hybrid_scores.append((pred.r_ui - hybrid_score) ** 2)

        hybrid_rmse = np.sqrt(np.mean(hybrid_scores))

        with mlflow.start_run():

            mlflow.log_param("svd_weight", w1)
            mlflow.log_param("content_weight", w2)

            mlflow.log_metric("svd_rmse", svd_rmse)
            mlflow.log_metric("hybrid_rmse", hybrid_rmse)

            print(f"\nWeights {w1}:{w2}")
            print("Hybrid RMSE:", hybrid_rmse)


if __name__ == "__main__":
    run_hybrid_experiment()