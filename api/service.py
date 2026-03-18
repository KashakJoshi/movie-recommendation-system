import pandas as pd
from pipeline.inference_pipeline import InferencePipeline


# load data once
ratings_df = pd.read_csv("api/ratings.csv")

pipeline = InferencePipeline()


def get_recommendations(user_id: int, n_recommendations: int = 5):

    # get movies already rated
    user_movies = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()

    all_movies = ratings_df["movieId"].unique().tolist()

    # candidate movies
    candidate_movies = [m for m in all_movies if m not in user_movies]

    predictions = []

    for movie in candidate_movies:
        score = pipeline.predict(user_id, movie)
        predictions.append((movie, score))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = [movie for movie, _ in predictions[:n_recommendations]]

    return top_movies