import os
import pandas as pd
from pipeline.inference_pipeline import InferencePipeline

#  loading model pipeline
pipeline = InferencePipeline()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
movies_path = os.path.join(BASE_DIR, "artifacts", "movies.csv")

movies_df = pd.read_csv(movies_path)


def get_recommendations(user_id: int):

    movie_ids = movies_df["MovieID"].unique().tolist()

    recommendations = pipeline.recommend_movies(user_id, movie_ids)

    results = []

    for rec in recommendations:

        movie_id = rec["movie_id"]
        score = rec["predicted_rating"]

        movie_name = movies_df[
            movies_df["MovieID"] == movie_id
        ]["Title"].values[0]

        results.append({
            "movie_id": int(movie_id),
            "movie_name": movie_name,
            "predicted_rating": round(float(score), 2)
        })

    return {
        "user_id": user_id,
        "recommendations": results
    }