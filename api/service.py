import pandas as pd
from pipeline.inference_pipeline import InferencePipeline

# load data
ratings_df = pd.read_csv("api/ratings.csv")
movies_df = pd.read_csv("data/movies.csv")

# load model pipeline
pipeline = InferencePipeline()


def get_recommendations(user_id: int, n_recommendations: int = 5):

    # movies already rated by user
    user_movies = ratings_df[ratings_df["UserID"] == user_id]["MovieID"].tolist()

    # all movies
    all_movies = movies_df["movieId"].tolist()

    # candidate movies
    candidate_movies = [m for m in all_movies if m not in user_movies]

    predictions = []

    for movie in candidate_movies:
        pred = pipeline.predict(user_id, movie)
        pred_rating = pred.est   # ⭐ VERY IMPORTANT FIX

        predictions.append({
            "movieId": movie,
            "predicted_rating": float(pred_rating)
        })

    # sort by rating
    predictions = sorted(predictions, key=lambda x: x["predicted_rating"], reverse=True)

    # top N
    top_movies = predictions[:n_recommendations]

    # attach movie titles
    results = []

    for item in top_movies:
        movie_id = item["movieId"]

        title = movies_df[movies_df["movieId"] == movie_id]["title"].values

        if len(title) > 0:
            title = title[0]
        else:
            title = "Unknown"

        results.append({
            "movieId": movie_id,
            "title": title,
            "predicted_rating": item["predicted_rating"]
        })

    return {
        "user_id": user_id,
        "recommendations": results
    }