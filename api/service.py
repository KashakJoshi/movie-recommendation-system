from pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()


def get_recommendations(user_id: int):

    movie_ids = list(range(1, 200))

    recommendations = pipeline.recommend_movies(user_id, movie_ids)

    results = []

    for movie, score in recommendations:
        results.append({
            "movie": movie,
            "predicted_rating": round(score, 2)
        })

    return {"recommendations": results}