import joblib
import pandas as pd
import os



class InferencePipeline:

    def __init__(self):

        # load trained model

        model_path = "artifacts/model/best_svd_model.pkl"

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
        # load movies dataset
        self.movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        encoding="latin1",
        names=["MovieID", "Title", "Genres"]
)


    def recommend_movies(self, user_id, movie_ids, top_n=5):

        predictions = []

        for movie_id in movie_ids:

            pred = self.model.predict(user_id, movie_id)

            predictions.append((movie_id, pred.est))


        predictions.sort(key=lambda x: x[1], reverse=True)

        top_movies = predictions[:top_n]

        recommendations = []

        for movie_id, score in top_movies:

            movie_name = self.movies[self.movies["MovieID"] == movie_id]["Title"].values[0]

            recommendations.append((movie_name, score))


        return recommendations


if __name__ == "__main__":

    pipeline = InferencePipeline()

    movie_ids = [1,2,3,4,5,6,7,8,9,10]

    recommendations = pipeline.recommend_movies(user_id=1, movie_ids=movie_ids)

    print("\nTop Recommendations:\n")

    for movie, score in recommendations:

        print(f"{movie} → Predicted Rating: {score:.2f}")