import joblib
import os


class InferencePipeline:

    def __init__(self):

        # absolute path for render + local
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(BASE_DIR, "model.pkl")

        print("MODEL PATH:", model_path)

        if not os.path.exists(model_path):
            raise Exception(f"Model file not found at {model_path}")

        self.model = joblib.load(model_path)

    #  recommend function
    def recommend_movies(self, user_id, movie_ids):

        predictions = []

        for movie_id in movie_ids:

            pred = self.model.predict(user_id, movie_id)

            predictions.append({
                "MovieID": movie_id,
                "predicted_rating": pred.est
            })

        #  sorting by rating
        predictions = sorted(
            predictions,
            key=lambda x: x["predicted_rating"],
            reverse=True
        )

        #  top 10
        return predictions[:10]