import joblib
import os

class InferencePipeline:

    def __init__(self):

        model_path = os.path.join(os.getcwd(), "model.pkl")

        self.model = joblib.load(model_path)

    def predict(self, user_id, movie_id):

        return self.model.predict(user_id, movie_id)