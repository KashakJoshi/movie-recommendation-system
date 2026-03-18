import os
import pickle
import pandas as pd
from pipeline.inference_pipeline import InferencePipeline


class RecommendationService:

    def __init__(self):

        # ---- MODEL PATH ----
        self.model_path = os.getenv("MODEL_PATH", "/app/model.pkl")

        if not os.path.exists(self.model_path):
            raise Exception(f"Model file not found at {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        # ---- DATA PATH ----
        self.data_path = os.getenv("DATA_PATH", "/app/api/ratings.csv")

        if not os.path.exists(self.data_path):
            raise Exception(f"Ratings file not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # ---- PIPELINE ----
        self.pipeline = InferencePipeline(self.model, self.df)

    def recommend(self, user_id: int, n: int = 5):

        recs = self.pipeline.get_recommendations(user_id, n)

        return {
            "user_id": user_id,
            "recommendations": recs
        }