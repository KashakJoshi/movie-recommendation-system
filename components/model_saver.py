import joblib
import os


class ModelSaver:

    def save_model(self, model):

        os.makedirs("artifacts/model", exist_ok=True)

        joblib.dump(model, "artifacts/model/best_svd_model.pkl")

        print("Model saved successfully")