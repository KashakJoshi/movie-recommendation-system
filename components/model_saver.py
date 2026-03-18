import joblib
import os


class ModelSaver:

    def save_model(self, model):

        os.makedirs("artifacts/model", exist_ok=True)

        joblib.dump(model, "model.pkl")

        print("Model saved successfully")