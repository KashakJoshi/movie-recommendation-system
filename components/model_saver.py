import os
import pickle
from src.logger import logger
from src.exception import CustomException
import sys


class ModelSaver:

    def save_model(self, model):
        try:

            os.makedirs("models", exist_ok=True)

            with open("models/recommendation_model.pkl", "wb") as f:
                pickle.dump(model, f)

            logger.info("Model saved successfully")

        except Exception as e:
            raise CustomException(e, sys)