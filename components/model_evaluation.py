from surprise import accuracy
from src.logger import logger
from src.exception import CustomException
import sys


class ModelEvaluation:

    def evaluate(self, model, testset):
        try:

            predictions = model.test(testset)

            rmse = accuracy.rmse(predictions)

            logger.info(f"Model RMSE: {rmse}")

            return rmse

        except Exception as e:
            raise CustomException(e, sys)