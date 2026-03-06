from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException
import sys


class ModelTraining:

    def train_model(self, data):
        try:

            reader = Reader(rating_scale=(1,5))

            dataset = Dataset.load_from_df(
                data[["UserID","MovieID","Rating"]],
                reader
            )

            trainset, testset = train_test_split(dataset, test_size=0.2)

            model = SVD()

            model.fit(trainset)

            logger.info("Model training completed")

            return model, testset

        except Exception as e:
            raise CustomException(e, sys)