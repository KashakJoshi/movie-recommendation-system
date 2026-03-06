import os
import pandas as pd
from src.logger import logger
from src.exception import CustomException
import sys


class DataIngestion:

    def load_data(self):
        try:

            # reading raw data
            ratings = pd.read_csv(
                "data/ratings.dat",
                sep="::",
                engine="python",
                names=["UserID","MovieID","Rating","Timestamp"],
                encoding="latin-1"
            )

            movies = pd.read_csv(
                "data/movies.dat",
                sep="::",
                engine="python",
                names=["MovieID","Title","Genres"],
                encoding="latin-1"
            )

            users = pd.read_csv(
                "data/users.dat",
                sep="::",
                engine="python",
                names=["UserID","Gender","Age","Occupation","Zip-code"],
                encoding="latin-1"
            )

            # creating artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # saving datasets
            ratings.to_csv("artifacts/ratings.csv", index=False)
            movies.to_csv("artifacts/movies.csv", index=False)
            users.to_csv("artifacts/users.csv", index=False)

            logger.info("Data ingestion completed and saved to artifacts")
            return ratings, movies, users

        except Exception as e:
            raise CustomException(e, sys)