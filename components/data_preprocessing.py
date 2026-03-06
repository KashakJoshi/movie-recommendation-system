from src.logger import logger
from src.exception import CustomException
import sys


class DataPreprocessing:

    def clean_data(self, ratings, movies, users):
        try:

            # removing duplicate rows
            ratings = ratings.drop_duplicates()
            
            # null value check
            if ratings.isnull().sum().sum() > 0:
                raise Exception("Null values found in ratings dataset")

            if movies.isnull().sum().sum() > 0:
                raise Exception("Null values found in movies dataset")

            if users.isnull().sum().sum() > 0:
                raise Exception("Null values found in users dataset")
            
            
            # merging datasets
            data = ratings.merge(movies, on="MovieID")
            data = data.merge(users, on="UserID")
            
            # Rating range check
            if not ratings["Rating"].between(1,5).all():
                raise Exception("Invalid rating values found")

            logger.info("Data cleaning completed")

            return data

        except Exception as e:
            raise CustomException(e, sys)