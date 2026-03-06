from src.logger import logger
from src.exception import CustomException
import sys


class DataValidation:

    def validate(self, ratings, movies, users):
        try:

            # column check
            required_ratings_cols = {"UserID","MovieID","Rating","Timestamp"}
            if not required_ratings_cols.issubset(ratings.columns):
                raise Exception("Ratings columns mismatch")

            # data type check
            if ratings["Rating"].dtype not in ["int64","float64"]:
                raise Exception("Rating datatype incorrect")

            # duplicate check
            duplicates = ratings.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"{duplicates} duplicate rows found")

            logger.info("Data validation completed")

            return True

        except Exception as e:
            raise CustomException(e, sys)