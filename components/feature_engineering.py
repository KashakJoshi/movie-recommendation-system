from src.logger import logger
from src.exception import CustomException
import sys
import os



class FeatureEngineering:

    def create_features(self, data):
        try:

            movie_popularity = data.groupby("MovieID")["Rating"].count().reset_index()
            movie_popularity.rename(columns={"Rating":"rating_count"}, inplace=True)

            movie_avg_rating = data.groupby("MovieID")["Rating"].mean().reset_index()
            movie_avg_rating.rename(columns={"Rating":"avg_rating"}, inplace=True)

            user_activity = data.groupby("UserID")["Rating"].count().reset_index()
            user_activity.rename(columns={"Rating":"user_rating_count"}, inplace=True)

            data = data.merge(movie_popularity, on="MovieID")
            data = data.merge(movie_avg_rating, on="MovieID")
            data = data.merge(user_activity, on="UserID")
            
            
            data.to_csv("artifacts/processed_data.csv", index=False)


            logger.info("Feature engineering completed")
            logger.info("Processed data saved to artifacts")

            return data

        except Exception as e:
            raise CustomException(e, sys)