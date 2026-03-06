import pandas as pd
import mlflow
import mlflow.sklearn

from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import SVD, KNNWithMeans, BaselineOnly
from surprise.model_selection import train_test_split

from src.logger import logging


def run_experiment():

    # MLflow tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # New experiment name
    mlflow.set_experiment("movie_recommendation_model_comparison")

    logging.info("Starting experiment pipeline")

    # Load processed dataset
    data = pd.read_csv("artifacts/processed_data.csv")
    data = data.sample(50000, random_state=42)

    logging.info("Processed data loaded")

    # Surprise dataset format
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['UserID', 'MovieID', 'Rating']], reader)

    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Models to compare
    models = {
        "SVD": SVD(),
        "BaselineOnly": BaselineOnly()
    }

    # Run experiments
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            logging.info(f"Training model: {model_name}")

            model.fit(trainset)

            predictions = model.test(testset)

            rmse = accuracy.rmse(predictions)

            # Log params
            mlflow.log_param("model", model_name)

            # Log metrics
            mlflow.log_metric("rmse", rmse)

            # Log model
            mlflow.sklearn.log_model(model, model_name)

            logging.info(f"{model_name} RMSE: {rmse}")

    logging.info("All experiments completed")


if __name__ == "__main__":
    run_experiment()