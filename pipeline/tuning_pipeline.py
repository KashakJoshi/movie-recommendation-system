import pandas as pd
import mlflow
import mlflow.sklearn

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

def run_tuning():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("movie_recommendation_tuning")

    data = pd.read_csv("artifacts/processed_data.csv")

    reader = Reader(rating_scale=(1,5))
    dataset = Dataset.load_from_df(data[['UserID','MovieID','Rating']], reader)

    trainset, testset = train_test_split(dataset, test_size=0.2)

    n_factors = [100, 120, 150]
    n_epochs = [30, 40]
    lr_all = [0.003, 0.005]
    reg_all = [0.02, 0.04]

    for f in n_factors:
        for e in n_epochs:
            for lr in lr_all:
                for reg in reg_all:

                    with mlflow.start_run():

                        model = SVD(
                            n_factors=f,
                            n_epochs=e,
                            lr_all=lr,
                            reg_all=reg
                        )

                        model.fit(trainset)

                        predictions = model.test(testset)

                        rmse = accuracy.rmse(predictions)

                        mlflow.log_param("n_factors", f)
                        mlflow.log_param("n_epochs", e)
                        mlflow.log_param("lr_all", lr)
                        mlflow.log_param("reg_all", reg)

                        mlflow.log_metric("rmse", rmse)

                        mlflow.sklearn.log_model(model, "svd_model")

                        print("RMSE:", rmse)


if __name__ == "__main__":
    run_tuning()