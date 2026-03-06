from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


class ModelTraining:

    def train_model(self, data, best_params):

        try:

            reader = Reader(rating_scale=(1, 5))

            dataset = Dataset.load_from_df(
                data[["UserID", "MovieID", "Rating"]],
                reader
            )

            trainset, testset = train_test_split(
                dataset,
                test_size=0.2,
                random_state=42
            )

            model = SVD(
                n_factors=best_params["n_factors"],
                n_epochs=best_params["n_epochs"],
                lr_all=best_params["lr_all"],
                reg_all=best_params["reg_all"]
            )

            model.fit(trainset)

            predictions = model.test(testset)

            rmse = accuracy.rmse(predictions)

            print(f"Model training completed with RMSE: {rmse}")

            return model, testset

        except Exception as e:
            raise Exception(e)