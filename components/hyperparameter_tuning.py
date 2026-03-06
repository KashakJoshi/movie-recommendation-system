import yaml
from surprise import SVD
from surprise.model_selection import GridSearchCV
from src.logger import logger
from src.exception import CustomException
import sys


class HyperparameterTuning:

    def tune(self, dataset):
        try:

            # read params
            with open("config/params.yaml","r") as file:
                params = yaml.safe_load(file)

            param_grid = params["svd"]

            grid_search = GridSearchCV(
                SVD,
                param_grid,
                measures=["rmse"],
                cv=3
            )

            grid_search.fit(dataset)

            best_params = grid_search.best_params["rmse"]

            logger.info(f"Best parameters: {best_params}")

            return best_params

        except Exception as e:
            raise CustomException(e, sys)