from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_preprocessing import DataPreprocessing
from components.feature_engineering import FeatureEngineering
from components.model_training import ModelTraining
from components.model_evaluation import ModelEvaluation
from components.model_saver import ModelSaver
from components.hyperparameter_tuning import HyperparameterTuning

from src.logger import logging

from surprise import Dataset, Reader


class TrainingPipeline:

    def start(self):

        # 1 Data Ingestion
        ingestion = DataIngestion()
        ratings, movies, users = ingestion.load_data()


        # 2 Data Validation
        validator = DataValidation()
        validator.validate(ratings, movies, users)


        # 3 Data Preprocessing
        preprocessing = DataPreprocessing()
        data = preprocessing.clean_data(ratings, movies, users)


        # 4 Feature Engineering
        feature_engineering = FeatureEngineering()
        data = feature_engineering.create_features(data)


        # 5 Convert to Surprise Dataset
        reader = Reader(rating_scale=(1, 5))

        dataset = Dataset.load_from_df(
            data[["UserID", "MovieID", "Rating"]],
            reader
        )


        # 6 Hyperparameter Tuning
        best_params = {
        "n_factors": 50,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02
}


        # 7 Model Training
        trainer = ModelTraining()
        model, testset = trainer.train_model(data, best_params)

        # 8 Model Evaluation
        evaluator = ModelEvaluation()
        rmse = evaluator.evaluate(model, testset)

        print("Final RMSE:", rmse)


        # 9 Model Saving
        saver = ModelSaver()
        saver.save_model(model)


        logging.info("Pipeline successfully executed")


if __name__ == "__main__":

    pipeline = TrainingPipeline()
    pipeline.start()