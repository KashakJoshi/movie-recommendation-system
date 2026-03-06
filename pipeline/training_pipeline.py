from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_preprocessing import DataPreprocessing
from components.feature_engineering import FeatureEngineering
from components.model_training import ModelTraining
from components.model_evaluation import ModelEvaluation
from components.model_saver import ModelSaver
from src.logger import logging
import numpy 
from components.hyperparameter_tuning import HyperparameterTuning
from surprise import Dataset, Reader


class TrainingPipeline:

    def start(self):

        ingestion = DataIngestion()
        ratings, movies, users = ingestion.load_data()

        validator = DataValidation()
        validator.validate(ratings, movies, users)

        preprocessing = DataPreprocessing()
        data = preprocessing.clean_data(ratings, movies, users)

        feature_engineering = FeatureEngineering()
        data = feature_engineering.create_features(data)
        
        reader = Reader(rating_scale=(1,5))

        dataset = Dataset.load_from_df(
            data[["UserID","MovieID","Rating"]],
            reader
        )

        tuner = HyperparameterTuning()
        best_params = tuner.tune(dataset)

        print("Best Params:", best_params)
        
        trainer = ModelTraining()
        model, testset = trainer.train_model(data)

        evaluator = ModelEvaluation()
        rmse = evaluator.evaluate(model, testset)

        saver = ModelSaver()
        saver.save_model(model)


        logging.info("pipeline successfully executed")

if __name__ == "__main__":

    pipeline = TrainingPipeline()
    pipeline.start()