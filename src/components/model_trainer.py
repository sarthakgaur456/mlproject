import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split the features and label in the training and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Linear Regressor': LinearRegression(),
                'Gradient Booster': GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


            model_scores = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            sorted_model_scores = dict(sorted(model_scores.items(), key = lambda x: x[1], reverse=True))
            best_model_name = list(sorted_model_scores.keys())[0]
            best_model_score = model_scores[best_model_name]
            
            if(best_model_score) < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Found out the best models for training and testing dataset')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model_name
            )

            return sorted_model_scores


        except Exception as e:
            raise CustomException(e, sys)