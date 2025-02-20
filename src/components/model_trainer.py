import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


#Starting the model training part

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split input data into train and test")
            X_train, y_train, X_test,y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            #create a dictionary of models (no hyperparameter tuning yet)
            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
            }

            #hyperparameter tuning
            params = {
                "DecisionTree": {'criterion': ['squared_error', 'friedman_mae','absolute_error','poisson'],
                                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                },

                "RandomForest": {'n_estimators': [8,16,32,64,128,256],
                                },

                "GradientBoosting": {'n_estimators': [8,16,32,64,128,256],
                                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                     'subsample': [0.6, 0.7, 0.75,0.8,0.85,0.9],
                                    },

                "LinearRegression": {},
                "KNeighbors": {'n_neighbors': [5, 7, 9,11],
                              },
                "XGBoost": {'n_estimators': [8,16,32,64,128,256],
                            'learning_rate': [0.1, 0.01, 0.05, 0.001],
                            },
                "CatBoost": {'depth': [6, 8, 10],
                             'learning_rate': [0.1, 0.01, 0.5, 0.001],
                             'iterations': [30, 50, 100]
                            },
                "AdaBoost": {'n_estimators': [8,16,32,64,128,256],
                            'learning_rate': [0.1, 0.01, 0.5, 0.001],
                            }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            #Model with the highest r2 score
            best_model_score = max(sorted(model_report.values()))

            #save the model with the highest r2 score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            