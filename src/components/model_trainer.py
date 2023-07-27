import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# from src.constant import *
from src.exception import CustomException
from src.logger import logging
# from src.utils import evaluate_models, load_object, save_object, upload_file
from src.utils import evaluate_models, load_object, save_object

# Using a data class to store the path of a selected model 
@dataclass
class ModelTrainerConfig:
    # Defining attribute that joins "artifacts" and "model.pk"
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# The Class performs the model prediction based the Model Training
class CustomModel:
    # The class takes in 2 arguments which is the preprocessor object and the model object trained from Model Trainer class below
    def __init__(self, preprocessing_object, trained_model_object):
        # We create an attribute of the preprocessor object passed in through the init constructor
        self.preprocessing_object = preprocessing_object
        # We create an attribute of the trained model object passed in through the init constructor
        self.trained_model_object = trained_model_object

    # This function returns prediction, it takes in input data
    def predict(self, X):
        # Using the preprocessed object entered through the init constructor we transform the input features
        transformed_feature = self.preprocessing_object.transform(X)
        # Returns the prediction of the trained model object entered through the init constructor on the transformed feature
        return self.trained_model_object.predict(transformed_feature)

    # The __repr__ method returns a string representation of the CustomModel instance. It displays the name of the class of the trained model object (e.g., 'RandomForestClassifier').
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    # The __str__ method also returns a string representation of the CustomModel instance. It also displays the name of the class of the trained model object (e.g., 'RandomForestClassifier').
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    # In the given CustomModel class, both __str__ and __repr__ methods are defined to provide a string representation of the trained model object (e.g., 'RandomForestClassifier()'). This way, if you use the print() function or str() function on a CustomModel instance, it will show the user-friendly representation. Similarly, when you use repr() on a CustomModel instance, it will show the developer-friendly representation. This can be useful for debugging and introspection purposes.

# This class trains the model
class ModelTrainer:
    # The class constructor takes in no arguments
    def __init__(self):
        # Defining an attribute of model_trainer_config as an instance of ModelTrainerConfig data class that hold the path where the model would be saved
        self.model_trainer_config = ModelTrainerConfig()

    # Method to do the training, it takes in train array, test array and the preprocessor path
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            # Logging our progress
            logging.info(f"Splitting training and testing input and target feature")

            # Here we split the train array recieved from this method to x_train, y_train, x_test, y_test by indexing the target and input of test and train array
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Defining a dictionary of algorithms and their object for model traing
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            # Logging our progress
            logging.info(f"Extracting model config file path")

            # Here we are defining a dictionary that is assigned to the model evaluation of report of our models
            model_report: dict = evaluate_models(X=x_train, y=y_train, models=models)

            # Printing out the models report
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dict we sort the values of the model_report which is the accuracy score, from the values we take the max value
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            # If we check the best model and it is not less than 0.6 then no good model is found
            if best_model_score < 0.6:
                raise Exception("No best model found")

            # Logging the progress
            logging.info(f"Best found model on both training and testing dataset")

            # loading the preprocessor object
            preprocessing_obj = load_object(file_path=preprocessor_path)

            # Using custom model class we defined above we are passing the preprocessor and the best model to it for prediction
            custom_model = CustomModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )

            # Logging the progress
            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            # Saving the model using the path specified in the model_trainer_config()
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model,
            )

            # The best model which was assigned as best_model variable is now used to predict the accuracy of X_test which is the test data
            predicted = best_model.predict(x_test)

            # Evaluating the accuracy score of the model
            accuracy = accuracy_score(y_test, predicted)

            # upload_file(
            #     from_filename=self.model_trainer_config.trained_model_file_path,
            #     to_filename="model.pkl",
            #     bucket_name=AWS_S3_BUCKET_NAME,
            # )

            # Returning the Accuracy of model
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
