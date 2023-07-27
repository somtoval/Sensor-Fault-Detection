import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

# This class handles the whole training pipeline
class TrainPipeline:
    def __init__(self) -> None:

        # Initializing components as attributes of the class

        self.data_ingestion = DataIngestion()

        self.data_transformation = DataTransformation()

        self.model_trainer = ModelTrainer()

    # This method runs the pipeline
    def run_pipeline(self):
        try:
            # Recieving the output of the data ingestion component in variables train_path and test_path
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # Recieving the output of the data transformation component after by passing in the train path and test path recieved from the data ingestion
            (
                train_arr,
                test_arr,
                preprocessor_file_path,
            ) = self.data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            # Getting the accuracy by passing in the train array, test_array and preprocessor path produced by the data transformation component
            accuracy = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr,
                preprocessor_path=preprocessor_file_path,
            )
            # Printin the accuracy
            print("training completed. Trained model score : ", accuracy)

        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj=TrainPipeline()
    obj.run_pipeline()