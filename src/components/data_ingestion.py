import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import export_collection_as_dataframe

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# Here instead of using a normal class we used a data class which is simply used to store information
@dataclass
class DataIngestionConfig:
    # We defined an attribute to store the path where our train data will be which is "artifacts/train.csv" by joining "artifacts" and "train.csv" to form the path
    train_data_path: str = os.path.join("artifacts", "train.csv")
    # We defined an attribute to store the path where our raw data will be which is "artifacts/data.csv" by joining "artifacts" and "data.csv" to form the path
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    # We defined an attribute to store the path where our test data will be which is "artifacts/test.csv" by joining "artifacts" and "test.csv" to form the path
    test_data_path: str = os.path.join("artifacts", "test.csv")

# A Class to do our data ingestion process
class DataIngestion:
    # Defining our init method to collect no input
    def __init__(self):
        # We defined an attribute which is an instance of the DatIngestionConfig() data class that stores our file path
        self.ingestion_config = DataIngestionConfig()

    # This method does the data ingestion
    def initiate_data_ingestion(self):
        # Logging our progress
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            # # Here we export our collection as dataframe using our predefined function in utils.py and assign it to variable called df
            # df: pd.DataFrame = export_collection_as_dataframe(
            #     db_name=MONGO_DATABASE_NAME, collection_name=MONGO_COLLECTION_NAME
            # )

            df = pd.read_csv("C:/Users/user/My Data Science/PW Skills/course/Projects/sensor-fault-detection2/notebook/data/wafer.csv")
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            # Dropping the first colum that contains name of sensors
            df = df.drop("Unnamed: 0", axis=1)
            # Replacing the na values with np.nan
            df.replace({"na": np.nan}, inplace=True)
            df = df.dropna(subset=["Good/Bad"])

            df = df.fillna(df.mode().iloc[0])

            X = df.drop("Good/Bad", axis=1)
            y = df['Good/Bad']

            smt = SMOTETomek(sampling_strategy="auto")
            
            # Using the SMOTETomek instance to resample the train data to be able to balance the imbalance target variable categories
            X_new, y_new = smt.fit_resample(X, y)

            # Convert the upsampled data back to a DataFrame
            df = pd.DataFrame(X_new, columns=X.columns)
            df['Good/Bad'] = y_new

            print('TTTTTTTTTTTThhhhhhhhhhhhhhhhhhhhhhedfd:dfd;;;', df)

            # # Logging our progress
            # logging.info("Exported collection as dataframe")

            # Making the directory for where our data will be stored as defined in our data class, it gets the directory name of one of our path and creates it "artifacts"
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Storing our dataframe in our specified raw data path 
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            print('Target Value Counts:', df['Good/Bad'].value_counts())

            # Splitting our data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['Good/Bad'], random_state=42)

            # Storing splitted train data to our train path
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            # Storing splitted test data to our test path
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            # Logging our progress
            logging.info(
                f"Ingested data from mongodb to {self.ingestion_config.raw_data_path}"
            )

            # Logging our progress
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            # Returning the train data path and the test data path
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

# import pandas as pd

# df = pd.read_csv("C:/Users/user/My Data Science/PW Skills/course/Projects/sensor-fault-detection2/notebook/data/wafer.csv")
# print(df)
# df = df.drop("Unnamed: 0", axis=1)
# print(df)