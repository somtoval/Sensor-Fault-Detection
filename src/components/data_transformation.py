import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

# Using this data class to specify where our preprocessor object would be stored
@dataclass
class DataTransformationConfig:
    # Defining an attribute to hold a path which is "artifacts/preprocessor.pkl"
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

def replace_na_with_nan(X):
    return np.where(X == 'na', np.nan, X)

# The Class to perform the data transformation
class DataTransformation:
    # The class takes no inputs from the init constructor
    def __init__(self):
        # Defining an attribute which is an instance of our data class which holds the path of where our preprocessor will be stored
        self.data_transformation_config = DataTransformationConfig()

    # This method returns the preprocessor object for transformation
    def get_data_transformer_object(self):
        try:
            
            # define custom function to replace 'NA' with np.nan
            # replace_na_with_nan = lambda X: np.where(X == 'na', np.nan, X)

            

            # define the steps for the preprocessor pipeline
            nan_replacement_step = ('nan_replacement', FunctionTransformer(replace_na_with_nan))
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                nan_replacement_step,
                imputer_step,
                scaler_step
                ]
            )
            
            # Returing the preprocessor
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # This method applies the data transformation to our data, it takes in two arguments which is the train and test path
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading the train dataframe
            train_df = pd.read_csv(train_path)
            print(train_df['Good/Bad'])
            print(train_df.info())
            print("----------------------------")
            print('Length of data:', train_df.shape)
            print('Uniques:', train_df['Good/Bad'].unique())
            print('Length of minus 1:', len(train_df[train_df['Good/Bad'] == -1]))
            print('Length of plus 1:', len(train_df[train_df['Good/Bad'] == +1]))

            # Reading the test dataframe
            test_df = pd.read_csv(test_path)
            print(test_df['Good/Bad'])
            print(test_df.info())
            print("----------------------------")
            print('Length of data:', test_df.shape)
            print('Uniques:', test_df['Good/Bad'].unique())
            print('Length of the minus 1:', len(test_df[test_df['Good/Bad'] == -1]))
            print('Length of plus 1:', len(test_df[test_df['Good/Bad'] == +1]))
 
            # Assigning the preprocessor object to a variable
            preprocessor = self.get_data_transformer_object()

            # Specifying the target variable
            target_column_name = "Good/Bad"
            # Encoding the target variable by mapping the categories
            target_column_mapping = {+1: 0, -1: 1}

            # Dropping the target column for the train data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            
            # Mapping the target variable of the train data using our target_column_mapping dictionary
            target_feature_train_df = train_df[target_column_name].map(target_column_mapping)
            print('Train y:', target_feature_train_df)
            print('zerossss:', target_feature_train_df[target_feature_train_df == 0])
            print('onesssss:', target_feature_train_df[target_feature_train_df == 1])

            # Dropping the target column for the test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            # Mapping the target variable of the test data using our target_column_mapping dictionary
            target_feature_test_df = test_df[target_column_name].map(target_column_mapping)
            print('Test y:', target_feature_test_df)
            print('zerossss test:', target_feature_test_df[target_feature_test_df == 0])
            print('onesssss test:', target_feature_test_df[target_feature_test_df == 1])

            # Fitting the preprocessor on the train input feature
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)

            # Fitting the preprocessor on the test input feature
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            # Initializing the SMOTETomek instance for imbalance data handling
            # smt = SMOTETomek(sampling_strategy='minority', random_state=42, k_neighbors=3)
            # smt = SMOTE(random_state=42, k_neighbors=1)

            ###########################
            # The error I am getting about k_neighbors in smote is because the data is a small highly imbalanced dataset (category counts = 94,6) and it works in the notebook because the smote was applied before splitting but here we have applied it after splitting so in the train we would get for the secound category about 5 and for the test we get only 1 data from the second category and smote need more than 1 data to generate instances 
            ############################

            # print("transformed_input_train_feature: ", transformed_input_train_feature)
            # print("target_feature_train_df: ", target_feature_train_df)
            
            # smt = SMOTETomek(sampling_strategy="auto")
            
            # # Using the SMOTETomek instance to resample the train data to be able to balance the imbalance target variable categories
            # input_feature_train_final, target_feature_train_final = smt.fit_resample(
            #     transformed_input_train_feature, target_feature_train_df
            # )

            # # Using the SMOTETomek instance to resample the test data to be able to balance the imbalance target variable categories
            # input_feature_test_final, target_feature_test_final = smt.fit_resample(
            #     transformed_input_test_feature, target_feature_test_df
            # )

            # Joining the now balanced train input data to the now balanced train target data because both are array, we make use of np.concatenate which it's short form is np.c_
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            # Joining the now balanced test input data to the now balanced test target data because both are array, we make use of np.concatenate which it's short form is np.c_
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]


            # Saving the preprocessor
            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj= preprocessor)

            # Returning the train array and test array and also the preprocessor object path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
