from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # to combine two pipeline

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
from dataclasses import dataclass
import numpy as np
import pickle
### we perform EDA

# Data transformation config
@dataclass
class DataTransfomationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

# Data ingestion class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransfomationConfig()
        
    def get_data_tranformation_object(self):
        try:
            logging.info("data transformation initiation")
            # segregated numericl and categorical columns
            categorical_cols=['cut', 'color', 'clarity']
            numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
            
        # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            # read data
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("read train and test data completed.")
            logging .info(f"Train Dataframe Head :\n{train_df.head().to_string()}")
            logging .info(f"Test Dataframe Head :\n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_tranformation_object()
            
            target_column_name='price'
            drop_column=[target_column_name,'id']
            
            ## dividing features into independent and dependent features

            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## apply the trransformation
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            logging.info("Applying preprocessing object on train and test dataset.")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessor pickle is creaated and saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except CustomException as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)