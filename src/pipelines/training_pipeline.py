import os
import sys

parent_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,parent_dir)

from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    
    data_tranformation=DataTransformation()

    train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)