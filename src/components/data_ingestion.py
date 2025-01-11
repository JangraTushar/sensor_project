import sys
import os
import pandas as pd
import numpy as np
from pymongo import MongoClient # type: ignore
from zipfile import Path
from src.constants import * # type: ignore
from src.exception import CustomException
from src.logger import logging
from src.utils.main.utils import Mainutils # type: ignore
from dataclass import dataclass # type: ignore


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = Mainutils()
    def export_collection_as_dataframe(self,collection_name,db_name):

        try:
            mongo_client = MongoClient(MONGO_DB_URL) # type: ignore

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if"_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            
            df.replace({"na": np.nan}, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_into_feature_store_file_path(self) -> pd.DataFrame: # type: ignore

        try:
        
            logging.info(f"exporting data from mongodb")
            raw_file_path = self.data_ingestion_config.artifact_folder # type: ignore

            os.makedirs(raw_file_path,exist_ok=True)

            sensor_data = self.export_collection_as_dataframe( # type: ignore
                collection_name=MONNGO_COLLECTION_NAME, # type: ignore
                db_name=MONGO_DATABSE_NAME # type: ignore
            )

            logging.info(f"saving exported data into feature stored file path : {raw_file_path}")

            feature_store_file_path = os.path.join(raw_file_path,"wafer_fault.csv")

            sensor_data.to_csv(feature_store_file_path,index=False)

            return sensor_data

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self) -> Path:

        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")

        try:
            feature_stored_file_path = self.export_data_into_feature_store_file_path() # type: ignore

            logging.info("got the data from mongodb")

            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")

            return feature_stored_file_path
        
        except Exception as e:
            raise CustomException(e,sys) from e



