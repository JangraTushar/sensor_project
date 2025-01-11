from pymongo.mongo_client import MongoClient # type: ignore
import pandas as pd
import json

#url
uri="mongodb+srv://Tushar:Jantus654@cluster0.09tep.mongodb.net/?retryWrites=true&w=majority"

#create new client and connect to server
client = MongoClient(uri)

#create database name and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

df = pd.read_csv("E:\sensor fault project\notebooks\wafer_fault.csv")

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)