from pymongo import MongoClient
import json
from time import sleep

# Load data
with open('filtered_snps.json', 'r') as f:
    snps = json.load(f)

## Injecting data

# connect to the database
HOST = 'cluster0.fccpb1y.mongodb.net'
USER = 'YuriCha042'
PASSWORD = 'dPYVYQatakEUDl8S'
DATABASE_NAME = 'Project'
COLLECTION_NAME = 'GWAS'
MONGO_URI = f"mongodb+srv://{USER}:{PASSWORD}@{HOST}/{DATABASE_NAME}?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
database = client[DATABASE_NAME]
gwas = database[COLLECTION_NAME]

#gwas.delete_many({}) #This will clean gwas database

for start in range(0, 280000, 5000):
    gwas.insert_many(snps['_embedded']['singleNucleotidePolymorphisms'][start: start + 5000])
    sleep(30)