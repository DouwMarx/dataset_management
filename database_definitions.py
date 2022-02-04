from pymongo import MongoClient

# Mongo database
client = MongoClient()

db = client.phenomenological_rapid

raw = db["raw"]
processed = db["processed"]
augmented = db["augmented"]
encoding = db["encoding"]
