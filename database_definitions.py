from pymongo import MongoClient

# Mongo database
client = MongoClient()

db = client.phenomenological_rapid

raw = db["raw"]
processed = db["processed"]
augmented = db["augmented"]
encoding = db["encoding"]
metrics = db["metrics"]


def make_new_client():
    client = MongoClient()
    db = client.phenomenological_rapid

    raw = db["raw"]
    processed = db["processed"]
    augmented = db["augmented"]
    encoding = db["encoding"]
    metrics = db["metrics"]

    return client, db, raw,processed,augmented,encoding,metrics
