from pymongo import MongoClient

def make_db(db_to_act_on):
    # Idea is that this function will later be used to call different types of datasets or rapid iteration vs normal iteration datasets.
    client = MongoClient()
    db = client[db_to_act_on]
    return db, client

