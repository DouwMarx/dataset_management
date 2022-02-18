from pymongo import MongoClient
from multiprocessing import Pool
import time
from database_definitions import make_new_client




def process(a, b):
    new_doc = {
        "this": a,
        "that": b,
    }


    time.sleep(2)

    client, db, raw, processed, augmented, encoding, metrics = make_new_client()
    a = raw.find_one({})
    print(a["mode"])


    raw.insert_one(new_doc)
    client.close()


# if __name__ == '__main__':
pool = Pool()  # Create a multiprocessing Pool
r = pool.starmap(process, (
    (1, 2),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
    (2, 3),
))  # process data_inputs iterable with pool

client = MongoClient()
db = client.parallel_test
col = db.parallel_test
for i in col.find():
    print(i)

col.delete_many({})
