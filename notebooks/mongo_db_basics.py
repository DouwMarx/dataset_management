from pymongo import MongoClient
import numpy as np
import pickle


def main():
    # Start the mongodb  client
    client = MongoClient()

    # Select the database to use
    db = client.saving_serialized

    # Select a given collection to use
    failure_dataset = db.failure_dataset

    # Clear the collection
    failure_dataset.delete_one({})

    # Make a document
    data = np.arange(1000)
    data = pickle.dumps(data)  # Serialize the python object

    doc = {"mode": "inner",
           "raw": data,
           "meta_data": {"thing1":"this",
                         "thing2":"that"}
           }

    failure_dataset.insert_one(doc)  # Insert document into the collection

    loaded_doc = failure_dataset.find_one({"mode": "inner"})  # ,

    recovered_array = pickle.loads(loaded_doc["raw"])

    # print(loaded_doc["meta_data"])

    return recovered_array


if __name__ == "__main__":
    main()