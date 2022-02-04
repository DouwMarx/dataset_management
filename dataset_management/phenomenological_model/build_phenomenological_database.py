from dataset_management import build_data_and_encodings as bde
import pickle
from pypm.phenomenological_bearing_model.make_data import PyBearingDataset
from database_definitions import client


def build_phenomenological_database(n_severities = 10, rapid=True):
    # Mongo database
    if rapid:
        client.phenomenological_rapid.failure_dataset.drop() # Remove the collection
        db = client.phenomenological_rapid # Use a specific dataset for rapid iteration


    else:
        client.phenomenological.failure_dataset.drop() # Remove the collection
        db = client.phenomenological

    failure_dataset = db.failure_dataset    # Select a given collection to use

    # failure_dataset.delete_one({}) # Clear the collection
    # failure_dataset.delete_many({}) # Clear the collection


    o = PyBearingDataset(n_severities=n_severities, failure_modes=["ball", "inner", "outer"],quick_iter=rapid)  # TODO: Drive these parameters with governing yaml file
    result_dict = o.make_measurements_for_different_failure_mode()

    # Pack the bearing dataset into the database
    for mode_name,mode_data in result_dict.items():
        for severity_name,severity_data in mode_data.items():

            meta_data = severity_data["meta_data"]
            time_series = severity_data["time_domain"]

            doc = {"mode": mode_name,
                   "severity": severity_name,
                   "meta_data": pickle.dumps(meta_data),
                   "time_series": pickle.dumps(time_series)
                   }

            failure_dataset.insert_one(doc)  # Insert document into the collection

    return failure_dataset # The mongodb collection


def main():
    sevs = 3
    fd = build_phenomenological_database(n_severities=sevs, rapid=True)
    r = fd.find({"mode":"inner"})
    l = len(list(r)) # Check that the length of the database will be the same as the number of severities
    return l,sevs

if __name__ == "__main__":
    main()

# print(l)

#
# for doc in fd.find({"mode":"inner"}):
#     print(doc["severity"])
#     # return l

# print(main())


# # print(loaded_doc["meta_data"])
# for i in failure_dataset.find():
#     print(i["severity"])