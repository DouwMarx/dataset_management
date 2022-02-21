from dataset_management import build_data_and_encodings as bde
import pickle
from pypm.phenomenological_bearing_model.make_data import PyBearingDataset
from database_definitions import raw, processed, augmented


def build_phenomenological_database(n_severities = 10, rapid=True):
    # Mongo database
    raw.delete_many({}) #  Remove the items in the collection

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
                   "time_series": pickle.dumps(time_series),
                   "augmented":False
                   }

            raw.insert_one(doc)  # Insert document into the collection

    return raw# The mongodb collection


def main():
    sevs = 3
    fd = build_phenomenological_database(n_severities=sevs, rapid=True)
    r = fd.find({"mode":"inner"})
    l = len(list(r)) # Check that the length of the database will be the same as the number of severities
    return l, sevs

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