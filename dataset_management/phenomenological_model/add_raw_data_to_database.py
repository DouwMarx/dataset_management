import pickle
from pypm.phenomenological_bearing_model.make_data import PyBearingDataset
from tqdm import tqdm

from database_definitions import make_db

def build_phenomenological_database(db_to_act_on, n_severities = 3, rapid=True):
    # Mongo database
    db,client = make_db(db_to_act_on)
    db["raw"].delete_many({}) #  Remove the items in the collection

    o = PyBearingDataset(n_severities=n_severities, failure_modes=["ball", "inner", "outer"],quick_iter=rapid)
    result_dict = o.make_measurements_for_different_failure_mode()

    # Pack the bearing dataset into the database
    for mode_name,mode_data in result_dict.items():
        print(mode_name)
        docs_for_mode = []
        for severity_name,severity_data in tqdm(mode_data.items()):
            meta_data = severity_data["meta_data"]
            time_series = severity_data["time_domain"]

            for signal in time_series: # Each row is a signal
                doc = {"mode": mode_name,
                       "severity": severity_name,
                       "meta_data": meta_data,
                       "time_series": list(signal),
                       "augmented":False
                       }

                docs_for_mode.append(doc)

        db["raw"].insert_many(docs_for_mode)  # Insert document into the collection

    print(db["raw"].count_documents({}))

    return db["raw"]# The mongodb collection


def main():
    db_to_act_on = "phenomenological_rapid"
    sevs = 10
    fd = build_phenomenological_database(db_to_act_on,n_severities=sevs, rapid=False)
    r = fd.find({"mode":"inner"})
    l = len(list(r)) # Check that the length of the database will be the same as the number of severities
    return l, sevs

if __name__ == "__main__":
    main()