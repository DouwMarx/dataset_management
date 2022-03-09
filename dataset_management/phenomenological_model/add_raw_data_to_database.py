import pickle

from pypm.phenomenological_bearing_model.bearing_model import Bearing
from pypm.phenomenological_bearing_model.make_data import PyBearingDataset
from tqdm import tqdm

from database_definitions import make_db

def build_phenomenological_database(db_to_act_on, n_severities = 3, rapid=True):
    # Mongo database
    db,client = make_db(db_to_act_on)
    db["raw"].delete_many({}) #  Remove the items in the collection
    # print(db["raw"].count_documents({}))

    o = PyBearingDataset(n_severities=n_severities, failure_modes=["ball", "inner", "outer"],quick_iter=rapid)
    result_dict = o.make_measurements_for_different_failure_mode()

    d = o.simulation_properties["d"]
    D = o.simulation_properties["D"]
    n_ball = o.simulation_properties["n_ball"]
    contact_angle = o.simulation_properties["contact_angle"]

    bearing_geom_obj = Bearing(d, D, contact_angle, n_ball)


    # Pack the bearing dataset into the database
    for mode_name,mode_data in result_dict.items():
        print(mode_name)
        docs_for_mode = []
        for severity_name, severity_data in tqdm(mode_data.items()):
            meta_data = severity_data["meta_data"]
            expected_fault_frequencies_dict = {"expected_fault_frequencies": {
                fault_type: bearing_geom_obj.get_expected_fault_frequency(fault_type, meta_data["mean_rotation_frequency"]) for
                fault_type in ["ball", "outer", "inner"]}}
            meta_data.update(expected_fault_frequencies_dict)

            time_series = severity_data["time_domain"]


            for signal in time_series: # Each row is a signal
                doc = {"mode": mode_name,
                       "severity": int(severity_name),
                       "meta_data": meta_data,
                       "time_series": list(signal),
                       "augmented":False
                       }

                docs_for_mode.append(doc)

        db["raw"].insert_many(docs_for_mode)  # Insert document into the collection

    print(db["raw"].count_documents({}))

    print(doc.keys)
    return db["raw"]# The mongodb collection


def main(db_to_act_on):
    if db_to_act_on == "phenomenological_rapid":
        sevs = 2
        rapid = True
    else:
        sevs = 10
        rapid = False

    fd = build_phenomenological_database(db_to_act_on,n_severities=sevs, rapid=rapid)

    return fd.count_documents({})

if __name__ == "__main__":
    main("phenomenological_rapid")