from database_definitions import make_db
from dataset_management.phenomenological_model.add_raw_data_to_database import main as pm_main
from dataset_management.ims_dataset.add_raw_data_to_database import main as ims_main
from dataset_management.ultils.mongo_test_train_split import test_train_split

def ims_raw(db_to_act_on):
    d = ims_main(db_to_act_on)
    return d

def pm_raw(db_to_act_on):
    d = pm_main(db_to_act_on)
    test_train_split(db_to_act_on)
    return d



def main(db_to_act_on):
    # if db_to_act_on in ["ims","ims_test"]:
    #     return ims_main(db_to_act_on)
    #
    # else:
    #     return pm_main(db_to_act_on)
    # Main is currently evaluating the test train split functionality
    test_train_split(db_to_act_on)

if __name__ == "__main__":
    main("ims_test2_bearing1_channel1")