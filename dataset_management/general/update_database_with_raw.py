from dataset_management.phenomenological_model.add_raw_data_to_database import main as pm_main
from dataset_management.ims_dataset.add_raw_data_to_database import main as ims_main

def ims_raw(db_to_act_on):
    return ims_main(db_to_act_on)

def pm_raw(db_to_act_on):
    return pm_main(db_to_act_on)

# def main(db_to_act_on):
#     if db_to_act_on in ["ims","ims_test"]:
#         return ims_main(db_to_act_on)
#
#     else:
#         return pm_main(db_to_act_on)

# if __name__ == "__main__":
#     main(db_to_act_on="phenomenological_rapid")