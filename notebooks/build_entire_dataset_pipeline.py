from dataset_management.general.update_database_with_raw import main as main_raw
from dataset_management.general.update_database_with_processed import main as main_proc
from dataset_management.general.update_database_with_augmented import main as main_aug
from dataset_management.general.update_database_with_encodings import main as main_enc
from dataset_management.general.update_database_with_models import main as main_model
from dataset_management.general.update_database_with_metrics import main as main_metric

# from informed_anomaly_detection.visualisation import plots for proof of concept # TODO do the import to automate

from time import time

# db_to_act_on = "ims_test"
db_to_act_on = "phenomenological_rapid"

t_start = time()
main_raw(db_to_act_on)
print("raw data generated in :", time()-t_start)

t_start = time()
main_proc(db_to_act_on)
print("processed data updated in :", time()-t_start)

t_start = time()
main_aug(db_to_act_on)
print("augmented data updated in :", time()-t_start)

# t_start = time()
# main_model()
# print("Models trained in :", time()-t_start)
#
# t_start = time()
# main_enc()
# print("Encodings computed in :", time()-t_start)
#
# t_start = time()
# main_metric()
# print("metric data updated in :", time()-t_start)



