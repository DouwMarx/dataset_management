from dataset_management.phenomenological_model.add_raw_data_to_database import main as main_raw
from dataset_management.general.update_database_with_processed import main as main_proc
from dataset_management.general.update_database_with_augmented import main as main_aug
from dataset_management.general.update_database_with_encodings import main as main_enc
from dataset_management.general.update_database_with_models import main as main_model
from dataset_management.general.update_database_with_metrics import main as main_metric

from time import time

t_start = time()
main_raw()
print("raw data generated in :", time()-t_start)

t_start = time()
main_proc()
print("processed data updated in :", time()-t_start)

t_start = time()
main_aug()
print("augmented data updated in :", time()-t_start)

t_start = time()
main_model()
print("Models trained in :", time()-t_start)

t_start = time()
main_enc()
print("Encodings computed in :", time()-t_start)

# t_start = time()
# main_metric()
# print("metric data updated in :", time()-t_start)



