from dataset_management.general.update_database_with_raw import ims_raw,pm_raw
from dataset_management.general.update_database_with_processed import process_ims,process_pm
from dataset_management.general.update_database_with_augmented import main as aug_main
from dataset_management.general.update_database_with_models import main as mod_main
from dataset_management.general.update_database_with_encodings import main as enc_main
from dataset_management.general.update_database_with_models import train_ims_models

experiments ={
 "ims_rapid1_t2_c1_outer":
  {"db_to_act_on": "ims_rapid1_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_ims,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main
   },

 "ims_rapid2_t2_c1_outer":
  {"db_to_act_on": "ims_rapid2_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_ims,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main
   },

 "ims_t2_c1_outer":
  {"db_to_act_on": "ims_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_ims,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main
   },

 "phenomenological_rapid":
  {"db_to_act_on": "phenomenological_rapid",
   "raw_func": pm_raw,
   "process_func": process_pm,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main
   },

 "phenomenological":
  {"db_to_act_on": "phenomenological",
   "raw_func": pm_raw,
   "process_func": process_pm,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main
   },

 # "phenomenological_rapid":
#     {"db_to_act_on": "phenomenological_rapid",
#      "raw_func":  general.update_database_with_raw.pm_raw,
#      "process_func":  general.update_database_with_processed.process_pm,
#      "augment_func":  general.update_database_with_augmented.main,
#      "model_func":  general.update_database_with_models.main,
#      "encoding_func":  general.update_database_with_encodings.main
#
#      }
}