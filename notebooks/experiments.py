from dataset_management.general.update_database_with_raw import ims_raw,pm_raw
from dataset_management.general.update_database_with_processed import process_bandpass,process_no_bandpass
from dataset_management.general.update_database_with_augmented import main as aug_main
from dataset_management.general.update_database_with_models import main as mod_main
from dataset_management.general.update_database_with_encodings import main as enc_main
from dataset_management.general.update_database_with_models import train_ims_models
from dataset_management.general.update_database_with_models import train_phenomenological_perfect_augmentation
from dataset_management.general.update_database_with_metrics import main as metric_main

experiments ={
 # IMS test 1 bearing 3: Inner fault
 "ims_t1_b3_inner":
  {"db_to_act_on": "ims_test1_bearing3_channel1",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 # IMS test 1 bearing 4:  Roller fault
 "ims_t1_b4_ball":
  {"db_to_act_on": "ims_test1_bearing4_channel1",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func": metric_main
   },

 # IMS test 2 channel 1: Outer faults
 "ims_rapid0_t2_c1_outer":
  {"db_to_act_on": "ims_rapid1_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 "ims_rapid1_t2_c1_outer":
  {"db_to_act_on": "ims_rapid1_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 "ims_t2_c1_outer":
  {"db_to_act_on": "ims_test2_bearing1_channel1",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 # IMS test 3 bearing 3:  Outer fault
 "ims_t3_b3_outer":
  {"db_to_act_on": "ims_test3_bearing3_channel3",
   "raw_func": ims_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func": metric_main
   },

 # Phenomenological experiments
 "phenomenological_rapid0":
  {"db_to_act_on": "phenomenological_rapid0",
   "raw_func": pm_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 "phenomenological":
  {"db_to_act_on": "phenomenological",
   "raw_func": pm_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_ims_models,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },

 "phenomenological_perfect_augmentation":
  {"db_to_act_on": "phenomenological",
   "raw_func": pm_raw,
   "process_func": process_no_bandpass,
   "augment_func": aug_main,
   "model_func": train_phenomenological_perfect_augmentation,
   "encoding_func": enc_main,
   "metric_func":metric_main
   },
}