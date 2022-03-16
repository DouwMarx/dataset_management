from experiments import experiments as experiment_dict
from time import time

# experiment = "ims_test"
# experiment = "ims"
experiment = "ims_outer_t2_c1"


experiment_specs = experiment_dict[experiment]
db_to_act_on = experiment_specs["db_to_act_on"]

# t_start = time()
# experiment_specs["raw_func"](db_to_act_on)
# print("raw data generated in :", time()-t_start)
#
# t_start = time()
# experiment_specs["process_func"](db_to_act_on)
# print("processed data updated in :", time()-t_start)

# t_start = time()
# experiment_specs["augment_func"](db_to_act_on)
# print("augmented data updated in :", time()-t_start)

t_start = time()
experiment_specs["model_func"](db_to_act_on)
print("Models trained in :", time()-t_start)

t_start = time()
experiment_specs["encoding_func"](db_to_act_on)
print("Encodings computed in :", time()-t_start)
#
# t_start = time()
# main_metric()
# print("metric data updated in :", time()-t_start)



