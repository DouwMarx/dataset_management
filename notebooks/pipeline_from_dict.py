from experiments import experiments as experiment_dict
from time import time

# experiment = "ims_rapid2_t2_c1_outer"
experiment = "ims_t2_c1_outer"
# experiment = "phenomenological_rapid"
# experiment = "phenomenological"

experiment_specs = experiment_dict[experiment]
db_to_act_on = experiment_specs["db_to_act_on"]

# t_start = time()
# experiment_specs["raw_func"](db_to_act_on)
# print("raw data generated in :", time()-t_start)

# t_start = time()
# experiment_specs["process_func"](db_to_act_on)
# print("processed data updated in :", time()-t_start)
#
# t_start = time()
# experiment_specs["augment_func"](db_to_act_on)
# print("augmented data updated in :", time()-t_start)

t_start = time()
experiment_specs["model_func"](db_to_act_on)
print("Models trained in :", time()-t_start)
#
t_start = time()
experiment_specs["encoding_func"](db_to_act_on)
print("Encodings computed in :", time()-t_start)


# from informed_anomaly_detection.visualisation import generate_plots_for_proof_of_concept

# t_start = time()
# main_metric()
# print("metric data updated in :", time()-t_start)



