from pathlib import Path

from experiments import experiments as experiment_dict
from time import time
from database_definitions import make_db

# experiment = "ims_rapid2_t2_c1_outer"
experiment = "ims_t2_c1_outer"
# experiment = "phenomenological_rapid0"
# experiment = "phenomenological"
# experiment = "phenomenological_perfect_augmentation"

experiment_specs = experiment_dict[experiment]
db_to_act_on = experiment_specs["db_to_act_on"]

# t_start = time()
# experiment_specs["raw_func"](db_to_act_on)
# print("raw data generated in :", time()-t_start)
#
# t_start = time()
# experiment_specs["process_func"](db_to_act_on)
# print("processed data updated in :", time()-t_start)
#
# t_start = time()
# experiment_specs["augment_func"](db_to_act_on)
# print("augmented data updated in :", time()-t_start)
#
t_start = time()
experiment_specs["model_func"](db_to_act_on)
print("Models trained in :", time()-t_start)

t_start = time()
experiment_specs["encoding_func"](db_to_act_on)
print("Encodings computed in :", time()-t_start)

t_start = time()
experiment_specs["metric_func"](db_to_act_on)
print("metric data updated in :", time()-t_start)


# meeting_path = Path("/home/douwm/projects/PhD/reports/meetings/20220321_informed_anomaly_detection_on_ims/20220321_beamer/images")
plots_path = Path("/home/douwm/projects/PhD/code/informed_anomaly_detection/reports/plots")

#
#
from informed_anomaly_detection.visualisation.visualisation_pipeline import make_plots
make_plots(db_to_act_on,plots_path,export_pdf=True,experiment_name=experiment)




