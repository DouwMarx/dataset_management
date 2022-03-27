import sys
from pathlib import Path

from experiments import experiments as experiment_dict
from time import time, sleep
from informed_anomaly_detection.visualisation.visualisation_pipeline import make_plots

# meeting_path = Path("/home/douwm/projects/PhD/reports/meetings/20220321_informed_anomaly_detection_on_ims/20220321_beamer/images")
# plots_path = Path("/home/douwm/projects/PhD/code/informed_anomaly_detection/reports/plots")
phme_report_path = Path("/home/douwm/projects/PhD/reports/conferences/PHME2022/paper_tex/src/images/plots")

# experiment = "ims_rapid1_t2_c1_outer"
# experiment = "ims_rapid0_t2_c1_outer"
# experiment = "phenomenological_rapid0"
# experiment = "phenomenological_perfect_augmentation"

experiment = "phenomenological"

# experiment = "ims_t2_c1_outer"
# experiment = "ims_t1_b3_inner"
# experiment = "ims_t1_b4_ball"
# experiment = "ims_t3_b3_outer"

experiment_specs = experiment_dict[experiment]
db_to_act_on = experiment_specs["db_to_act_on"]


def run_raw():
    t_start = time()
    experiment_specs["raw_func"](db_to_act_on)
    print("raw data generated in :", time() - t_start)


def run_process():
    t_start = time()
    experiment_specs["process_func"](db_to_act_on)
    print("processed data updated in :", time() - t_start)


def run_augment():
    t_start = time()
    experiment_specs["augment_func"](db_to_act_on)
    print("augmented data updated in :", time() - t_start)


def run_models():
    t_start = time()
    experiment_specs["model_func"](db_to_act_on, experiment_specs["training_parameters"])
    print("Models trained in :", time() - t_start)


def run_encodings():
    t_start = time()
    experiment_specs["encoding_func"](db_to_act_on)
    print("Encodings computed in :", time() - t_start)


def run_metrics():
    t_start = time()
    experiment_specs["metric_func"](db_to_act_on)
    print("metric data updated in :", time() - t_start)


def run_plots():
    make_plots(db_to_act_on, "latent_directions_maximally_different_" + experiment, phme_report_path, export_pdf=True,
               experiment_name=experiment)


def main(to_run):
    run_dict = {"raw": run_raw,
                "process": run_process,
                "augment": run_augment,
                "models": run_models,
                "encodings": run_encodings,
                "metrics": run_metrics,
                "plots": run_plots,
                }

    for func_to_run in to_run:
        run_dict[func_to_run]()


if __name__ == "__main__":
    # if sys.argv[1] == 0:
    # main(["raw"])
    # main(["process","augment"])

    main(["models","encodings","metrics","plots"])

    # main(["models","encodings","metrics"])
    # main(["metrics","plots"])
    # main(["plots"])
