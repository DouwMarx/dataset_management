import definitions
import pickle
import numpy as np
from pathlib import Path
import torch


def run_or_load(function,*args, recompute = False,result_name="intermediate"):
    path = definitions.data_dir.joinpath("expensive_compute_cache", result_name + ".pkl")
    if recompute == False:
        try:
            with open(path,'rb') as f:
                result = pickle.load(f)
                return result
        except FileNotFoundError:
            raise FileNotFoundError("File does not exist, must likely set the recompute argument to True")
    else:
        result = function(*args)
        with open(path, 'wb') as f:
            pickle.dump(result, f)
        return result

# Dataset
def save_dataset(object,name):
    path = definitions.data_dir.joinpath("processed",name + ".pkl")
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def load_dataset(name):
    path = definitions.data_dir.joinpath("processed",name + ".pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Pytorch model
def save_pytorch_model(model,name,mongo=True):
    if mongo:
        path = definitions.models_dir.joinpath("trained_models", "mongo", name + ".pt")
    else:
        path = definitions.models_dir.joinpath("trained_models", name + ".pt")

    torch.save(model, path)
    print("model saved to: " + str(path))
    return path


def load_pytorch_model(name):
    path = definitions.models_dir.joinpath("trained_models",name + ".pt")
    return torch.load(path)

# Anomaly results
def save_anomaly_metrics(object,name):
    path = definitions.models_dir.joinpath("anomaly_metrics_results",name + ".pkl")
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def load_anomaly_metrics(name):
    path = definitions.models_dir.joinpath("anomaly_metrics_results",name + ".pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

