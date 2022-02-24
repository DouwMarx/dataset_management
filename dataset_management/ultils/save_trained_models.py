import file_definitions as fd
import torch
import pickle

# Pytorch model
def save_trained_model(model,name,model_implementation="torch"):
    if model_implementation == "torch":
        path = fd.models_dir.joinpath(name + ".pt")
        torch.save(model, path)
    else:
        path = fd.models_dir.joinpath(name + ".pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    print("{} model saved to: ".format(model_implementation) + str(path))
    return path
