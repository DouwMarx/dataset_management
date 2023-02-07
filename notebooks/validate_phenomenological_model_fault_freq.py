import matplotlib.pyplot as plt
import numpy as np
from signal_processing.spectra import env_spec

from database_definitions import make_db

# Purpose of this scipt is to verify that the fault frequency shown by the envelope spectrum is the same as the fault frequency provided to the phenomenological model.

db,client = make_db("phenomenological_rapid")

doc = db["raw"].find_one({"mode":"outer",
                          "severity":1})

sig = doc["time_series"]
fs = doc["meta_data"]["sampling_frequency"]


plt.figure()
plt.plot(np.linspace(0,1/fs*len(sig),len(sig)),sig)

freq,mag,phase= env_spec(sig,fs)

plt.figure()
plt.plot(freq,mag)

print("fault type: ", doc["mode"])
print("mean rotation frequency :", doc["meta_data"]["mean_rotation_frequency"])
print("Expected fault frequencies :", doc["meta_data"]["expected_fault_frequencies"])


# TODO: This should be moved to the phenomenological project. Envelope spectrum must just be be implemented again so that it does not rely on signal processing library