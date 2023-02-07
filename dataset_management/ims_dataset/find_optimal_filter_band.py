"""
This project is initially interested in using the envelope spectrum

For this we need the optimal frequency band for impulsive information.

We use the Kurtogram to get this frequency band.

This script searches for the optimal frequency band for the IMS dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from database_definitions import make_db
from signal_processing.grams import kurtogram, plotKurtogram

db,client = make_db("ims_test")

print(db["raw"].count_documents({}))

# for mode in ["inner","outer","ball"]:
#     kurt = 0
#     list_nw = 2 ** np.arange(4, 9, 1)
#     for example_signal in db["raw"].find({"mode":mode}):
#         meta_data = example_signal["meta_data"]
#         sampling_frequnency = meta_data["sampling_frequency"]
#         sig = example_signal["time_series"]
#         sig = sig - np.average(sig)
#         kurtdat = kurtogram(sig,sampling_frequnency,list_nw)
#         kurt += kurtdat["Kurt"]
#         print("samp")
#
#
#     kurtdat.update({"Kurt":kurt})
#     plotKurtogram(kurtdat)
#     plt.title(mode)

kurts = []
list_nw = 2 ** np.arange(4, 9, 1)
for test in db["raw"].distinct("ims_test_number"):
    channels = db["raw"].find({"ims_test_number":test}).distinct("ims_channel_number")
    for channel in channels:
        data_to_use = db["raw"].find({"ims_test_number": test,
                                      "ims_channel_number":channel,
                                      "$or": [{"mode": "outer"},
                                              {"mode": "inner"},
                                              {"mode": "ball"},
                                              ]})
        kurtdat = None
        kurt = 0
        for example_signal in tqdm(data_to_use):
            meta_data = example_signal["meta_data"]
            sampling_frequnency = meta_data["sampling_frequency"]
            sig = example_signal["time_series"]
            sig = (sig - np.average(sig))/np.std(sig)
            kurtdat = kurtogram(sig,sampling_frequnency,list_nw)
            kurt += kurtdat["Kurt"]


        if kurtdat is not None:
            kurtdat.update({"Kurt":kurt})
            name = "test" + test + "channel " + channel + "mode" + str(example_signal["mode"])
            kurtdat.update({"name":name})
            kurts.append(kurtdat)
            plotKurtogram(kurtdat,logscale=True)
            plt.title(name)
            plt.savefig("kurtograms/"+name+".pdf")
