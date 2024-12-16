Description by Deepti Kunte



The dataset contains the following:

    Data of two vehicles: Mondeo and Vectra
    4000 raw sound signals of 5s duration each. These are simulated at constant speed (e.g. Mondeo_sounds.npy). It would be best to use only a fraction of this as given sufficient data, probably all methods will work equally well on a simple dataset.
    2 types of faults: whistle and leakage. These can occur simultaneously. It is a balanced dataset.
    The faults themselves have different amplitudes and positions in the frequency domain. The simulation parameters can be found in the corresponding excel sheets (e.g. Mondeo_simdata.xlsx) 

 

I have calculated the following input feature candidates:

    Mel-spectrum (Input feature to the semi-supervised methods)
    Skewness and Kurtosis of the mel-spectrum (Leakage indicators)

 

You can use the attached code to load the files/re-calculate the features (extract_features.py)
