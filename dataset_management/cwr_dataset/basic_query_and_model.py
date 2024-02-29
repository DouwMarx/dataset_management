import pathlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load pickle file from processed data folder
processed_dir = pathlib.Path(__file__).parent.joinpath("processed_data")
df = pd.read_pickle(processed_dir.joinpath("cwru_env_spec.pkl"))

# Explore the different sub-datasets
print("Sampling rates: ", df["Sampling Rate [kHz]"].unique())
print("Fault locations: ", df["Fault Location"].unique())
print("Fault widths: ", df["Fault Width [mm]"].unique())
print("Shaft speeds: ", df["Shaft speed [rpm]"].unique())
print("Fault modes: ", df["Fault Mode"].unique())  # None is healthy
print("Measurement locations: ", df["Measurement Location"].unique())

# Limit data to a specific problem
sampling_rate = "12"  # Cheaper to process (Interestingly also easier to classify)
fault_location = "DE"  # Use drive end fault location
measurement_location = "DE"  # Measure at the same place where the fault is
fault_width = "0.53"  # Use large fault
shaft_speed = "1797"  # Highest speed (Many faults per segment length)
fault_mode = "Outer Race Fault: Centre"  # Use Outer race Fault in center or load zone (Easy)

faulty_query = (df["Sampling Rate [kHz]"] == sampling_rate) & \
                       (df["Fault Location"] == fault_location) & \
                       (df["Measurement Location"] == measurement_location) & \
                       (df["Fault Width [mm]"] == fault_width) & \
                       (df["Shaft speed [rpm]"] == shaft_speed) & \
                       (df["Fault Mode"] == fault_mode)

healthy_query = (df["Sampling Rate [kHz]"] == sampling_rate) & \
                        (df["Fault Location"] == "NONE") & \
                        (df["Measurement Location"] == measurement_location) & \
                        (df["Fault Width [mm]"] == "0") & \
                        (df["Shaft speed [rpm]"] == shaft_speed) & \
                        (df["Fault Mode"] == "Reference")


# Each line in the dataframe can be viewed as a dataset. The original dataset is therefore not a traditional tabular dataset
# After the query, in this example, the dataframe should only have one row (Datasets from different operating conditions/fault severities can of course be combined)
faulty_data = df[faulty_query]
healthy_data = df[healthy_query]
input_features = "Envelope Spectrum"

X_healthy = np.array(list(healthy_data[input_features])).squeeze() # Data has a channel dimension which is squeezed away here
X_faulty = np.array(list(faulty_data[input_features])).squeeze()

# Create labels
y_healthy = np.zeros(X_healthy.shape[0])
y_faulty = np.ones(X_faulty.shape[0])

# Split the data
X = np.concatenate([X_healthy, X_faulty])
y = np.concatenate([y_healthy, y_faulty])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Basic model pipeline: Box-Cox, PCA, SVC
pipe = Pipeline(steps=[('boxcox', PowerTransformer()),
                       ('pca', PCA(n_components=0.95, whiten=True)),
                       ('svm', SVC())])

# Fit the model
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("")
print("Classification report: ", sklearn.metrics.classification_report(y_test, y_pred))


