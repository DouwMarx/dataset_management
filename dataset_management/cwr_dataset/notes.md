# Getting started
1) Download the data by running the `download_data.py` script. This will download the data from the Case Western Reserve University Bearing Data Center. Make sure to update the path where the data is saved.
2) Get an organised dataframe (time series, cut into chunks) for different operating conditions, fault modes, fault severity, sampling rate and sensor location with the `write_data_to_standard_structure.py` script. This will write the data as a chunky pickle file (Alternatively, you can just use the `get_cwru_data_frame` function on demand).
3) For envelope spectrum data and some derived features including expected fault frequencies, run `envelope_spectrum_and_expected_fault_spectrum.py`. This will add columns to the existing dataframe with the envelope spectra and an expected fault spectrum.

# More detailed instructions
## Instructions to Download and Run Script

### Step 1: Clone the Repository

First, you need to clone the repository from GitHub. Open a terminal and navigate to the directory where you want to clone the repository. Then, run the following command:

```bash
git clone https://github.com/DouwMarx/dataset_management.git
```

This will create a new directory named `dataset_management` in your current directory.

Alternatively, you can download the repository as a zip file and extract it to a directory of your choice.

### Step 2: Navigate to the Repository Directory

Change your current directory to the `dataset_management` directory:

```bash
cd dataset_management
```

### Step 3: Run the `download_data.py` Script

Now, you are in the correct directory to run the `download_data.py` script. Run the following command:

```bash
python cwr_dataset/download_data.py
```

This will download the necessary data and save it in the appropriate directories.

### Step 4: Run the `get_envelope_and_fault_locations.py` Script

After the `download_data.py` script has successfully completed, you can run the `get_envelope_and_fault_locations.py` script. Run the following command:

```bash
python cwr_dataset/get_envelope_and_fault_locations.py
```

This will execute the script and perform the necessary operations.

#### (If trouble with finding module): Add Parent Directory to Python Path

##### For Windows:
```bash
set PYTHONPATH=%cd%\..;%PYTHONPATH%
```

##### For Linux:
```bash
export PYTHONPATH=$(dirname $PWD):$PYTHONPATH
```

### Note

Ensure that you have the necessary Python environment and all the required packages installed. You can check the `requirements.txt` file in the repository for the list of required packages. You can install them using pip:

```bash
pip install -r requirements.txt
```

Remember to run all commands from the root directory of the cloned repository (`dataset_management`). This will prevent errors related to running from the wrong directory or `dataset_management` being undefined.


# Source and references

See 

https://engineering.case.edu/bearingdatacenter/normal-baseline-data

and 

https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data

DE - drive end accelerometer data
FE - fan end accelerometer data
BA - base accelerometer data
time - time series data
RPM - rpm during testing

See https://doi-org.kuleuven.e-bronnen.be/10.1016/j.ymssp.2015.04.021 for a good explanation of the data (Not the original source, but a good explanation of the data).

# Example dataset for validation
It is typically much harder to detect a fault with the Fan-end (FE) measurement location.

Dataset that have detectable fault for all modes (According to Randall and Smith, 2009) are:
12k data, Drive end (DE)
Record numbers are:
Inner:            210 
Ball:             223
Outer (Centered): 235
Healthy            98
These are all at motor load 1hp and 1772 RPM.
Fault severity is 0.021 inches or 0.53mm

# Length of cut signals
It makes sense to ensure that signal segments are sufficiently long to include multiple fault events.
Signals are segmented based on the lowest expected fault frequency for the average operating speed.
Ideally, different segment length would be used for different operating conditions. 


# Expected fault frequencies (multiple of shaft speed)
Drive end: SKF 6205-2RS JEM

BPFI: 5.415
BPFO: 3.585
FTF: 0.3983
BSF: 2.357

# Other tips
Something that is rather confusing is that terms like drive-end and fan-end can refer to two different things.
In one case, if refers to where the faulty bearing is placed for the test.
In the other case, it refers to the location where the measurement is made.

