# MFPT Bearing Fault Dataset

## Overview
The MFPT (Mechanical Failure Prevention Technology) bearing fault dataset contains vibration data collected from a test rig equipped with a NICE bearing. The dataset includes baseline (healthy) conditions and various fault conditions with different loads.

## Bearing Parameters
- Roller diameter: rd = 0.235
- Pitch diameter: pd = 1.245
- Number of elements: ne = 8
- Contact angle: ca = 0

## Dataset Structure
The dataset is organized into the following categories:

1. **Three Baseline Conditions**
   - 270 lbs of load
   - Input shaft rate of 25 Hz
   - Sample rate of 97,656 sps
   - 6 seconds of data

2. **Three Outer Race Fault Conditions**
   - 270 lbs of load
   - Input shaft rate of 25 Hz
   - Sample rate of 97,656 sps
   - 6 seconds of data

3. **Seven More Outer Race Fault Conditions**
   - Varying loads: 25, 50, 100, 150, 200, 250, and 300 lbs
   - Input shaft rate of 25 Hz
   - Sample rate of 48,828 sps
   - 3 seconds of data

4. **Seven Inner Race Fault Conditions**
   - Varying loads: 0, 50, 100, 150, 200, 250, and 300 lbs
   - Input shaft rate of 25 Hz
   - Sample rate of 48,828 sps
   - 3 seconds of data

5. **Analyses Files**
   - MATLAB (.m) files for data analysis

6. **Real World Examples**
   - Intermediate shaft bearing from a wind turbine
   - Oil pump shaft bearing from a wind turbine
   - Planet bearing fault

## Data Format
The data is stored in MATLAB (.mat) double-precision binary format. Each file contains:
- Load information
- Shaft rate
- Sample rate
- Vibration data in 'g' units

## Usage
To download the dataset, run:
```
python -m dataset_management.mfpt.download_data
```

To convert the data to the standard structure, run:
```
python -m dataset_management.mfpt.write_data_to_standard_structure
```

## Limitations
Note that the baseline condition is only provided for the 270 lbs load case, meaning that a comparison between baseline and faulty conditions is only possible for one load condition for an outer race fault.
Note that there is a discrepancy between the fault frequencies provided and those calculated using bearing parameters. I have had most success with the ones calculated from the bearing geometry.

Also, when plotting the PSD of the normal and faulty data that have the same load of 270 lbs, there is not really a reasonable overlap in the spectra between the normal and faulty data. 
For example, the normal data has some resonance at 14kHz, which somehow disappears under faulty conditions, which is unexpected.

## Source
The dataset was originally published by the MFPT Society and is available at:
https://www.mfpt.org/fault-data-sets/