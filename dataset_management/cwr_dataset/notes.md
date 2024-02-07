# Source

See 

https://engineering.case.edu/bearingdatacenter/normal-baseline-data

and 

https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data

DE - drive end accelerometer data
FE - fan end accelerometer data
BA - base accelerometer data
time - time series data
RPM - rpm during testing


# Example dataset for validation
Dataset that have detectable fault for all modes (According to Randall and Smith, 2009) are:
12k data, Drive end (DE)
Record numbers are:
Inner:            210 
Ball:             223
Outer (Centered): 235
Healthy            98
These are all at motor load 1hp and 1772 RPM.
Fault severity is 0.021inches or 0.53mm


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
