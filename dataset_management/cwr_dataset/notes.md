# Working with wost case data 

See 

https://engineering.case.edu/bearingdatacenter/normal-baseline-data

and 

https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data

DE - drive end accelerometer data
FE - fan end accelerometer data
BA - base accelerometer data
time - time series data
RPM - rpm during testing

Dataset that has detectable ball fault
12k data, Drive end (DE)

Record numbers are:
Inner:            210 
Ball:             223
Outer (Centered): 235
Healthy            98

These are all at motor load 1hp and 1772 RPM.
Fault severity is 0.021inches or 0.53mm


Expected fault frequencies (multiple of shaft speed)

Drive end: SKF 6205-2RS JEM

BPFI: 5.415
BPFO: 3.585
FTF: 0.3983
BSF: 2.357

To capture BFPI, we expect that we will have fault frequency of 5.415*1772/60 = 159.92299999999997 Hz 
For BPFO, we expect that we will have fault frequency of 3.585*1772/60 = 105.877 Hz (Consistent with plot in Figure 4 of paper by Smith and Randall)

If we want at least 10 transients per sample for inner race, we need a signal duration of 
10*1/159.92 = 0.06253126563281641 seconds 

This corresponds to at least 
12000*0.06253 = 750 samples (fs*t_10_impulses)