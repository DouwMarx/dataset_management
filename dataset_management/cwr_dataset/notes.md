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

For BPFO, we expect that we will have fault frequency of 3.585*1772/60 = 105.877 Hz (Consistent with plot in Figure 4 of paper by Smith and Randall)

If we want at least 10 transients per sample for inner race, we need a signal duration of 
10*1/159.92 = 0.06253126563281641 seconds 

This corresponds to at least 
12000*0.06253 = 750 samples (fs*t_10_impulses)



The shaft frequency is around 1772/60 = 29.533333333333335 Hz


One thing that is very confusing is that there is a separate mat file for each channel mode operating condition etc. for the faulty data.
However, for the healthy, reference data, there is a single mat file with the same channel and having keys for different runs.

Dataset does not lend iself to be used as multi channel data, because sometimes data is present and other times it is not, the same is true for reference vs. not. 

Deviation from data standard:
Although some of the channels are sampled simultaneously, the data is still stored with channel dimension 1 (batch,1,sequence).
This is since all channels were not always measured and all measurement are accelerometer signals

A min and max speed measuremnt is available inside each of the mat files.
