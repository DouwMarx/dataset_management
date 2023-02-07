At the end of the day, the data should match the existing data structure where the raw time series is available as a numpy array
* Each row of the time series will represent a sample at a given severity.

In this case we have samples every 10 minuits which technically represents a new healthy state.
This is probably fine for the test data with small sample sizes.
The trianing data will be grouped in larger groups.

Running at 2000 RPM is 2000/60 = 33.33 Hz rotation rate
Sample rate is 20KHz and there is 20480 Samples in each measurement.

This means that the duration of a measurement is 20480/20000 = 1.024 seconds.

This means that during a single measurement, there will be 33.33*1.024 = 34.13 revolutions of the bearing for each measurement.

Since the sample rate is 20kHz, the highest frequency we could measure is 10KHz due to the nyquist limit 

I could chop up the signals into shorter segments but also join the samples from different severities into groups of severities. This is probably the simplest way to start.