

In this case we have samples every 10 minuits which technically represents a new health state.

Running at 2000 RPM is 2000/60 = 33.33 Hz rotation rate
Sample rate is 20KHz and there is 20480 Samples in each measurement.

This means that the duration of a measurement is 20480/20000 = 1.024 seconds.

This means that during a single measurement, there will be 33.33*1.024 = 34.13 revolutions of the bearing for each measurement.

Since the sample rate is 20kHz, the highest frequency we could measure is 10KHz due to the nyquist limit 

It is possible to segement the existing signals into multiple samples, but this is not done here.
Instead we group all the reference data (1 second segements), each being a sample and the fault data