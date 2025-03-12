import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt

class SP(object):
    """
    Signal processing class method for the DGEN380 turbofan engine data

    If angular resampling is used, the instantaneous speed must be provided when calculating the test statistic.
    If angular resampling is not used, the mean speed needs to be provided when instantiating the class.

    """
    def __init__(self, mean_speed_estimate_rpm=1, target_order=1, fs=40960):
        """
        Initialize the signal processing class.

        :param mean_speed_estimate_rpm: Estimated mean speed in RPM
        :param target_order: Events per revolution
        :param fs: Sampling frequency in Hz
        """
        self.fs = fs  # [Hz]
        self.mean_speed = mean_speed_estimate_rpm  # [RPM]
        self.target_order = target_order  # [Events per revolution] At what frequency is the fault expected to manifest

    def get_test_statistic(self, measurement, **kwargs):
        """
        Calculate the test statistic for set of signals

        :param measurement: Set of input signals of shape (n_signals,n_channels, n_samples)
                 If angular resampling is used, the speed in RPM is required in the second channel
        :param order: Events per revolution. Default is 14.
        :return: The mean value of the demodulated signal.
        """



        measurement,theta_vec = self.get_signal_as_function_of_revolutions(measurement)
        demodulated_signal = self.demodulate_at_target_order(measurement,theta_vec,self.target_order)
        energy_at_target_order = np.abs(np.mean(demodulated_signal,axis=-1))  # Get magnitude of complex valued
        return energy_at_target_order

    def demodulate_at_target_order(self, signal, theta_vec, target_order):
        """
        Demodulate the signal at the target order.

        :param signal: The signal to demodulate
        :param theta_vec: Array of angles of rotation as a function of time [revolutions]
        :param target_order: The target order for demodulation [Events per revolution]
        :return: The demodulated signal
        """
        return signal * np.exp(-1j * 2 * np.pi * target_order * theta_vec)  # Like normal Fourier transform as a function of revolutions, rather than seconds

    def get_signal_as_function_of_revolutions(self, measurement):
        """
        Get the signal as a function of revolutions

        :param measurement: The measurement to process
        :return: The signal as a function of revolutions and the theta vector
        """
        raise NotImplementedError



class NoAngularResampleSP(SP):
    """
    Signal processing class that does not perform angular resampling, requires prescription of the mean speed when instantiating the class
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def get_signal_as_function_of_revolutions(self, measurement):

        # Warn user that speed is not taken into account in case channel dimension is bigger than 1
        if measurement.shape[1] > 1:
            print("Warning: No angular resampling is done. The second channel is ignored.")

        # simple case where signal is un-modified
        # In this case, theta is linearly spaced according to the mean speed, number of samples and sampling rate
        t_duration = measurement.shape[-1] / self.fs # samples / (samples / s) = s
        n_total_revolutions = self.mean_speed * t_duration / 60 # (revs / min) * (min / s) = revs
        theta_vec = np.linspace(0, n_total_revolutions, measurement.shape[-1]) # [revs]

        # Make it match shape (batch, 1, samples)
        theta_vec = np.repeat(theta_vec, measurement.shape[0]).reshape(measurement.shape[0], 1, measurement.shape[-1])
        return measurement, theta_vec


class AngularResampleSP(SP):
    """
    Signal processing class that performs angular resampling, requires prescription of the instantaneous speed when computing test statistics
    """
    def __init__(self, desired_samples_per_revolution=None, **kwargs):
        super().__init__(**kwargs)
        self.desired_samples_per_revolution = desired_samples_per_revolution # What is the resolution at which the signal is sampled in the angular domain


    def get_theta_as_function_of_time(self, instantaneous_speed_rpm):
        """
        Calculates the angle of rotation as a function of time.

        :param instantaneous_speed_rpm: speed measurement (batch, 1, samples)
        :return: Array of angles of rotation as a function of time (batch, 1, samples) [revolutions]
        """
        instantaneous_speed_rps = instantaneous_speed_rpm / 60  # [revolutions per second]
        theta = np.cumsum(instantaneous_speed_rps, axis=-1) / self.fs  # (revs / s) * (s/1) = revs
        return theta

    def get_signal_as_function_of_revolutions(self, measurement):

        signal_of_t = measurement[:, 0, :] # The first channel is the signal
        instantaneous_speed = measurement[:, 1, :] # The second channel is the speed measurement

        theta_of_t = self.get_theta_as_function_of_time(instantaneous_speed)

        # Do angular resampling
        interp_model = interp.interp1d(theta_of_t, signal_of_t, kind='linear', fill_value='extrapolate')

        chosen_thetas = np.arange(np.min(theta_of_t), np.max(theta_of_t), 1 / self.desired_samples_per_revolution)
        signal_of_theta = interp_model(chosen_thetas) # Evaluate the interpolation model at the chosen thetas
        return signal_of_theta, chosen_thetas

class SimpleDemodulatedEnergySP(object):
    def __init__(self,target_order=1,fs=1,mean_rpm=1):
        self.target_order = target_order
        self.fs = fs
        self.mean_rpm = mean_rpm

    def simple_demodulated_energy(self, measurement):
        t_duration = measurement.shape[-1] / self.fs  # samples / (samples / s) = s
        n_total_revolutions = self.mean_rpm * t_duration / 60  # (revs / min) * (min / s) = revs
        theta_vec = np.linspace(0, n_total_revolutions, measurement.shape[-1])  # [revs]

        demodulated_signal = measurement * np.exp(-1j * 2 * np.pi * self.target_order * theta_vec)
        energy_at_target_order = np.abs(np.mean(demodulated_signal, axis=-1))  # Get magnitude of complex valued
        return energy_at_target_order

    def get_test_statistic(self, measurement):
        return self.simple_demodulated_energy(measurement)

if __name__ == "__main__":
    import numpy as np
    from scipy import interpolate as interp
    import matplotlib.pyplot as plt

    # Parameters for test
    fs = 40960  # Sampling frequency (Hz)
    duration = 1.0  # Duration of signal (seconds)
    fault_order = 9  # Target order for detection
    n_measurements = 30  # Number of measurements
    n_samples = int(fs * duration)  # Number of samples
    time_vector = np.linspace(0, duration, n_samples).reshape(1, 1, n_samples)

    for operating_regime, health_state in zip(['stationary','non_stationary'],["normal","faulty"]):
        constant_speed = 20  # RPM
        if operating_regime == 'stationary':
            rpm_as_function_of_time = np.ones((n_measurements,1,n_samples)) * constant_speed
            # Define the SP object
        else:
            n_components = 4
            random_amplitudes = np.random.uniform(0, 1 , (n_measurements, n_components)).reshape(n_measurements, n_components, 1)
            random_frequencies = np.random.uniform(1, 5, (n_measurements, n_components)).reshape(n_measurements, n_components, 1)
            random_phases = np.random.uniform(0, 2 * np.pi, (n_measurements, n_components)).reshape(n_measurements, n_components, 1)
            random_offsets = np.random.uniform(0.9*constant_speed, 1.1*constant_speed, (n_measurements, 1)).reshape(n_measurements, 1, 1)


            rpm_as_function_of_time= np.sum(random_amplitudes * np.sin(random_frequencies * time_vector + random_phases), axis=1, keepdims=True) + random_offsets

        rps_as_function_of_time = rpm_as_function_of_time / 60  #  (revolutions per minute) * (minutes / second) = revolutions per second
        thetas =  np.cumsum(rps_as_function_of_time, axis=-1) / fs  # (revolutions per second) * (seconds / sample) = revolutions per sample
        fault_component = np.sin(2 * np.pi * fault_order * thetas)  # The fault component

         # Plot all the speed profiles
        subset_to_plot = 5
        plt.figure()
        for i in range(subset_to_plot):
            plt.plot(time_vector[0,0,:],rpm_as_function_of_time[i,0,:],label='Measurement {}'.format(i))

        plt.xlabel('Time (s)')
        plt.ylabel('Speed (RPM)')
        plt.title( '{} speed profiles'.format(operating_regime))
        plt.legend()

        # Plot the fault component
        plt.figure()
        for i in range(subset_to_plot):
            plt.plot(time_vector[0,0,:],fault_component[i,0,:],label='Measurement {}'.format(i))
        plt.xlabel('Time (s)')
        plt.ylabel('Fault component')
        plt.title('{} fault component'.format(operating_regime))
        plt.legend()

        # Generate some measurement noise
        noise_component_normal = np.random.normal(0, 1, (n_measurements, 1, n_samples))
        noise_component_faulty = np.random.normal(0, 1, (n_measurements, 1, n_samples))
        # Generate the signals for evaluation
        normal_signal = noise_component_normal
        faulty_signal = fault_component + noise_component_faulty

        # Calculate the test statistics
        normal_test_statistic = simple_demodulated_energy(normal_signal, target_order=fault_order, fs=fs, mean_rpm=constant_speed)
        faulty_test_statistic = simple_demodulated_energy(faulty_signal, target_order=fault_order, fs=fs, mean_rpm=constant_speed)

        # Calculate deflection coefficient
        deflection_coefficient = np.mean(faulty_test_statistic) - np.mean(normal_test_statistic)
        print("Deflection coefficient for {} regime: {}".format(operating_regime,deflection_coefficient))

        # Plot the test statistics as histograms
        plt.figure()
        plt.hist(normal_test_statistic,label='Normal')
        plt.hist(faulty_test_statistic,label='Faulty')
        plt.title('{} test statistic'.format(operating_regime))
        plt.xlabel('Test statistic')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()











