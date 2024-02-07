import numpy as np

class TriangularPeaks():
    """
    Augment an existing envelope spectrum towards the expected envelope spectrum for a given failure mode.
    """
    def __init__(self, freqs_to_simulate_for,
                 alpha=0.001,
                 traingle_base = 50
                 ):
        self.alpha =alpha
        self.width = traingle_base

        self.freqs_to_simulate_for = freqs_to_simulate_for

        self.augmentation_meta_data = {"triangle_base": traingle_base,
                                       "alpha": alpha,
                                       }
        # self.augmentation = self.get_augmentation(alpha,traingle_base,peak_magnitude)

    def triangular_peak(self, amplitude,fc):
        peak = []
        for f in self.freqs_to_simulate_for:
            if -self.width / 2 <= (f - fc) <= self.width / 2:
                peak.append(-np.abs(-2 * amplitude * (f - fc) / self.width) + amplitude)
            else:
                peak.append(0)
        return np.array(peak)

    def exponential_decay(self, harmonic,fault_frequency):
        return np.exp(-self.alpha * (harmonic - 1) * fault_frequency)

    def get_expected_fault_behaviour(self,peak_magnitude,fault_frequency):
        """
        Add the three peaks and make sure that the amplitude decays in the process
        """

        # Return all zeros if fault_frequency is None (Machine is healthy)
        if fault_frequency is None:
            return np.zeros_like(self.freqs_to_simulate_for)

        augmentation = self.triangular_peak(peak_magnitude,fault_frequency)
        for harmonic in range(2, 20):
            augmentation = augmentation + self.exponential_decay(harmonic,fault_frequency) * self.triangular_peak(
                peak_magnitude, fault_frequency * harmonic)
        return augmentation

if __name__ == "__main__":
    sig = np.arange(0,1000,1)
    freqs = np.fft.rfftfreq(sig.size, d=1/12000)
    obj = TriangularPeaks(freqs_to_simulate_for=freqs)

    # Plot the peaks
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(freqs, obj.get_expected_fault_behaviour(1, 500))
    plt.show()

