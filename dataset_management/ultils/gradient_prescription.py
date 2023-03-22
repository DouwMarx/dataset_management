import numpy as np
from matplotlib import pyplot as plt


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
        augmentation = self.triangular_peak(peak_magnitude,fault_frequency)
        for harmonic in range(2, 20):
            augmentation = augmentation + self.exponential_decay(harmonic,fault_frequency) * self.triangular_peak(
                peak_magnitude, fault_frequency * harmonic)
        return augmentation

# sig = np.arange(0,1000,1)
#
# freqs = np.fft.rfftfreq(sig.size, d=1/12000)
# obj = TriangularPeaks(freqs_to_simulate_for=freqs)
#
# # Plot the peaks
# plt.figure()
# plt.plot(freqs, obj.get_augmentation(1, 500))
# plt.show()

#
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# class TriangularPeaks():
#     """
#     Augment an existing envelope spectrum towards the expected envelope spectrum for a given failure mode.
#     """
#     def __init__(self,healthy_ses_freq=None,
#                  alpha= 0.005,
#                  traingle_base = 10
#                  ):
#         self.alpha =alpha
#         self.width = traingle_base
#
#         self.healthy_ses_freq = healthy_ses_freq
#
#         self.augmentation_meta_data = {"triangle_base": traingle_base,
#                                        "alpha": alpha,
#                                        }
#         # self.augmentation = self.get_augmentation(alpha,traingle_base,peak_magnitude)
#
#     def triangular_peak(self, amplitude,fc):
#         peak = []
#         for f in self.healthy_ses_freq:
#             if -self.width / 2 <= (f - fc) <= self.width / 2:
#                 # peak.append(-np.abs(-2 * amplitude * (f - fc) / self.width) + amplitude)
#                 peak.append(amplitude)
#             else:
#                 peak.append(0)
#         return np.array(peak)
#
#     def exponential_decay(self, harmonic,fault_frequency):
#         return np.exp(-self.alpha * (harmonic - 1) * fault_frequency)
#
#     def get_augmentation(self,peak_magnitude,fault_frequency):
#         """
#         Add the three peaks and make sure that the amplitude decays in the process
#         """
#         augmentation = self.triangular_peak(peak_magnitude,fault_frequency)
#         for harmonic in range(2, 20):
#             augmentation = augmentation + self.exponential_decay(harmonic,fault_frequency) * self.triangular_peak(
#                 peak_magnitude, fault_frequency * harmonic)
#         return augmentation
#
# def get_cwr_expected_fault_behaviour(siglen,rpm,plot=False):
#
#     rotation_rate = rpm / 60
#     expected_fault_frequencies = {"ball": 2.357 * rotation_rate,
#                                   "outer": 3.585 * rotation_rate,
#                                   "inner": 5.415 * rotation_rate}
#     sig = np.arange(0,siglen,1)
#
#     freqs = np.fft.rfftfreq(sig.size*2, d=1/12000)
#     freqs = freqs[0:siglen]
#     obj = TriangularPeaks(healthy_ses_freq=freqs)
#
#     peaks = 0
#     for fault_freq in expected_fault_frequencies.values():
#         peaks = peaks + obj.get_augmentation(0.98, fault_freq)
#         # break
#     peaks+=0.02
#
#     if plot:
#         # Plot the peaks
#         plt.figure()
#         plt.plot(freqs, peaks)
#
#     return peaks
