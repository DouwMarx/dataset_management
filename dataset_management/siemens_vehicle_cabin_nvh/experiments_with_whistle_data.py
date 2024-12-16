import pathlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

data_dir = pathlib.Path("/home/douwm/data/siemens_isma_2024")
for vehicle in ["Mondeo",  "Vectra"]:
    whistle_data = np.load(data_dir.joinpath(vehicle + "_whistle_sounds.npy"))
    healthy_data = np.load(data_dir.joinpath(vehicle + "_normal_sounds.npy"))


    random_index = np.random.randint(0, len(whistle_data))
    whistle_example = whistle_data[random_index].flatten()
    healthy_example = healthy_data[random_index].flatten()

    # Plot time domain example
    plt.figure()
    plt.plot(whistle_example, label="Whistle")
    plt.plot(healthy_example, label="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()

    # Plot frequency domain example
    plt.figure()
    whistle_example_freq = np.fft.rfft(whistle_example)
    healthy_example_freq = np.fft.rfft(healthy_example)
    freqs = np.fft.rfftfreq(len(whistle_example), d=1/44100)
    plt.plot(freqs, np.abs(whistle_example_freq), label="Whistle")
    plt.plot(freqs, np.abs(healthy_example_freq), label="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()

    # Band pass filter between 1000 and 10000 Hz
    low_cut = 1000
    high_cut = 10000

    def bandpass(signals, lower, upper, fs, order=6):
        b, a = butter(order, [lower / (0.5 * fs), upper / (0.5 * fs)], btype='band')
        return filtfilt(b, a, signals, axis=-1)

    whistle_example_bandpass = bandpass(whistle_example, low_cut, high_cut, 44100)
    healthy_example_bandpass = bandpass(healthy_example, low_cut, high_cut, 44100)

    # Plot time domain example
    plt.figure()
    plt.plot(whistle_example_bandpass, label="Whistle")
    plt.plot(healthy_example_bandpass, label="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()

    # Plot frequency domain example
    plt.figure()
    whistle_example_bandpass_freq = np.abs(np.fft.rfft(whistle_example_bandpass))
    healthy_example_bandpass_freq = np.abs(np.fft.rfft(healthy_example_bandpass))
    freqs = np.fft.rfftfreq(whistle_example_bandpass.shape[-1], d=1/44100)
    plt.plot(freqs, whistle_example_bandpass_freq, label="Whistle")
    plt.plot(freqs, healthy_example_bandpass_freq, label="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()

    # apply a high-pass filter on the frequency domaindata  to remove the trend
    def highpass(signals, lower, fs, order=6):
        b, a = butter(order, lower / (0.5 * fs), btype='high')
        return filtfilt(b, a, signals, axis=-1)

    # Apply high-pass filter
    whistle_example_highpass = highpass(whistle_example_bandpass_freq, 100, len(whistle_example_bandpass_freq))
    healthy_example_highpass = highpass(healthy_example_bandpass_freq, 100, len(healthy_example_bandpass_freq))

    # Plot the filtered frequency domain example
    plt.figure()
    plt.plot(freqs, whistle_example_highpass, label="Whistle")
    plt.plot(freqs, healthy_example_highpass, label="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()

    # # Instead do moving average filter on the frequency domain amplitude
    # def moving_average(signals, window_size):
    #     return np.convolve(signals, np.ones(window_size)/window_size, mode='same')
    # window_size = 100
    # whistle_example_moving_average = moving_average(whistle_example_bandpass_freq, window_size)
    # healthy_example_moving_average = moving_average(healthy_example_bandpass_freq, window_size)
    #
    # Plot the filtered frequency domain example
    # plt.figure()
    # plt.plot(freqs, whistle_example_moving_average, label="Whistle")
    # plt.plot(freqs, healthy_example_moving_average, label="Healthy")
    # plt.legend()
    # plt.title(vehicle)
    # plt.show()

    # Instead do a bandpass filter on the frequency domain amplitude

    # Apply bandpass filter
    # Leave out anyting slower than 1 oscillation per 50 samples and faster than 1 oscillation per 20 samples
    whistle_example_bandpass_freq = bandpass(whistle_example_bandpass_freq, 1/100, 1/10, 1) # 1 oscillation per 50 samples to 1 oscillation per 20 samples
    healthy_example_bandpass_freq = bandpass(healthy_example_bandpass_freq, 1/100, 1/10, 1)

    # Plot the filtered frequency domain example
    plt.figure()
    plt.plot(freqs, whistle_example_bandpass_freq, label
    ="Whistle")
    plt.plot(freqs, healthy_example_bandpass_freq, label
    ="Healthy")
    plt.legend()
    plt.title(vehicle)
    plt.show()







