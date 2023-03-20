import numpy as np

def get_signal_length_for_number_of_events(rpm,n_faults_per_rev,fs,n_events):
    rotation_rate = rpm / 60  # Rev/s
    faults_per_second = n_faults_per_rev * rotation_rate
    n_samples_per_fault = fs / faults_per_second
    n_samples_required = n_samples_per_fault * n_events

    signal_length = int(np.floor(n_samples_required / 2) * 2)  # Ensure that the signals have an even length
    return signal_length

