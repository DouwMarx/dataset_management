import numpy as np


def get_required_signal_length_for_required_number_of_events(rpm, lowest_expected_num_faults_per_rev,
                                                             sampling_frequency, number_of_events_required):
    rotation_rate = rpm / 60  # Rev/s
    faults_per_second = lowest_expected_num_faults_per_rev * rotation_rate  # Number of fault events expected per second (faults/rev)*(rev/s) = faults/s
    n_samples_per_fault = sampling_frequency / faults_per_second  # Samples per fault (samples/s) / (faults/s) = samples/fault
    n_samples_required = n_samples_per_fault * number_of_events_required  # Samples required to cover the required number of events
    signal_length = int(np.floor(
        n_samples_required / 2) * 2)  # Ensure that the signals have an even length for the sake of fft-based methods
    return signal_length

def get_number_of_fault_events_for_segment_length(rpm, lowest_expected_num_faults_per_rev, sampling_frequency, segment_length):
    rotation_rate = rpm / 60  # Rev/s
    faults_per_second = lowest_expected_num_faults_per_rev * rotation_rate  # Number of fault events expected per second (faults/rev)*(rev/s) = faults/s
    n_samples_per_fault = sampling_frequency / faults_per_second  # Samples per fault (samples/s) / (faults/s) = samples/fault
    number_of_events_in_segment = segment_length / n_samples_per_fault
    return number_of_events_in_segment