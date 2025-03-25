import pathlib

import numpy as np
from scipy import io as sio
from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import biased_anomaly_detection_path




def cut_into_non_overlapping_segments(signal, segment_length=1000):
    """
    Cuts a signal into non-overlapping segments of length segment_length
    :param signal: The signal to be cut
    :return: Array of segments
    """
    num_segments = len(signal) // segment_length
    segments = signal[:num_segments * segment_length].reshape(num_segments, segment_length)
    return segments

def get_data(condition,
             stationarity,
             channel,
             segment_length=1000,
             dataset_path="/home/douwm/data/DGEN380_turbofan/",
             decimate_factor=1,
             ):
    # Construct file paths based on condition and stationarity
    stationarity_suffix = '1' if stationarity == 'stationary' else '2'
    config_num = '2' if condition == 'normal' else '4' # config 2 for normal, 4 for faulty

    dataset_path = pathlib.Path(dataset_path)

    data_path = f"desir_ii_configuration_{config_num}_{condition}_{stationarity_suffix}.mat"
    performance_path = f"desir_ii_config_{config_num}_perfo_{condition}_{stationarity_suffix}.mat"

    data_path = dataset_path.joinpath(data_path)
    performance_path = dataset_path.joinpath(performance_path)

    # Load data
    data = sio.loadmat(data_path.__str__())

    performance = sio.loadmat(performance_path.__str__())

    # print("Channels available in meusurement data: ", data.keys())
    # print("Channels available in performance data: ", performance.keys())

    full_measurement = data[channel].astype(float).flatten()
    time_at_speed = performance['t'].flatten()
    # speed_rpm = performance['NH'].flatten()
    speed_rpm = performance['NL'].flatten()  # Use the low pressure rotor speed (Connected to the fan) since this is the relative rotational speed relative to the staionary OGV blades following directly after the fan

    if decimate_factor > 1:
        from scipy.signal import decimate
        full_measurement = decimate(full_measurement, decimate_factor)
        fs = 40960 // decimate_factor
    else:
        fs = 40960

    measurement_segments = cut_into_non_overlapping_segments(full_measurement, segment_length)

    return {'full_measurement': full_measurement,
            'time_at_speed': time_at_speed,
             'speed_rpm': speed_rpm,
              'measurement_segments': measurement_segments,
            # 'speed_segments_rpm': speed_segments_rpm
            'mean_rpm': np.mean(speed_rpm),
            "fs": fs
            }

if __name__ == "__main__":
    channel = "acc1_Y" # , "acc1_X"
    stationarity = "stationary"
    segment_length = 2048

    normal_data = get_data("normal", stationarity, channel, segment_length=segment_length, decimate_factor=1)
    faulty_data = get_data("faulty", stationarity, channel, segment_length=segment_length, decimate_factor=1)

    mean_normal_rpm = normal_data['mean_rpm']
    mean_faulty_rpm = faulty_data['mean_rpm']

    normal_segments = normal_data['measurement_segments']
    faulty_segments = faulty_data['measurement_segments']

    # add a channel dimension (batch, dim) -> (batch, 1, dim)
    normal_segments = np.expand_dims(normal_segments, axis=1)
    faulty_segments = np.expand_dims(faulty_segments, axis=1)

    faulty_data = {
        "ogv_blades": faulty_segments
    }

    meta_data = {
        "sampling_frequency": normal_data['fs'],
        "segment_length": segment_length,
        "mean_normal_rpm": mean_normal_rpm,
        "mean_faulty_rpm": mean_faulty_rpm,
        "mean_rpm" : np.mean([mean_normal_rpm, mean_faulty_rpm])
    }

    export_data_to_file_structure(dataset_name= "DGEN380_" + channel + "_" + stationarity,
                                    healthy_data=normal_segments,
                                    faulty_data_dict=faulty_data,
                                    export_path=biased_anomaly_detection_path,
                                    metadata=meta_data
                                    )