import scipy

from dataset_management.ultils.write_data_in_standard_format import export_data_to_file_structure
from file_definitions import biased_anomaly_detection_path

for difficulty_level in ["simple", "moderate", "extreme"]:
    healthy_data = "/home/douwm/data/safran_simulated_gearbox_and_bearing/data/simple_healthy_dataset.mat"
    faulty_data = "/home/douwm/data/safran_simulated_gearbox_and_bearing/data/simple_faulty_dataset.mat"

    healthy_data = scipy.io.loadmat(healthy_data)
    faulty_data = scipy.io.loadmat(faulty_data)

    Fs = healthy_data['Fs'][0, 0]

    # Retrieve healthy and faulty data signals
    x0 = healthy_data['data'][0]
    x = faulty_data['data'][0]

    # Find the number of samples required to ensure a sufficient number of fault events per segment
    n_faut_events_per_segement = 10
    n_samples_required_to_capture_fault_events = n_faut_events_per_segement * healthy_data['Event'][0, 0]
    resulting_number_of_segments = len(x0) // n_samples_required_to_capture_fault_events

    # Export to standard file structure
    metadata = { "sampling_frequency": int(Fs),
                    "n_faut_events_per_segement": int(n_faut_events_per_segement),
                    "n_samples_required_to_capture_fault_events": int(n_samples_required_to_capture_fault_events),
                    "resulting_number_of_segments": int(resulting_number_of_segments),
                  "level_of_difficulty": difficulty_level
                 }
    export_data_to_file_structure("safran_" + difficulty_level, x0, {"faulty":x}, export_path=biased_anomaly_detection_path, metadata=metadata)
