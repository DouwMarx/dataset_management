import json
import pathlib

import numpy as np

def export_data_to_file_structure(dataset_name: str, healthy_data: np.array, faulty_data_dict: dict,
                                  export_path: pathlib.Path, metadata: dict):
    export_path = pathlib.Path(export_path)

    healthy_data_path = export_path.joinpath(dataset_name, 'healthy')
    faulty_data_path = export_path.joinpath(dataset_name, 'faulty')
    meta_data_path = export_path.joinpath(dataset_name, 'meta_data')

    # Create the directories if they do not exist
    for path in [healthy_data_path, faulty_data_path, meta_data_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Save the data
    np.save(str(healthy_data_path.joinpath('healthy.npy')), healthy_data)
    # np.save(str(healthy_data_path.joinpath('healthy.npy')), healthy_data)
    for severity, faulty_data in faulty_data_dict.items():
        # faulty_data.to_pickle(str(faulty_data_path.joinpath(f'faulty_{severity}.pkl')))
        np.save(str(faulty_data_path.joinpath(f'faulty_{severity}.npy')), faulty_data)

    # Export the metadata to a json file
    with open(str(meta_data_path.joinpath('meta_data.json')), 'w') as f:
        json.dump(metadata, f)

    print(f"Data exported to {export_path}")
