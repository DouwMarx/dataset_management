import os
import scipy.io

"""
Scrip for loading full dataset, not the reduced dataset that is publically available.
"""

class DesirDatasetLoader:
    """
    A class to load samples from the DESIR dataset.

    Attributes:
    -----------
    root_dir : str
        The root directory where the dataset files are located.
    configurations : list
        A list of available configurations in the dataset.

    Methods:
    --------
    get_configurations():
        Retrieves the list of available configurations in the dataset.
    get_measurement_types(configuration):
        Retrieves the measurement types ('palier' or 'rampe') for a given configuration.
    get_data_types(configuration, measurement_type):
        Retrieves the data types ('donnees_fil_chaud', 'donnees_perfo', 'donnees_vibro_acoustique') for a given configuration and measurement type.
    get_measurements(configuration, measurement_type, data_type):
        Retrieves the available measurements for a given configuration, measurement type, and data type.
    get_runs(configuration, measurement_type, data_type, measurement_name):
        Retrieves the available runs for a given configuration, measurement type, data type, and measurement.
    load_data(configuration, measurement_type, data_type, measurement_name, run_number):
        Loads the data file for the specified parameters.
    """

    def __init__(self, root_dir):
        """
        Initializes the DesirDatasetLoader with the given root directory.

        Parameters:
        -----------
        root_dir : str
            The root directory where the dataset files are located.
        """
        self.root_dir = root_dir
        self.configurations = self.get_configurations()

    def get_configurations(self):
        """
        Retrieves the list of available configurations in the dataset.

        Returns:
        --------
        configurations : list
            List of configuration directory names (e.g., ['configuration_1', 'configuration_2', ...])
        """
        configurations = []
        for entry in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, entry)) and entry.startswith('configuration_'):
                configurations.append(entry)
        configurations.sort()
        return configurations

    def get_measurement_types(self, configuration):
        """
        Retrieves the measurement types ('palier' or 'rampe') for a given configuration.

        Parameters:
        -----------
        configuration : str
            The configuration directory name.

        Returns:
        --------
        measurement_types : list
            List of measurement types available for the configuration.
        """
        measurement_types = []
        config_path = os.path.join(self.root_dir, configuration, configuration) # The file is nested with configuration/configuration
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration {configuration} does not exist.")
        for entry in os.listdir(config_path):
            if os.path.isdir(os.path.join(config_path, entry)) and entry in ['palier', 'rampe']:
                measurement_types.append(entry)
        measurement_types.sort()
        return measurement_types

    def get_data_types(self, configuration, measurement_type):
        """
        Retrieves the data types for a given configuration and measurement type.

        Parameters:
        -----------
        configuration : str
            The configuration directory name.
        measurement_type : str
            The measurement type ('palier' or 'rampe').

        Returns:
        --------
        data_types : list
            List of data types available for the configuration and measurement type.
        """
        data_types = []
        measurement_path = os.path.join(self.root_dir, configuration,configuration, measurement_type)
        if not os.path.exists(measurement_path):
            raise ValueError(f"Measurement type {measurement_type} does not exist in configuration {configuration}.")
        for entry in os.listdir(measurement_path):
            if os.path.isdir(os.path.join(measurement_path, entry)):
                data_types.append(entry)
        data_types.sort()
        return data_types

    def get_measurements(self, configuration, measurement_type, data_type):
        """
        Retrieves the available measurements for a given configuration, measurement type, and data type.

        Parameters:
        -----------
        configuration : str
            The configuration directory name.
        measurement_type : str
            The measurement type ('palier' or 'rampe').
        data_type : str
            The data type ('donnees_fil_chaud', 'donnees_perfo', 'donnees_vibro_acoustique').

        Returns:
        --------
        measurements : list
            List of measurement names available.
        """
        measurements = set()
        data_path = os.path.join(self.root_dir, configuration,configuration, measurement_type, data_type)
        if not os.path.exists(data_path):
            raise ValueError(
                f"Data type {data_type} does not exist in {measurement_type} of configuration {configuration}.")
        for filename in os.listdir(data_path):
            if filename.endswith('.mat'):
                parts = filename.split('_')
                # Find the measurement name by excluding known parts
                # Filename format varies between data types
                if data_type == 'donnees_fil_chaud':
                    # Example: Desir_II_config_1_palier_inter1_bis_FC_1.mat
                    try:
                        idx = parts.index('config') + 3  # Skip 'Desir', 'II', 'config', '1'
                        measurement_name = '_'.join(parts[idx:-2])  # Exclude 'FC', '1.mat'
                        measurements.add(measurement_name)
                    except ValueError:
                        continue  # Unexpected filename format
                elif data_type == 'donnees_perfo':
                    # Example: Desir_II_config_1_palier_inter1_bis_perfo_1.mat
                    try:
                        idx = parts.index('config') + 3
                        measurement_name = '_'.join(parts[idx:-2])  # Exclude 'perfo', '1.mat'
                        measurements.add(measurement_name)
                    except ValueError:
                        continue
                elif data_type in ['donnees_vibro_acoustique', 'donnees_vibro_acoustiques']:
                    # Example: Desir_II_configuration_1_palier_inter1_bis_VA_1.mat
                    try:
                        idx = parts.index('configuration') + 3
                        measurement_name = '_'.join(parts[idx:-2])  # Exclude 'VA', '1.mat'
                        measurements.add(measurement_name)
                    except ValueError:
                        continue
                else:
                    continue
        measurements = sorted(measurements)
        return measurements

    def get_runs(self, configuration, measurement_type, data_type, measurement_name):
        """
        Retrieves the available runs for a given configuration, measurement type, data type, and measurement.

        Parameters:
        -----------
        configuration : str
            The configuration directory name.
        measurement_type : str
            The measurement type ('palier' or 'rampe').
        data_type : str
            The data type ('donnees_fil_chaud', 'donnees_perfo', 'donnees_vibro_acoustique').
        measurement_name : str
            The name of the measurement.

        Returns:
        --------
        runs : list
            List of run numbers available for the measurement.
        """
        runs = []
        data_path = os.path.join(self.root_dir, configuration, configuration, measurement_type, data_type)
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        for filename in os.listdir(data_path):
            if filename.endswith('.mat'):
                if measurement_name in filename:
                    # Extract the run number from the filename
                    base_name = filename[:-4]  # Remove '.mat'
                    parts = base_name.split('_')
                    run_str = parts[-1]
                    if run_str.isdigit():
                        runs.append(int(run_str))
        runs = sorted(runs)
        return runs

    def load_data(self, configuration, measurement_type, data_type, measurement_name, run_number):
        """
        Loads the data file for the specified parameters.

        Parameters:
        -----------
        configuration : str
            The configuration directory name (e.g., 'configuration_1').
        measurement_type : str
            The measurement type ('palier' or 'rampe').
        data_type : str
            The data type ('donnees_fil_chaud', 'donnees_perfo', 'donnees_vibro_acoustique').
        measurement_name : str
            The name of the measurement (e.g., 'inter1_bis', 'ventil', etc.).
        run_number : int
            The run number to load.

        Returns:
        --------
        data : dict
            The data loaded from the .mat file.

        Raises:
        -------
        FileNotFoundError:
            If the data file does not exist.
        """
        # Build the file name based on parameters
        if data_type == 'donnees_fil_chaud':
            # Example: Desir_II_config_1_palier_inter1_bis_FC_1.mat
            filename = f"Desir_II_config_{configuration[-1]}_{measurement_type}_{measurement_name}_FC_{run_number}.mat"
        elif data_type == 'donnees_perfo':
            # Example: DESIR_II_config_1_palier_inter1_bis_perfo_1.mat
            filename = f"DESIR_II_config_{configuration[-1]}_{measurement_type}_{measurement_name}_perfo_{run_number}.mat"
        elif data_type in ['donnees_vibro_acoustique', 'donnees_vibro_acoustiques']:
            # The data_type folder might be plural or singular
            # Example: Desir_II_configuration_1_palier_inter1_bis_VA_1.mat
            filename = f"Desir_II_configuration_{configuration[-1]}_{measurement_type}_ventil_VA_{run_number}.mat"
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        file_path = os.path.join(self.root_dir, configuration,configuration, measurement_type, data_type, filename)

        if os.path.exists(file_path):
            data = scipy.io.loadmat(file_path)
            return data
        else:
            raise FileNotFoundError(f"The data file {file_path} does not exist.")

    # Desir_II_configuration_1_palier_bruit_fond_VA_1.mat
    def get_all_runs_for_measurement(self, configuration, measurement_type, data_type, measurement_name):
        """
        Retrieves a dictionary of run numbers and their corresponding file paths for a measurement.

        Parameters:
        -----------
        configuration : str
            The configuration directory name.
        measurement_type : str
            The measurement type ('palier' or 'rampe').
        data_type : str
            The data type.
        measurement_name : str
            Name of the measurement.

        Returns:
        --------
        run_files : dict
            Dictionary with run numbers as keys and file paths as values.
        """
        run_files = {}
        data_path = os.path.join(self.root_dir, configuration, measurement_type, data_type)
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        for filename in os.listdir(data_path):
            if filename.endswith('.mat') and measurement_name in filename:
                base_name = filename[:-4]  # Remove '.mat'
                parts = base_name.split('_')
                run_str = parts[-1]
                if run_str.isdigit():
                    run_number = int(run_str)
                    file_path = os.path.join(data_path, filename)
                    run_files[run_number] = file_path
        return run_files

    def check_data_consistency(self):
        """
        Checks for inconsistencies between the data available in the file structure and the description.

        Returns:
        --------
        inconsistencies : list
            List of strings describing inconsistencies found.
        """
        inconsistencies = []

        # The description mentions 4 configurations, but there are 8 configurations
        expected_configs = ['configuration_' + str(i) for i in range(1, 5)]
        actual_configs = self.configurations
        if set(expected_configs) != set(actual_configs[:4]):
            inconsistencies.append(
                f"Description mentions 4 configurations {expected_configs}, but found {actual_configs}")

        # Check for missing data in configurations
        for config in self.configurations:
            measurement_types = self.get_measurement_types(config)
            for m_type in measurement_types:
                data_types = self.get_data_types(config, m_type)
                for d_type in data_types:
                    measurements = self.get_measurements(config, m_type, d_type)
                    if not measurements:
                        inconsistencies.append(f"No measurements found in {d_type} of {m_type} in {config}")
                    for measurement in measurements:
                        runs = self.get_runs(config, m_type, d_type, measurement)
                        if not runs:
                            inconsistencies.append(
                                f"No runs found for measurement {measurement} in {d_type} of {m_type} in {config}")

        return inconsistencies


# Example usage:
if __name__ == '__main__':
    # Suppose the dataset is located in './dataset/'
    loader = DesirDatasetLoader('/run/media/douwm/DATA/DGEN380/')

    # List configurations
    print("Available configurations:", loader.configurations)

    # Get measurement types for a configuration
    config = 'configuration_1'
    measurement_types = loader.get_measurement_types(config)
    print(f"Measurement types in {config}:", measurement_types)

    # Get data types for a measurement type
    measurement_type = 'palier'
    data_types = loader.get_data_types(config, measurement_type)
    print(f"Data types in {measurement_type} of {config}:", data_types)

    # Get measurements for data type
    data_type = 'donnees_vibro_acoustique'
    measurements = loader.get_measurements(config, measurement_type, data_type)
    print(f"Measurements in {data_type} of {measurement_type} in {config}:", measurements)

    # Get runs for a measurement
    measurement_name = 'inter1_bis'
    runs = loader.get_runs(config, measurement_type, data_type, measurement_name)
    print(f"Runs for measurement type {measurement_name} in {data_type} of {measurement_type} in {config}:", runs)

    # Load a data file
    run_number = runs[0]  # Load the first available run
    data = loader.load_data(config, measurement_type, data_type, measurement_name, run_number)

    # Plot the "acc1_X" data
    import matplotlib.pyplot as plt
    plt.plot(data['t'].flatten(),data['acc1_X'].flatten())
    plt.show()

    print(f"Loaded data for run {run_number} of measurement {measurement_name} in {config}")

    # Check for inconsistencies
    inconsistencies = loader.check_data_consistency()
    if inconsistencies:
        print("Inconsistencies found:")
        for inc in inconsistencies:
            print("-", inc)
    else:
        print("No inconsistencies found.")
