Ideally, you would have one database that points to all signals that are on your computer. 
The dataset would have the following form

Dataset identification                                                              Links to signals                                                   Repeated meta across samples Sample meta data              Sample specific meta data   

|------------------------------------|---------------------|--------------------|-------------------------------------------|-----------------------|---------------------------------------|----------------------------|---------------------------|
| Dataset name/ Measurement Campaign | Operating condition | Health state /Mode | path to signal channel measurement        | Sample number         | path_to_meta_data_shared_over_samples | [path_to_sample_meta_data] | dict of sample meta data  |
|                                    |                     |                    | [path_to_channel1, path_to_channel2, ...] | 0                     |                                       |                            | 




The current idea is to have a dataset with the following

dataset:
    - dataset_meta_data.json # Mentions naming convention, etc. Can also contain information about data splits, Also channel names. 
    - labels.csv # Maps an index to things like fault mode, operating condition, record number, etc.
    - derived_labels.csv  # Other useful info but not required to uniquely identify a sample
    - class1.hdf5 # Contains all signals of class 1 has shape (batch_size, n_channels, dim1, dim2, ...)


# Todos to achieve this
- [ ] Have a single signal for a condition and split it later, unless it is provided in the split state.
- [ ] Requires standardized filed naming underscores camel case etc.