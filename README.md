# Dataset Management Repository

Repository for managing and processing open fault detection datasets, mainly bearings and gears. It includes tools for downloading, processing, and standardizing various datasets.

## Overview

- **Purpose**: Standardize mechanical fault detection datasets for machine learning
- **Datasets**: IMS, CWR, MFPT, Paderborn, and others
- **Database**: MongoDB for data storage and retrieval
- **Processing**: Signal processing utilities and feature extraction

## Installation

Set up new environment and install required packages:

```bash
pip install -r requirements.txt
# or
conda install --file requirements.txt
```

## Usage

1. Download dataset using `download_data.py`
2. Process data with `write_data_to_standard_structure.py`
3. Use notebooks for analysis and experimentation

## Structure

- `dataset_management/`: Individual dataset modules
- `utils/`: Core utilities for processing and database operations
- `notebooks/`: Analysis scripts
