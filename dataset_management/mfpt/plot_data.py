import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import LogNorm

# Import the MFPT class from write_data_to_standard_structure.py
from dataset_management.mfpt.write_data_to_standard_structure import MFPT

"""
This script plots time series and STFT of baseline and first set of outer race fault segments from the MFPT dataset.
It only uses the baseline data (3 files) and the first set of outer race fault data (3 files) with 270 lbs load,
not the other fault datasets with varying loads.
"""

def plot_time_series(data, title, fs, ax=None):
    """
    Plot time series data
    
    :param data: Time series data
    :param title: Plot title
    :param fs: Sampling frequency
    :param ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create time vector
    time = np.arange(len(data)) / fs
    
    # Plot time series
    ax.plot(time, data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (g)')
    ax.set_title(title)
    ax.grid(True)
    
    return ax

def plot_stft(data, title, fs, ax=None):
    """
    Plot Short-Time Fourier Transform (STFT)
    
    :param data: Time series data
    :param title: Plot title
    :param fs: Sampling frequency
    :param ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Compute STFT
    f, t, Zxx = signal.stft(data, fs=fs)#, nperseg=1024, noverlap=512)
    
    # Plot STFT
    im = ax.pcolormesh(t, f, np.abs(Zxx), norm=LogNorm(), shading='gouraud')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    # Limit frequency range to 0-20000 Hz for better visualization
    ax.set_ylim(0, 20000)
    
    return ax

def plot_baseline_and_fault_data():
    """
    Plot time series and STFT of baseline and first set of outer race fault segments.
    
    Note: This only plots data from the baseline conditions and the first set of outer race
    fault conditions (both with 270 lbs load and 97,656 Hz sampling rate). It does not include
    the other fault datasets with varying loads.
    """
    # Initialize MFPT dataset handler
    mfpt_data = MFPT(name="MFPT", overlap=0.0)
    
    # Load baseline and fault data (only the first set of outer race faults with 270 lbs load)
    baseline_data, fault_data = mfpt_data.load_data()
    
    # Create output directory for plots
    plot_dir = pathlib.Path(__file__).parent.joinpath("plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Plotting baseline and first set of outer race fault data (270 lbs load)...")
    print("Note: This script only uses the baseline data and the first set of outer race fault data,")
    print("      not the other fault datasets with varying loads.")
    
    # Plot a few segments from each baseline dataset
    for i, (baseline_segments, baseline_metadata) in enumerate(baseline_data):
        baseline_num = os.path.splitext(baseline_metadata['file_name'])[0].split('_')[-1]
        fs = baseline_metadata['sample_rate']
        
        # Select 2 segments from each baseline dataset
        for seg_idx in [0, 2]:  # Choose first and third segments
            if seg_idx < baseline_segments.shape[0]:
                segment = baseline_segments[seg_idx, 0, :]
                
                # Create figure with 2 subplots (time series and STFT)
                fig, axs = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot time series
                plot_time_series(segment, f"Baseline {baseline_num} - Segment {seg_idx+1} (270 lbs load)", fs, ax=axs[0])
                
                # Plot STFT
                plot_stft(segment, f"STFT - Baseline {baseline_num} - Segment {seg_idx+1} (270 lbs load)", fs, ax=axs[1])
                
                # Adjust layout and save figure
                plt.tight_layout()
                plt.savefig(plot_dir.joinpath(f"baseline_{baseline_num}_segment_{seg_idx+1}.png"))
                plt.close()
    
    # Plot a few segments from each fault dataset (first set of outer race faults with 270 lbs load)
    for i, (fault_segments, fault_metadata) in enumerate(fault_data):
        fault_num = fault_metadata['fault_num']
        fs = fault_metadata['sample_rate']
        
        # Select 2 segments from each fault dataset
        for seg_idx in [0, 2]:  # Choose first and third segments
            if seg_idx < fault_segments.shape[0]:
                segment = fault_segments[seg_idx, 0, :]
                
                # Create figure with 2 subplots (time series and STFT)
                fig, axs = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot time series
                plot_time_series(segment, f"Outer Race Fault {fault_num} - Segment {seg_idx+1} (270 lbs load)", fs, ax=axs[0])
                
                # Plot STFT
                plot_stft(segment, f"STFT - Outer Race Fault {fault_num} - Segment {seg_idx+1} (270 lbs load)", fs, ax=axs[1])
                
                # Adjust layout and save figure
                plt.tight_layout()
                plt.savefig(plot_dir.joinpath(f"fault_{fault_num}_segment_{seg_idx+1}.png"))
                plt.close()
    
    print(f"Plots saved to {plot_dir}")
    
    # Create a comparison plot with one baseline and one fault segment
    if baseline_data and fault_data:
        # Get first segment from first baseline dataset
        baseline_segment = baseline_data[0][0][0, 0, :]
        baseline_fs = baseline_data[0][1]['sample_rate']
        baseline_num = os.path.splitext(baseline_data[0][1]['file_name'])[0].split('_')[-1]
        
        # Get first segment from first fault dataset
        fault_segment = fault_data[0][0][0, 0, :]
        fault_fs = fault_data[0][1]['sample_rate']
        fault_num = fault_data[0][1]['fault_num']
        
        # Create figure with 2x2 subplots (time series and STFT for baseline and fault)
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot baseline time series
        plot_time_series(baseline_segment, f"Baseline {baseline_num} (270 lbs load)", baseline_fs, ax=axs[0, 0])
        
        # Plot baseline STFT
        plot_stft(baseline_segment, f"STFT - Baseline {baseline_num} (270 lbs load)", baseline_fs, ax=axs[1, 0])
        
        # Plot fault time series
        plot_time_series(fault_segment, f"Outer Race Fault {fault_num} (270 lbs load)", fault_fs, ax=axs[0, 1])
        
        # Plot fault STFT
        plot_stft(fault_segment, f"STFT - Outer Race Fault {fault_num} (270 lbs load)", fault_fs, ax=axs[1, 1])
        
        # Add a title to the figure
        fig.suptitle("Comparison of Baseline and Outer Race Fault Data (270 lbs load, 97,656 Hz sampling rate)", fontsize=16)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        plt.savefig(plot_dir.joinpath("baseline_vs_fault_comparison.png"))
        plt.close()
        
        print(f"Comparison plot saved to {plot_dir}")

if __name__ == "__main__":
    plot_baseline_and_fault_data()