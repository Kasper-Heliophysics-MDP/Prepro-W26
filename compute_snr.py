"""
===============================================================================
File: compute_snr.py
Author: Callen Fields (fcallen@umich.edu), Aashi Mishra (aashim@umich.edu)
Date: 2025-09-29
Group: University of Michigan SunRISE Mission

Description:
This script takes two npy files as command-line arguments. The first is a spectrogram
of solar energy over time. The second is a dict describing the indices at which 
bursts occur (see one_day.py --save_burst_labels). This dict indicates the burst zones
of the spectrogram. This script first takes the mean at each times step to create a
two-dimensional graph of flux. It then finds the average flux value within the burst
zones, and compares that to the average flux value within the non-burst zones to 
calculate the signal-to-noise ratio.
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

def compute_snr(spectrogram, burst_labels):
    """
    Compute SNR of a spectrogram given burst index ranges.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        2D array (n_freqs, n_times) of intensities.
    burst_labels : list of dicts
        Each dict has {"burst": str, "start_idx": int, "end_idx": int}.
    
    Returns
    -------
    snr_db : float
        Signal-to-noise ratio in dB.
    signal_mean : float
        Average signal level.
    noise_mean : float
        Average noise level.
    """
    n_times = spectrogram.shape[1]
    flux_time = spectrogram.mean(axis=0)  # collapse freqs → flux vs time

    # Mask for burst times
    burst_mask = np.zeros(n_times, dtype=bool)
    for entry in burst_labels:
        start = max(0, entry["start_idx"])
        end = min(n_times - 1, entry["end_idx"])
        burst_mask[start:end+1] = True

    inside_flux = flux_time[burst_mask]
    outside_flux = flux_time[~burst_mask]

    signal_mean = inside_flux.mean() if inside_flux.size > 0 else np.nan
    noise_mean  = outside_flux.mean() if outside_flux.size > 0 else np.nan

    snr_db = 10 * np.log10(signal_mean / noise_mean) if signal_mean and noise_mean else np.nan
    return snr_db, signal_mean, noise_mean


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_snr.py <spectrogram_file_path> <labels_file_path>")
        sys.exit(1)

    # Example: load data
    big_array = np.load(sys.argv[1])
    burst_labels = np.load(sys.argv[2], allow_pickle=True)

    snr_db, signal, noise = compute_snr(big_array, burst_labels)

    print(f"Signal mean: {signal:.3f}")
    print(f"Noise mean: {noise:.3f}")
    print(f"SNR: {snr_db:.2f} dB")

    # Optional visualization
    flux_time = big_array.mean(axis=0)
    plt.figure(figsize=(12, 4))
    plt.plot(flux_time, label="Flux (mean over freqs)", color="blue")

    for entry in burst_labels:
        plt.axvspan(entry["start_idx"], entry["end_idx"], color="red", alpha=0.2)

    plt.xlabel("Time index")
    plt.ylabel("Flux")
    plt.title(f"Flux time series with bursts (SNR={snr_db:.2f} dB)")
    plt.legend()
    plt.show()