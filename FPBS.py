'''
===============================================================================
File: FPBS.py
Author: Zoe Fisch (zoefisch@umich.edu), Ella Ricci (earicci@umich.edu), Maria Herrmann (marherr@umich.edu)
Date: 2026-3-17
Group: University of Michigan SunRISE Mission

Description: Frequency-Persistent Background Suppression is the third stage of
our preprocessing pipeline.
===============================================================================
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

def temporal_median_intensity(S):
    if S.size == 0:
        raise ValueError("Input spectrogram must not be empty")
    elif S.ndim != 2:
        raise ValueError(f"Input spectrogram must be 2D, got {S.ndim}D")
    mu = np.median(S, axis=1)
    return mu

def temporal_variability(S, mu):
    deviation = np.abs(S - mu[:, np.newaxis])
    sigma = 1.4826 * np.median(deviation, axis=1)
    return sigma

# Score P = 1 --> band is persistent
# Score P = 0 --> band is quiet / transient
def persistence_score(S, k=2.0):
    # Background over time from all frequencies
    background_t = np.median(S, axis=0)

    # Robust variability of the time-varying background
    bg_median = np.median(background_t)
    bg_sigma = 1.4826 * np.median(np.abs(background_t - bg_median))

    # Threshold for each time step
    theta_t = background_t + k * bg_sigma

    # Fraction of time each frequency row stays above threshold
    P = np.mean(S > theta_t[None, :], axis=1)
    return P

def compute_persistence_scores(S, k=2.0):
    mu = temporal_median_intensity(S)
    sigma = temporal_variability(S, mu)
    P = persistence_score(S, k=k)
    return P, mu, sigma

# Binary mask: 1 = suppress, 0 = keep
def build_suppression_mask(P, alpha=0.4):
    M = (P > alpha).astype(int)
    return M

def suppress_persistent_channels(S, M):
    """
    Apply frequency-aware suppression using interpolation.

    S: spectrogram (F x T)
    M: mask (F,) where 1 = suppress
    """
    if M.shape[0] != S.shape[0]:
        raise ValueError("Mask must have the same number of frequencies as spectrogram")

    S_clean = S.copy()
    F, T = S.shape

    for f in range(F):
        if M[f] == 1:
            if f == 0:
                S_clean[f, :] = S[f + 1, :]
            elif f == F - 1:
                S_clean[f, :] = S[f - 1, :]
            else:
                S_clean[f, :] = 0.5 * (S[f - 1, :] + S[f + 1, :])

    return S_clean

def reconstruct_spectrogram(S_original, S_clean, M):
    return S_clean

def run_fpbs(S, k=2.0, alpha=0.4):
    """
    Runs the full FPBS pipeline.

    Returns:
        S_final : cleaned spectrogram
        P       : persistence scores
        M       : suppression mask
        mu      : median intensity per frequency
        sigma   : variability per frequency
    """
    P, mu, sigma = compute_persistence_scores(S, k=k)
    M = build_suppression_mask(P, alpha=alpha)
    S_clean = suppress_persistent_channels(S, M)
    S_final = reconstruct_spectrogram(S, S_clean, M)
    return S_final, P, M, mu, sigma

def frequency_variance(S):
    mean_per_freq = np.mean(S, axis=1)
    return np.var(mean_per_freq)

def compute_snr(signal, noise):
    return np.mean(signal) / (np.std(noise) + 1e-8)

def false_suppression_rate(S_before, S_after):
    E_before = np.sum(S_before ** 2)
    E_after = np.sum(S_after ** 2)
    return E_after / (E_before + 1e-8)

def make_demo_spectrogram():
    F, T = 100, 300
    S = 0.15 * np.random.randn(F, T) + 0.5

    persistent_rows = [20, 40, 60]
    for r in persistent_rows:
        S[r, :] += 2.5

    for f in range(F):
        for t in range(T):
            S[f, t] += 2.0 * np.exp(-((f - 75) ** 2) / 20 - ((t - 150) ** 2) / 200)

    return S, persistent_rows

import sys
from plot_npy import plot_spectrogram

if __name__ == "__main__":

    # --- LOAD INPUT ---
    if len(sys.argv) == 2:
        spec_file_path = sys.argv[1]
        print(f"Loading spectrogram: {spec_file_path}")
        S = np.load(spec_file_path)
    else:
        print("Usage: python FPBS.py <spectrogram.npy>")
        sys.exit(1)

    print("Input shape:", S.shape)

    # --- SHOW ORIGINAL ---
    print("Showing ORIGINAL spectrogram...")
    plot_spectrogram(S)

    # --- RUN FPBS ---
    S_clean, P, M, mu, sigma = run_fpbs(S, k=1.0, alpha=0.3)

    print("FPBS completed")
    print("Suppressed channels:", np.where(M == 1)[0])
    print("Total suppressed:", np.sum(M))

    # --- SAVE OUTPUT ---
    np.save("fpbs_cleaned.npy", S_clean)
    print("Saved cleaned spectrogram → fpbs_cleaned.npy")

    # --- SHOW CLEANED ---
    print("Showing CLEANED spectrogram...")
    plot_spectrogram(S_clean)

    # --- SHOW REMOVED ---
    print("Showing REMOVED signal...")
    plot_spectrogram(S - S_clean)