'''
===============================================================================
File: FPBS.py
Author: Zoe Fisch (zoefisch@umich.edu), Ella Ricci (earicci@umich.edu), Maria Herrmann (marherr@umich.edu)
Date: 2026-3-17
Group: University of Michigan SunRISE Mission

Description: Frequency-Persistent Background Suppression is the third stage of our preprocessing pipeline. 
By systematically addressing frequency-dependent artifacts while preserving transient burst signals, 
FPBS will improve the quality of data available to downstream detection and classification algorithms, 
ultimately enhancing our ability to study solar radio emissions. 


===============================================================================
'''
import numpy as np

def temporal_median_intensity(S):
    if S.ndim == 0:
        raise ValueError("Input spectrogram must not be empty")
    elif S.ndim != 2:
        raise ValueError("Input spectrogram must be 2D, given spectrogram is ", S.ndim, " dimensions")
    mu = np.median(S, axis=1)
    return mu       

def temporal_variability(S, mu):
    deviation = np.abs(S - mu[:, np.newaxis])
    sigma = 1.4826 * np.median(deviation, axis=1)
    return sigma

# Score P = 1 --> band is persistent
# Score P = 0 --> band is quiet / transient
def persistence_score(S, mu, sigma, k=2.0):
    theta = mu + k * sigma
    P = np.mean(S > theta[:, None], axis=1)
    return P

def compute_persistence_scores(S, k=2.0):
    mu = temporal_median_intensity(S)
    sigma = temporal_variability(S, mu)
    P = persistence_score(S, mu, sigma, k=k)
    return P, mu, sigma

def classify_channels(P, alpha = 0.4):
    return (P > alpha).astype(int)

# Binary mask: 1 = suppress, 0 = keep
def build_suppression_mask(P, alpha = 0.4):
    M = (P > alpha).astype(int)
    return M

def suppress_persistent_channels(S, M):
    """
    Apply frequency-aware suppression using interpolation.
    
    S: spectrogram (F x T)
    M: mask (F,) where 1 = suppress
    """
    S_clean = S.copy()
    F, T = S.shape

    for f in range(F):
        if M[f] == 1:
            # Handle edges carefully
            if f == 0:
                S_clean[f, :] = S[f + 1, :]
            elif f == F - 1:
                S_clean[f, :] = S[f - 1, :]
            else:
                # Interpolate from neighbors
                S_clean[f, :] = 0.5 * (S[f - 1, :] + S[f + 1, :])

    return S_clean

def reconstruct_spectogram(S_original, S_clean, M):
    # Combine suppressed background with preserved signals.
    # Currectly equivalent to S_clean, but structured for functionality
    # Future: could blend instead of hard replace
    return S_clean

def run_fpbs(S, k=2.0, alpha=0.4):
    """
    Runs the full FPBS pipeline.
    Returns:
        S_final : cleaned spectogram
        P       : persistence scores
        M       : suppression mask
        mu      : median intensity per frequency
        sigma   : variability per frequency
    """
    P, mu, sigma = compute_persistence_scores(S, k=k)
    M = build_suppression_mask(P, alpha=alpha)
    S_clean = suppress_persistent_channels(S, M)
    S_final = reconstruct_spectogram(S, S_clean, M)
    return S_final, P, M, mu, sigma

# evaluation 
