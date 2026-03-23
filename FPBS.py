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
    if M.shape[0] != S.shape[0]:
        raise ValueError("Mask must have the same number of frequencies as spectrogram")
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

def reconstruct_spectrogram(S_original, S_clean, M):
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


if __name__ == "__main__":
    # Example usage
    S = np.random.rand(100, 300) # fake spectrogram
    S_clean, P, M, mu, sigma = run_fpbs(S)
    print("FPBS completed")
    print("Persistence scores shape:", P.shape)
    print("Mask sum (suppressed channels):", np.sum(M))

