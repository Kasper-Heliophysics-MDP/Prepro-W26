"""
===============================================================================
File: FPBS.py
Author: Zoe Fisch (zoefisch@umich.edu), Ella Ricci (earicci@umich.edu),
        Maria Herrmann (marherr@umich.edu)
Date: 2026-03-17
Group: University of Michigan SunRISE Mission

Description: Frequency-Persistent Background Suppression is the third stage of
our preprocessing pipeline.
===============================================================================
"""

import re
import sys

import numpy as np


EPS = 1e-8


def validate_spectrogram(S):
    if S.size == 0:
        raise ValueError("Input spectrogram must not be empty")
    if S.ndim != 2:
        raise ValueError(f"Input spectrogram must be 2D, got {S.ndim}D")


def temporal_median_intensity(S):
    validate_spectrogram(S)
    return np.median(S, axis=1)


def temporal_variability(S, mu):
    deviation = np.abs(S - mu[:, np.newaxis])
    return 1.4826 * np.median(deviation, axis=1)


def running_median_1d(x, window):
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    radius = window // 2
    padded = np.pad(x, (radius, radius), mode="edge")
    out = np.empty_like(x, dtype=float)

    for idx in range(x.shape[0]):
        out[idx] = np.median(padded[idx : idx + window])

    return out


def frequency_median_filter(S, window=9):
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    radius = window // 2
    padded = np.pad(S, ((radius, radius), (0, 0)), mode="edge")
    filtered = np.empty_like(S, dtype=float)

    for f in range(S.shape[0]):
        filtered[f] = np.median(padded[f : f + window], axis=0)

    return filtered


def persistent_excess_profile(S, freq_window=9, quantile=0.25, db_threshold=7.0):
    """
    Estimate the additive horizontal-band offset for each frequency row.

    The local median over nearby frequencies captures the smooth background.
    The lower quantile of the residual keeps only the persistent positive floor
    and ignores short-lived burst energy.
    """
    local_background = frequency_median_filter(S, window=freq_window)
    residual = S - local_background
    band_level = np.empty(S.shape[0], dtype=float)

    for idx in range(S.shape[0]):
        strong_samples = residual[idx, residual[idx] >= db_threshold]
        if strong_samples.size == 0:
            band_level[idx] = 0.0
        else:
            band_level[idx] = np.quantile(strong_samples, quantile)

    return residual, local_background, band_level


def persistence_score(residual, db_threshold=7.0):
    """
    Score in [0, 1]. High when a row stays meaningfully above the local
    background across most of the observation.
    """
    return np.mean(residual >= db_threshold, axis=1)


def local_band_statistics(band_level, window=11):
    baseline = running_median_1d(band_level, window=window)

    radius = window // 2
    padded = np.pad(band_level, (radius, radius), mode="edge")
    scale = np.empty_like(band_level, dtype=float)

    for idx in range(band_level.shape[0]):
        neighborhood = padded[idx : idx + window]
        scale[idx] = 1.4826 * np.median(np.abs(neighborhood - baseline[idx])) + EPS

    return baseline, scale


def compute_band_weights(
    band_level,
    P,
    band_window=11,
    z0=1.0,
    z1=3.0,
    min_occupancy=0.4,
):
    """
    Soft confidence for each row. This avoids the all-or-nothing behavior that
    was causing over-removal and under-removal.
    """
    baseline, scale = local_band_statistics(band_level, window=band_window)
    contrast_z = (band_level - baseline) / scale

    persistence_term = np.clip((P - min_occupancy) / max(1.0 - min_occupancy, EPS), 0.0, 1.0)
    contrast_term = np.clip((contrast_z - z0) / max(z1 - z0, EPS), 0.0, 1.0)
    weights = persistence_term * contrast_term

    return weights, contrast_z, baseline, scale


def build_suppression_mask(weights, threshold=0.35):
    return (weights >= threshold).astype(int)


def suppress_persistent_channels(S, band_level, weights):
    """
    Subtract only the estimated persistent offset instead of replacing the
    entire row with neighbors. This preserves transient burst structure.
    """
    subtraction = (weights * band_level)[:, np.newaxis]
    return S - subtraction


def reconstruct_spectrogram(S_original, S_soft, local_background, weights):
    """
    Clamp rows only if soft subtraction drives them well below the local
    spectral background. This keeps the output from carving dark trenches.
    """
    floor = local_background - 0.75 * np.std(S_original - local_background, axis=1, keepdims=True)
    relaxed_floor = np.where(weights[:, np.newaxis] > 0.0, floor, -np.inf)
    return np.maximum(S_soft, relaxed_floor)


def compute_persistence_scores(S, freq_window=9, quantile=0.25, db_threshold=7.0):
    mu = temporal_median_intensity(S)
    sigma = temporal_variability(S, mu)
    residual, local_background, band_level = persistent_excess_profile(
        S,
        freq_window=freq_window,
        quantile=quantile,
        db_threshold=db_threshold,
    )
    P = persistence_score(residual, db_threshold=db_threshold)
    return P, mu, sigma, residual, local_background, band_level


def run_fpbs(
    S,
    freq_window=9,
    quantile=0.25,
    band_window=11,
    db_threshold=7.0,
    min_occupancy=0.4,
    mask_threshold=0.35,
):
    """
    Runs the full FPBS pipeline.

    Returns:
        S_final      : cleaned spectrogram
        P            : persistence scores
        M            : suppression mask
        mu           : median intensity per frequency
        sigma        : variability per frequency
        band_level   : estimated persistent band amplitude per frequency
        weights      : soft suppression weights
        contrast_z   : local contrast score of each row
    """
    validate_spectrogram(S)

    P, mu, sigma, residual, local_background, band_level = compute_persistence_scores(
        S,
        freq_window=freq_window,
        quantile=quantile,
        db_threshold=db_threshold,
    )
    weights, contrast_z, _, _ = compute_band_weights(
        band_level,
        P,
        band_window=band_window,
        min_occupancy=min_occupancy,
    )
    M = build_suppression_mask(weights, threshold=mask_threshold)
    S_soft = suppress_persistent_channels(S, band_level, weights)
    S_final = reconstruct_spectrogram(S, S_soft, local_background, weights)

    diagnostics = {
        "residual": residual,
        "local_background": local_background,
        "band_level": band_level,
        "weights": weights,
        "contrast_z": contrast_z,
    }

    return S_final, P, M, mu, sigma, diagnostics


def frequency_variance(S):
    mean_per_freq = np.mean(S, axis=1)
    return np.var(mean_per_freq)


def compute_snr(signal, noise):
    return np.mean(signal) / (np.std(noise) + EPS)


def false_suppression_rate(S_before, S_after):
    E_before = np.sum(S_before**2)
    E_after = np.sum(S_after**2)
    return E_after / (E_before + EPS)





def plot_results(S, S_clean):
    import matplotlib.pyplot as plt

    removed = S - S_clean
    vmin = np.min(S)
    vmax = np.max(S)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(
        S,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Original Spectrogram")
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel("Frequency bin")
    fig.colorbar(im0, ax=axes[0], label="Intensity")

    im1 = axes[1].imshow(
        S_clean,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title("Cleaned Spectrogram")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Frequency bin")
    fig.colorbar(im1, ax=axes[1], label="Intensity")

    im2 = axes[2].imshow(
        removed,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[2].set_title("Removed Signal")
    axes[2].set_xlabel("Time index")
    axes[2].set_ylabel("Frequency bin")
    fig.colorbar(im2, ax=axes[2], label="Intensity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--catalog-to-test":
        catalog_path = sys.argv[2]
        output_path = sys.argv[3]
        with open(catalog_path, "r", encoding="utf-8") as handle:
            catalog_text = handle.read()
        S, metadata = make_synthetic_spectrogram_from_catalog_text(catalog_text)
        np.save(output_path, S)
        selected = metadata["selected_event"]
        print(f"Saved catalog-based synthetic spectrogram -> {output_path}")
        print("Selected event:", selected["raw_line"])
        print("Persistent rows:", metadata["persistent_rows"])
        print("Partial rows:", metadata["partial_rows"])
        sys.exit(0)

    if len(sys.argv) == 3 and sys.argv[1] == "--make-test":
        output_path = sys.argv[2]
        S, metadata = make_realistic_test_spectrogram()
        np.save(output_path, S)
        print(f"Saved synthetic test spectrogram -> {output_path}")
        print("Persistent rows:", metadata["persistent_rows"])
        print("Partial rows:", metadata["partial_rows"])
        sys.exit(0)

    if len(sys.argv) != 2:
        print("Usage: python FPBS.py <spectrogram.npy>")
        print("   or: python FPBS.py --make-test <output.npy>")
        print("   or: python FPBS.py --catalog-to-test <catalog.txt> <output.npy>")
        sys.exit(1)

    spec_file_path = sys.argv[1]
    print(f"Loading spectrogram: {spec_file_path}")
    S = np.load(spec_file_path)
    print("Input shape:", S.shape)

    S_clean, P, M, mu, sigma, diagnostics = run_fpbs(
        S,
        freq_window=9,
        quantile=0.25,
        band_window=11,
        db_threshold=7.0,
        min_occupancy=0.4,
        mask_threshold=0.35,
    )

    print("FPBS completed")
    print("Suppressed channels:", np.where(M == 1)[0])
    print("Total suppressed:", np.sum(M))
    print("Top weighted channels:", np.argsort(diagnostics["weights"])[-10:][::-1])

    np.save("fpbs_cleaned.npy", S_clean)
    print("Saved cleaned spectrogram -> fpbs_cleaned.npy")

    plot_results(S, S_clean)
