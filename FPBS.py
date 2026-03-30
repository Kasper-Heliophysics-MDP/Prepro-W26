"""
Frequency-Persistent Background Suppression for post-AGBS/AMF spectrograms.
"""
import sys
from pathlib import Path

import numpy as np

EPS = 1e-8
FREQ_WINDOW = 7
QUANTILE = 0.9
BAND_WINDOW = 17
BAND_THRESHOLD_SIGMA = 0.02
PERSISTENCE_THRESHOLD_SIGMA = 0.3
BURST_SIGMA_THRESHOLD = 1
PERSISTENCE_SMOOTH_WINDOW = 21
PERSISTENCE_HIT_FRACTION = 0.2
MIN_OCCUPANCY = 0.2
ROW_FLAG_THRESHOLD = 0.35
SUPPRESSION_GAIN = 2.4

def validate_spectrogram(S):
    if S.size == 0:
        raise ValueError("Input spectrogram must not be empty")
    if S.ndim != 2:
        raise ValueError(f"Input spectrogram must be 2D, got {S.ndim}D")

def running_median_1d(x, window):
    radius = window // 2
    padded = np.pad(x, (radius, radius), mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(x.shape[0])], dtype=float)

def frequency_median_filter(S, window):
    radius = window // 2
    padded = np.pad(S, ((radius, radius), (0, 0)), mode="edge")
    return np.array([np.median(padded[i : i + window], axis=0) for i in range(S.shape[0])], dtype=float)

def moving_average_1d(x, window):
    return np.convolve(x, np.ones(window, dtype=float) / window, mode="same")

def analyze_persistent_bands(S):
    validate_spectrogram(S)
    column_energy = np.median(S, axis=0)
    col_med = np.median(column_energy)
    col_mad = 1.4826 * np.median(np.abs(column_energy - col_med)) + EPS
    valid_cols = column_energy < (col_med + BURST_SIGMA_THRESHOLD * col_mad)
    if np.mean(valid_cols) < 0.2:
        valid_cols[:] = True

    local_background = frequency_median_filter(S, FREQ_WINDOW)
    residual = S - local_background

    band_level = np.zeros(S.shape[0], dtype=float)
    persistence = np.zeros(S.shape[0], dtype=float)

    for row in range(S.shape[0]):
        row_valid = residual[row, valid_cols]
        row_med = np.median(row_valid)
        row_mad = 1.4826 * np.median(np.abs(row_valid - row_med)) + EPS

        strong = row_valid[row_valid >= row_med + BAND_THRESHOLD_SIGMA * row_mad]
        if strong.size:
            band_level[row] = np.quantile(strong, QUANTILE)

        hits = (row_valid >= row_med + PERSISTENCE_THRESHOLD_SIGMA * row_mad).astype(float)
        persistence[row] = np.mean(moving_average_1d(hits, PERSISTENCE_SMOOTH_WINDOW) >= PERSISTENCE_HIT_FRACTION)

    raw_band_level = band_level.copy()
    baseline = running_median_1d(raw_band_level, BAND_WINDOW)
    radius = BAND_WINDOW // 2
    padded = np.pad(raw_band_level, (radius, radius), mode="edge")
    scale = np.array(
        [1.4826 * np.median(np.abs(padded[i : i + BAND_WINDOW] - baseline[i])) + EPS for i in range(raw_band_level.shape[0])],
        dtype=float,
    )

    band_level = running_median_1d(raw_band_level, 5)
    contrast = np.clip((raw_band_level - baseline) / scale, 0.0, 1.0)
    occupancy = np.clip((persistence - MIN_OCCUPANCY) / max(1.0 - MIN_OCCUPANCY, EPS), 0.0, 1.0)
    weights = np.maximum(contrast, occupancy)
    weights = running_median_1d(weights, 5)

    band_rows = running_median_1d((weights > ROW_FLAG_THRESHOLD).astype(float), 5) > 0.4
    weights[band_rows] = np.maximum(weights[band_rows], ROW_FLAG_THRESHOLD)
    weights = np.clip(weights, 0.0, 1.0)

    return band_level, weights, valid_cols

def has_persistent_bands(S):
    band_level, weights, _ = analyze_persistent_bands(S)
    return bool(np.any((weights > ROW_FLAG_THRESHOLD) & (band_level > 0)))

def get_persistent_band_rows(S):
    band_level, weights, _ = analyze_persistent_bands(S)
    return np.where((weights > ROW_FLAG_THRESHOLD) & (band_level > 0))[0]

def run_fpbs(S):
    band_level, weights, valid_cols = analyze_persistent_bands(S)
    cleaned = S.copy()
    cleaned[:, valid_cols] -= (SUPPRESSION_GAIN * weights * band_level)[:, np.newaxis]
    return cleaned

def plot_results(S, S_clean):
    import matplotlib.pyplot as plt

    diff = S - S_clean

    vmin = np.min(S)
    vmax = np.max(S)

    diff_limit = np.max(np.abs(diff))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(
        S,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
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
        interpolation="nearest"
    )
    axes[1].set_title("Cleaned Spectrogram")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Frequency bin")
    fig.colorbar(im1, ax=axes[1], label="Intensity")

    im2 = axes[2].imshow(
        diff,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=-diff_limit,
        vmax=diff_limit,
        interpolation="nearest"
    )
    axes[2].set_title("Intensity Difference (Original - Cleaned)")
    axes[2].set_xlabel("Time index")
    axes[2].set_ylabel("Frequency bin")
    fig.colorbar(im2, ax=axes[2], label="Difference")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python FPBS.py <spectrogram.npy>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    spectrogram = np.load(input_path)
    persistent_rows = get_persistent_band_rows(spectrogram)

    if persistent_rows.size == 0:
        print("No persistent bands detected.")
        sys.exit(0)

    print(f"Persistent bands detected in rows: {persistent_rows.tolist()}")
    cleaned = run_fpbs(spectrogram)

    output_path = input_path.with_name(f"{input_path.stem}-FPBS.npy")
    np.save(output_path, cleaned)
    print(f"Saved cleaned spectrogram to {output_path}")

    plot_results(spectrogram, cleaned)
