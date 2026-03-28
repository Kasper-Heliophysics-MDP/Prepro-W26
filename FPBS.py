"""
Frequency-Persistent Background Suppression for post-AGBS/AMF spectrograms.
"""
import sys
import numpy as np

EPS = 1e-8
FREQ_WINDOW = 7
QUANTILE = 0.35
BAND_WINDOW = 17
BAND_THRESHOLD_SIGMA = 0.2
PERSISTENCE_THRESHOLD_SIGMA = 0.3
BURST_SIGMA_THRESHOLD = 1
PERSISTENCE_SMOOTH_WINDOW = 9
PERSISTENCE_HIT_FRACTION = 0.2
MIN_OCCUPANCY = 0.2
ROW_FLAG_THRESHOLD = 0.5

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

def interpolate_flagged_rows(S, bad_rows):
    S_out = S.copy()
    F = S.shape[0]

    for r in np.where(bad_rows)[0]:
        up = r - 1
        while up >= 0 and bad_rows[up]:
            up -= 1

        down = r + 1
        while down < F and bad_rows[down]:
            down += 1

        if up >= 0 and down < F:
            S_out[r, :] = 0.5 * (S[up, :] + S[down, :])
        elif up >= 0:
            S_out[r, :] = S[up, :]
        elif down < F:
            S_out[r, :] = S[down, :]

    return S_out

def run_fpbs(S):
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
            band_level[row] = np.median(strong)

        hits = (row_valid >= row_med + PERSISTENCE_THRESHOLD_SIGMA * row_mad).astype(float)
        persistence[row] = np.mean(moving_average_1d(hits, PERSISTENCE_SMOOTH_WINDOW) >= PERSISTENCE_HIT_FRACTION)

    baseline = running_median_1d(band_level, BAND_WINDOW)
    radius = BAND_WINDOW // 2
    padded = np.pad(band_level, (radius, radius), mode="edge")
    scale = np.array(
        [1.4826 * np.median(np.abs(padded[i : i + BAND_WINDOW] - baseline[i])) + EPS for i in range(band_level.shape[0])],
        dtype=float,
    )

    contrast = np.clip((band_level - baseline) / scale / 1.5, 0.0, 1.0)
    occupancy = np.clip((persistence - MIN_OCCUPANCY) / max(1.0 - MIN_OCCUPANCY, EPS), 0.0, 1.0)
    weights = np.maximum(contrast, occupancy)

    cleaned = S - (weights * band_level)[:, np.newaxis]

    bad_rows = weights > ROW_FLAG_THRESHOLD
    cleaned = interpolate_flagged_rows(cleaned, bad_rows)

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

    spectrogram = np.load(sys.argv[1])
    plot_results(spectrogram, run_fpbs(spectrogram))