"""
Evaluate how well FPBS meets the project requirements on one spectrogram.
"""

import sys
from pathlib import Path

import numpy as np

from FPBS import ROW_FLAG_THRESHOLD, analyze_persistent_bands, run_fpbs

EPS = 1e-8
BURST_COL_PERCENTILE = 95
BURST_PIXEL_PERCENTILE = 90


def infer_burst_columns(S):
    column_energy = np.median(S, axis=0)
    threshold = np.percentile(column_energy, BURST_COL_PERCENTILE)
    burst_cols = column_energy >= threshold

    if not np.any(burst_cols):
        burst_cols[np.argmax(column_energy)] = True

    return burst_cols


def infer_persistent_rows(S):
    band_level, weights, valid_cols = analyze_persistent_bands(S)
    persistent_rows = np.where((weights > ROW_FLAG_THRESHOLD) & (band_level > 0))[0]
    return persistent_rows, valid_cols


def build_burst_mask(S, burst_cols, persistent_rows):
    burst_mask = np.zeros_like(S, dtype=bool)
    row_mask = np.ones(S.shape[0], dtype=bool)
    row_mask[persistent_rows] = False

    burst_region = S[:, burst_cols]
    if burst_region.size == 0:
        return burst_mask

    intensity_threshold = np.percentile(burst_region, BURST_PIXEL_PERCENTILE)
    burst_mask[:, burst_cols] = S[:, burst_cols] >= intensity_threshold
    burst_mask &= row_mask[:, np.newaxis]
    return burst_mask


def compute_burst_preservation(S_before, S_after, burst_mask):
    if not np.any(burst_mask):
        return np.nan, np.nan, np.nan

    before_mean = np.mean(S_before[burst_mask])
    after_mean = np.mean(S_after[burst_mask])
    percent_change = 100.0 * (after_mean - before_mean) / (before_mean + EPS)
    return before_mean, after_mean, percent_change


def compute_snr_metrics(S_before, S_after, burst_mask, persistent_rows, valid_cols):
    noise_row_mask = np.ones(S_before.shape[0], dtype=bool)
    noise_row_mask[persistent_rows] = False
    noise_mask = noise_row_mask[:, np.newaxis] & valid_cols[np.newaxis, :] & (~burst_mask)

    if not np.any(burst_mask) or not np.any(noise_mask):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    signal_before = np.mean(S_before[burst_mask])
    signal_after = np.mean(S_after[burst_mask])
    noise_before = np.mean(S_before[noise_mask])
    noise_after = np.mean(S_after[noise_mask])

    snr_before = signal_before / (noise_before + EPS)
    snr_after = signal_after / (noise_after + EPS)
    snr_improvement = 100.0 * (snr_after - snr_before) / (snr_before + EPS)

    return signal_before, signal_after, snr_before, snr_after, snr_improvement, noise_after


def compute_band_reduction(S_before, S_after, persistent_rows, valid_cols):
    if persistent_rows.size == 0 or not np.any(valid_cols):
        return np.nan, np.nan, np.nan

    band_mask = np.zeros_like(S_before, dtype=bool)
    band_mask[persistent_rows, :] = True
    band_mask &= valid_cols[np.newaxis, :]

    before_mean = np.mean(S_before[band_mask])
    after_mean = np.mean(S_after[band_mask])
    reduction = 100.0 * (before_mean - after_mean) / (before_mean + EPS)
    return before_mean, after_mean, reduction


def requirement_status(condition):
    return "PASS" if condition else "FAIL"


def evaluate_spectrogram(S):
    S_clean = run_fpbs(S)
    persistent_rows, valid_cols = infer_persistent_rows(S)
    burst_cols = infer_burst_columns(S)
    burst_mask = build_burst_mask(S, burst_cols, persistent_rows)

    burst_before, burst_after, burst_change = compute_burst_preservation(S, S_clean, burst_mask)
    signal_before, signal_after, snr_before, snr_after, snr_improvement, noise_after = compute_snr_metrics(
        S,
        S_clean,
        burst_mask,
        persistent_rows,
        valid_cols,
    )
    band_before, band_after, band_reduction = compute_band_reduction(S, S_clean, persistent_rows, valid_cols)

    return {
        "persistent_rows": persistent_rows,
        "burst_before": burst_before,
        "burst_after": burst_after,
        "burst_change": burst_change,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement": snr_improvement,
        "band_before": band_before,
        "band_after": band_after,
        "band_reduction": band_reduction,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_spectrogram.py <spectrogram.npy>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    spectrogram = np.load(input_path)
    results = evaluate_spectrogram(spectrogram)

    print(f"Evaluating: {input_path}")

    if np.isnan(results["burst_change"]):
        print("Burst preservation: N/A (could not infer burst region)")
    else:
        burst_ok = abs(results["burst_change"]) <= 5.0
        print(
            f"Burst preservation: {results['burst_change']:+.2f}% "
            f"({requirement_status(burst_ok)}; requirement: within +/-5%)"
        )

    if np.isnan(results["snr_improvement"]):
        print("SNR improvement: N/A (could not infer burst/noise regions)")
    else:
        snr_ok = results["snr_improvement"] >= 15.0
        print(
            f"SNR before/after: {results['snr_before']:.3f} -> {results['snr_after']:.3f}"
        )
        print(
            f"SNR improvement: {results['snr_improvement']:.2f}% "
            f"({requirement_status(snr_ok)}; requirement: at least 15%)"
        )

    if np.isnan(results["band_reduction"]):
        print("Persistent band reduction: N/A (no persistent rows detected)")
    else:
        band_ok = results["band_reduction"] >= 60.0
        print(
            f"Persistent band intensity: {results['band_before']:.3f} -> {results['band_after']:.3f}"
        )
        print(
            f"Persistent band reduction: {results['band_reduction']:.2f}% "
            f"({requirement_status(band_ok)}; requirement: at least 60%)"
        )
