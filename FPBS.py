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


def make_demo_spectrogram(seed=0):
    rng = np.random.default_rng(seed)
    F, T = 100, 300
    S = 0.15 * rng.standard_normal((F, T)) + 0.5

    persistent_rows = [20, 40, 60]
    for r in persistent_rows:
        S[r, :] += 9.0

    freq = np.arange(F)[:, np.newaxis]
    time = np.arange(T)[np.newaxis, :]
    burst = 2.0 * np.exp(-((freq - 75) ** 2) / 20.0 - ((time - 150) ** 2) / 200.0)
    S += burst

    return S, persistent_rows


def make_realistic_test_spectrogram(seed=7, shape=(256, 720)):
    """
    Build a more realistic post-AMF/AGBS-like test spectrogram with:
    - low-level structured background
    - a drifting burst backbone with localized bright knots
    - several frequency-persistent horizontal bands
    - a few partial-duration bands that should be harder to classify
    """
    rng = np.random.default_rng(seed)
    F, T = shape

    freq = np.arange(F)[:, np.newaxis]
    time = np.arange(T)[np.newaxis, :]

    base = 0.35 * rng.standard_normal((F, T))
    slow_time = 0.45 * np.sin(2.0 * np.pi * time / 180.0)
    slow_freq = 0.35 * np.cos(2.0 * np.pi * freq / 96.0)
    S = 5.0 + base + slow_time + slow_freq

    for shift, weight in ((1, 0.18), (2, 0.10), (6, 0.08)):
        S[:, shift:] += weight * base[:, :-shift]

    ridge_center = 185.0 - 0.12 * np.arange(T) + 11.0 * np.sin(np.arange(T) / 55.0)
    ridge_width = 8.0 + 1.5 * np.sin(np.arange(T) / 70.0)
    ridge_env = np.exp(-0.5 * ((np.arange(T) - 360.0) / 140.0) ** 2)

    for t_idx in range(T):
        ridge = np.exp(-0.5 * ((np.arange(F) - ridge_center[t_idx]) / ridge_width[t_idx]) ** 2)
        S[:, t_idx] += 10.5 * ridge_env[t_idx] * ridge

    for knot_time in [250, 330, 405, 470]:
        knot_freq = ridge_center[knot_time] + rng.normal(0.0, 5.0)
        knot = np.exp(
            -0.5 * ((freq - knot_freq) / 5.0) ** 2
            -0.5 * ((time - knot_time) / 16.0) ** 2
        )
        S += 9.0 * knot

    branch_center = ridge_center - 22.0
    for t_idx in range(T):
        branch = np.exp(-0.5 * ((np.arange(F) - branch_center[t_idx]) / 5.5) ** 2)
        S[:, t_idx] += 4.5 * ridge_env[t_idx] * branch

    persistent_rows = [42, 97, 154, 211]
    persistent_strengths = [8.5, 7.9, 9.3, 8.1]
    for row, strength in zip(persistent_rows, persistent_strengths):
        S[row, :] += strength + 0.35 * np.sin(np.arange(T) / 23.0 + row / 12.0)

    partial_rows = [68, 176]
    partial_windows = [(110, 390), (280, 620)]
    partial_strengths = [6.2, 5.7]
    for row, (start, end), strength in zip(partial_rows, partial_windows, partial_strengths):
        S[row, start:end] += strength + 0.25 * np.cos(np.arange(end - start) / 18.0)

    metadata = {
        "persistent_rows": persistent_rows,
        "partial_rows": partial_rows,
        "shape": shape,
        "seed": seed,
    }
    return S, metadata


def parse_numeric_token(token):
    match = re.search(r"\d+(?:\.\d+)?", token)
    if match is None:
        return None
    return float(match.group(0))


def hhmm_to_minutes(value):
    value = int(value)
    hours = value // 100
    minutes = value % 100
    return 60 * hours + minutes


def parse_culgoora_catalog_line(line, current_window=None):
    if "CULG" not in line:
        return None, current_window

    parts = line.split()
    if len(parts) < 2:
        return None, current_window

    date_token = parts[0]
    if not re.fullmatch(r"\d{6}", date_token):
        return None, current_window

    idx = 1
    window_start = current_window[0] if current_window else None
    window_end = current_window[1] if current_window else None

    if idx + 1 < len(parts) and re.fullmatch(r"\d{4}", parts[idx]) and re.fullmatch(r"\d{4}", parts[idx + 1]):
        window_start = hhmm_to_minutes(parts[idx])
        window_end = hhmm_to_minutes(parts[idx + 1])
        idx += 2

    if idx >= len(parts) or parts[idx] != "CULG":
        return None, (window_start, window_end)
    idx += 1

    if idx + 1 >= len(parts):
        return None, (window_start, window_end)

    start_value = parse_numeric_token(parts[idx])
    end_value = parse_numeric_token(parts[idx + 1])
    if start_value is None or end_value is None:
        return None, (window_start, window_end)
    idx += 2

    if idx >= len(parts):
        return None, (window_start, window_end)
    burst_type = parts[idx]
    idx += 1

    leading_numbers = []
    for token in parts[idx:]:
        value = parse_numeric_token(token)
        if value is not None:
            leading_numbers.append(value)
            if len(leading_numbers) == 3:
                break

    if len(leading_numbers) < 3:
        return None, (window_start, window_end)

    f_low = leading_numbers[1]
    f_high = leading_numbers[2]

    event = {
        "date": date_token,
        "window_start_min": window_start,
        "window_end_min": window_end,
        "start_min": hhmm_to_minutes(start_value),
        "end_min": hhmm_to_minutes(end_value),
        "burst_type": burst_type,
        "f_low_mhz": min(f_low, f_high),
        "f_high_mhz": max(f_low, f_high),
        "raw_line": line.rstrip(),
    }
    return event, (window_start, window_end)


def parse_culgoora_catalog_text(text):
    events = []
    current_window = None

    for line in text.splitlines():
        event, current_window = parse_culgoora_catalog_line(line, current_window=current_window)
        if event is not None:
            events.append(event)

    return events


def choose_catalog_event(events):
    scored = []
    for event in events:
        duration = max(1, event["end_min"] - event["start_min"])
        bandwidth = max(1.0, event["f_high_mhz"] - event["f_low_mhz"])
        type_bonus = {
            "III": 3.0,
            "II": 2.0,
            "IV": 1.5,
            "V": 1.0,
            "CONT": 0.5,
        }.get(event["burst_type"], 0.25)
        score = type_bonus * bandwidth * np.sqrt(duration)
        scored.append((score, event))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else None


def make_synthetic_spectrogram_from_catalog_event(event, seed=13, shape=(256, 720)):
    rng = np.random.default_rng(seed)
    F, T = shape

    freq = np.arange(F)[:, np.newaxis]
    time = np.arange(T)[np.newaxis, :]
    S = 4.8 + 0.3 * rng.standard_normal((F, T))
    S += 0.4 * np.sin(2.0 * np.pi * time / 210.0)
    S += 0.25 * np.cos(2.0 * np.pi * freq / 88.0)

    if event["window_start_min"] is None or event["window_end_min"] is None:
        window_start = max(0, event["start_min"] - 20)
        window_end = event["end_min"] + 20
    else:
        window_start = event["window_start_min"]
        window_end = event["window_end_min"]

    if window_end <= window_start:
        window_end = window_start + 240

    span_minutes = max(1, window_end - window_start)
    start_idx = int(np.clip((event["start_min"] - window_start) / span_minutes * (T - 1), 0, T - 1))
    end_idx = int(np.clip((event["end_min"] - window_start) / span_minutes * (T - 1), start_idx, T - 1))

    f_min = event["f_low_mhz"]
    f_max = event["f_high_mhz"]
    freq_span = max(1.0, f_max - f_min)

    top_freq = max(650.0, f_max + 40.0)
    bottom_freq = max(15.0, min(18.0, f_min) - 5.0)
    if top_freq <= bottom_freq:
        top_freq = bottom_freq + 100.0

    def mhz_to_bin(mhz_value):
        ratio = (mhz_value - bottom_freq) / (top_freq - bottom_freq)
        return float(np.clip(ratio * (F - 1), 0, F - 1))

    high_bin = mhz_to_bin(f_max)
    low_bin = mhz_to_bin(f_min)
    duration_bins = max(1, end_idx - start_idx + 1)
    type_name = event["burst_type"]

    if type_name == "III":
        for offset, t_idx in enumerate(range(start_idx, end_idx + 1)):
            phase = offset / max(1, duration_bins - 1)
            center = high_bin + (low_bin - high_bin) * phase
            width = 4.0 + 1.5 * np.sin(offset / 9.0)
            ridge = np.exp(-0.5 * ((np.arange(F) - center) / width) ** 2)
            amplitude = 10.0 + 2.0 * np.sin(offset / 7.0)
            S[:, t_idx] += amplitude * ridge

        for knot_time in np.linspace(start_idx, end_idx, 4, dtype=int):
            knot_center = high_bin + (low_bin - high_bin) * ((knot_time - start_idx) / max(1, end_idx - start_idx))
            knot = np.exp(
                -0.5 * ((freq - knot_center) / 5.0) ** 2
                -0.5 * ((time - knot_time) / 8.0) ** 2
            )
            S += 6.5 * knot

    elif type_name == "II":
        for offset, t_idx in enumerate(range(start_idx, end_idx + 1)):
            phase = offset / max(1, duration_bins - 1)
            center = high_bin + (low_bin - high_bin) * (0.25 * phase)
            ridge = np.exp(-0.5 * ((np.arange(F) - center) / 3.8) ** 2)
            harmonic = np.exp(-0.5 * ((np.arange(F) - (center + 18.0)) / 4.6) ** 2)
            S[:, t_idx] += 7.0 * ridge + 4.5 * harmonic

    elif type_name == "IV" or type_name == "CONT":
        center = 0.5 * (high_bin + low_bin)
        band = np.exp(-0.5 * ((freq - center) / max(8.0, 0.18 * freq_span)) ** 2)
        envelope = np.exp(-0.5 * ((time - 0.5 * (start_idx + end_idx)) / max(10.0, 0.4 * duration_bins)) ** 2)
        S += 8.0 * band * envelope

    else:
        center = 0.5 * (high_bin + low_bin)
        event_blob = np.exp(
            -0.5 * ((freq - center) / max(5.0, 0.12 * freq_span)) ** 2
            -0.5 * ((time - 0.5 * (start_idx + end_idx)) / max(6.0, 0.35 * duration_bins)) ** 2
        )
        S += 7.5 * event_blob

    persistent_rows = [34, 92, 173, 224]
    persistent_strengths = [8.1, 7.3, 8.8, 7.7]
    for row, strength in zip(persistent_rows, persistent_strengths):
        S[row, :] += strength + 0.3 * np.sin(np.arange(T) / 19.0 + row / 14.0)

    partial_rows = [63, 201]
    partial_windows = [(90, 310), (360, 610)]
    for row, (left, right) in zip(partial_rows, partial_windows):
        S[row, left:right] += 5.9 + 0.25 * np.cos(np.arange(right - left) / 20.0)

    metadata = {
        "selected_event": event,
        "persistent_rows": persistent_rows,
        "partial_rows": partial_rows,
        "window_minutes": (window_start, window_end),
        "shape": shape,
        "seed": seed,
    }
    return S, metadata


def make_synthetic_spectrogram_from_catalog_text(text, seed=13, shape=(256, 720)):
    events = parse_culgoora_catalog_text(text)
    chosen_event = choose_catalog_event(events)
    if chosen_event is None:
        raise ValueError("No usable Culgoora event lines were found in the catalog text")
    return make_synthetic_spectrogram_from_catalog_event(chosen_event, seed=seed, shape=shape)


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
