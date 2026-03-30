# File Descriptions

## `FPBS.py`

- Purpose: Load an AGBS/AMF-processed `.npy` spectrogram, detect persistent
  horizontal frequency bands, suppress them, save the cleaned spectrogram, and
  display the original and cleaned spectrograms.

- Key Functions:
  - `validate_spectrogram(S: np.ndarray)`: Confirms that the input is a valid
    2D spectrogram array.
  - `running_median_1d(x, window)`: Applies a 1D running median used to smooth
    row-based band estimates.
  - `frequency_median_filter(S, window)`: Builds a local frequency background
    estimate for the spectrogram.
  - `moving_average_1d(x, window)`: Smooths persistence hits over time so
    broken but persistent bands are still detected.
  - `analyze_persistent_bands(S)`: Detects persistent horizontal bands and
    returns their estimated band levels, weights, and valid non-burst columns.
  - `has_persistent_bands(S)`: Checks whether the spectrogram contains
    persistent bands worth suppressing.
  - `get_persistent_band_rows(S)`: Returns the row indices currently identified
    as persistent bands.
  - `run_fpbs(S)`: Runs the full FPBS step and returns the cleaned spectrogram.
  - `plot_results(S, S_clean)`: Displays the original, cleaned, and difference
    spectrograms.

- Usage:

```bash
python3 FPBS.py <spectrogram.npy>
```

- Example:

```bash
python3 FPBS.py sample_spectrogram.npy
```

This saves a file called `sample_spectrogram-FPBS.npy`.

## `evaluate_spectrogram.py`

- Purpose: Load a `.npy` spectrogram, run FPBS on it, and print how well the
  result meets the project requirements for burst preservation, SNR
  improvement, and persistent band reduction.

- Key Functions:
  - `infer_burst_columns(S)`: Estimates which time columns belong to the burst.
  - `infer_persistent_rows(S)`: Uses FPBS band analysis to estimate which rows
    correspond to persistent horizontal bands.
  - `build_burst_mask(S, burst_cols, persistent_rows)`: Builds a burst-region
    mask while excluding persistent-band rows.
  - `compute_burst_preservation(S_before, S_after, burst_mask)`: Measures how
    much the burst intensity changes after FPBS.
  - `compute_snr_metrics(S_before, S_after, burst_mask, persistent_rows, valid_cols)`:
    Computes SNR before and after FPBS and reports percent improvement.
  - `compute_band_reduction(S_before, S_after, persistent_rows, valid_cols)`:
    Measures how much persistent band intensity was reduced.
  - `requirement_status(condition)`: Converts a metric check into `PASS` or
    `FAIL`.
  - `evaluate_spectrogram(S)`: Runs the full evaluation and returns the
    collected metrics.

- Usage:

```bash
python3 evaluate_spectrogram.py <spectrogram.npy>
```

- Example:

```bash
python3 evaluate_spectrogram.py sample_spectrogram.npy
```

This prints burst preservation, SNR improvement, and persistent band reduction
results to the terminal.
