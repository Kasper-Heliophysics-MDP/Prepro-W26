"""
Run FPBS on every .npy spectrogram in a folder and zip the cleaned outputs.
"""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from FPBS import has_persistent_bands, run_fpbs


def clean_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        raise ValueError(f"No .npy spectrograms found in {input_dir}")

    saved_files = []

    for input_file in input_files:
        spectrogram = np.load(input_file)
        cleaned = run_fpbs(spectrogram) if has_persistent_bands(spectrogram) else spectrogram

        output_file = output_dir / f"{input_file.stem}-FPBS.npy"
        np.save(output_file, cleaned)
        saved_files.append(output_file)

    return saved_files


def zip_outputs(output_dir, zip_path):
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for output_file in sorted(output_dir.glob("*.npy")):
            zip_file.write(output_file, arcname=output_file.name)

    return zip_path


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print("Usage: python batch_fpbs.py <folder_of_npy_files> [output_zip_path]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}")
        sys.exit(1)

    output_dir = input_dir.parent / f"{input_dir.name}-FPBS-cleaned"
    saved_files = clean_folder(input_dir, output_dir)

    if len(sys.argv) == 3:
        zip_path = Path(sys.argv[2])
    else:
        zip_path = input_dir.parent / f"{input_dir.name}-FPBS-cleaned.zip"

    zip_outputs(output_dir, zip_path)

    print(f"Processed {len(saved_files)} spectrogram(s)")
    print(f"Cleaned files saved in: {output_dir}")
    print(f"Zip archive saved to: {zip_path}")
