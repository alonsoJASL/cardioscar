# scripts/fix_lge_normalization.py
"""
Normalize LGE intensities in training data from [0, 273] to [0, 1].
Quick fix for already-prepared data.
"""
import argparse
import numpy as np
from pathlib import Path

def main(args) : 
    input_path = args.input 
    output_path = input_path.parent / f'normalised_{input_path.name}' if args.output is None else args.output

    print("Loading data...")
    data = np.load(input_path)

    # Get intensities
    intensities = data['intensities']

    print(f"Original intensity range: [{intensities.min():.1f}, {intensities.max():.1f}]")

    # Normalize to [0, 1]
    intensities_norm = intensities / intensities.max()

    print(f"Normalized intensity range: [{intensities_norm.min():.3f}, {intensities_norm.max():.3f}]")

    # Save corrected version
    np.savez_compressed(
        output_path,
        coordinates=data['coordinates'],
        intensities=intensities_norm,
        group_ids=data['group_ids'],
        group_sizes=data['group_sizes'],
        scaler_min=data['scaler_min'],
        scaler_max=data['scaler_max']
    )

    print(f"\n✓ Saved normalized data to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=False)

    args = parser.parse_args()
    main(args)