import argparse
import pyvista as pv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

def main(args):

    directory = args.input_dir

    # Load all models
    mesh_adam = pv.read( directory / 'legacy_tf_scar_adam.vtk')
    mesh_adamax = pv.read( directory / 'legacy_tf_scar_adamax.vtk')
    mesh_cardio = pv.read( directory / 'cardioscar_model_with_scar_legacy.vtk')
    mesh_cardio_mc5 = pv.read( directory / 'cardioscar_model_with_scar_legacy_mc5.vtk')

    # Extract predictions
    pred_adam = mesh_adam['scar_probability_tf']
    pred_adamax = mesh_adamax['scar_probability_tf']
    pred_cardio = mesh_cardio['scar_probability']
    pred_cardio_mc5 = mesh_cardio_mc5['scar_probability']

    print("=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)

    for name, pred in [
        ("Legacy Adam", pred_adam),
        ("Legacy Adamax", pred_adamax),
        ("CardioScar Default", pred_cardio),
        ("CardioScar MC5", pred_cardio_mc5)
    ]:
        print(f"\n{name}:")
        print(f"  Mean: {pred.mean():.4f}")
        print(f"  Std:  {pred.std():.4f}")
        print(f"  Min:  {pred.min():.4f}")
        print(f"  Max:  {pred.max():.4f}")
        print(f"  Median: {np.median(pred):.4f}")
        print(f"  >0.5 threshold: {(pred > 0.5).sum()} nodes ({100*(pred > 0.5).sum()/len(pred):.1f}%)")

    print("\n" + "=" * 60)
    print("CORRELATIONS (vs Legacy Adamax)")
    print("=" * 60)

    corr_cardio, _ = pearsonr(pred_adamax, pred_cardio)
    corr_mc5, _ = pearsonr(pred_adamax, pred_cardio_mc5)

    print(f"CardioScar Default:  r = {corr_cardio:.4f}")
    print(f"CardioScar MC5:      r = {corr_mc5:.4f}")

    # MAE
    mae_cardio = np.abs(pred_adamax - pred_cardio).mean()
    mae_mc5 = np.abs(pred_adamax - pred_cardio_mc5).mean()

    print("\n" + "=" * 60)
    print("MEAN ABSOLUTE ERROR (vs Legacy Adamax)")
    print("=" * 60)
    print(f"CardioScar Default:  MAE = {mae_cardio:.4f}")
    print(f"CardioScar MC5:      MAE = {mae_mc5:.4f}")

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir', 
        type=Path,
    )
    args = parser.parse_args()
    main(args)