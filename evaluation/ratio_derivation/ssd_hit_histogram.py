#!/usr/bin/env python3
"""ssd_hit_histogram.py

Histogram of total hit counts per voxel in an SSD postprocessed simulation file.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_SSD_FILE = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "MLFormatMusunNCsZylSSD300PMTs/ncscore_output_0.hdf5"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Histogram of SSD voxel hit counts.")
    p.add_argument('--ssd-file', type=Path, default=Path(_DEFAULT_SSD_FILE),
                   help="SSD postprocessed HDF5 file (default: %(default)s)")
    p.add_argument('--output-dir', type=Path, default=Path.cwd() / "comparison_plots",
                   help="Output directory for plots (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.ssd_file, "r") as f:
        voxel_ids = [c.decode() if isinstance(c, bytes) else str(c)
                     for c in f["target_columns"][:]]
        mat = f["target_matrix"]
        num_ncs, num_voxels = mat.shape
        total_hits = np.zeros(num_voxels, dtype=np.int64)
        _BATCH = 1000
        for _rs in range(0, num_ncs, _BATCH):
            total_hits += mat[_rs:min(_rs + _BATCH, num_ncs), :].astype(np.int64).sum(axis=0)

    print(f"Total voxels: {len(total_hits)}")
    print(f"Voxels with 0 hits: {np.sum(total_hits == 0)}")
    print(f"Min/Max hits: {total_hits.min()} / {total_hits.max()}")
    print(f"Mean ± std: {total_hits.mean():.1f} ± {total_hits.std():.1f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    nonzero = total_hits[total_hits > 0]
    if len(nonzero) > 0:
        bin_edges = np.concatenate([[-0.5, 0.5],
                                    np.linspace(0.5, nonzero.max() + 0.5, 80)])
    else:
        bin_edges = np.array([-0.5, 0.5, 1.5])

    ax.hist(total_hits, bins=bin_edges, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0.25, color="red", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Total hits per voxel (summed over all NCs)")
    ax.set_ylabel("Number of voxels")
    ax.set_title(f"SSD Postprocessed: Hit Distribution ({len(total_hits)} voxels)")
    ax.annotate(f"0-hits: {np.sum(total_hits == 0)}",
                xy=(0, np.sum(total_hits == 0)),
                fontsize=9, ha="center", va="bottom")

    plt.tight_layout()
    fpath = args.output_dir / "ssd_voxel_hit_histogram.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
