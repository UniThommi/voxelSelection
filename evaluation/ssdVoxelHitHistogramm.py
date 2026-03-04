#!/usr/bin/env python3
"""Histogram of total hit counts per voxel in SSD postprocessed simulation."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SSD_FILE = (
    "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
    "MLFormatMusunNCsZylSSD300PMTs/ncscore_output_0.hdf5"
)
OUTPUT_DIR = Path.cwd() / "comparison_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

with h5py.File(SSD_FILE, "r") as f:
    voxel_ids = sorted(f["target"].keys())
    total_hits = np.array([np.sum(f["target"][v][:]) for v in voxel_ids])

print(f"Total voxels: {len(total_hits)}")
print(f"Voxels with 0 hits: {np.sum(total_hits == 0)}")
print(f"Min/Max hits: {total_hits.min()} / {total_hits.max()}")
print(f"Mean ± std: {total_hits.mean():.1f} ± {total_hits.std():.1f}")

fig, ax = plt.subplots(figsize=(10, 6))

# Separate bin for 0, then regular bins for the rest
nonzero = total_hits[total_hits > 0]
if len(nonzero) > 0:
    bin_edges = np.concatenate([[-0.5, 0.5], np.linspace(0.5, nonzero.max() + 0.5, 80)])
else:
    bin_edges = np.array([-0.5, 0.5, 1.5])

ax.hist(total_hits, bins=bin_edges, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0.25, color="red", linestyle="--", linewidth=0.5)  # visual separator
ax.set_xlabel("Total hits per voxel (summed over all NCs)")
ax.set_ylabel("Number of voxels")
ax.set_title(f"SSD Postprocessed: Hit Distribution ({len(total_hits)} voxels)")
ax.annotate(f"0-hits: {np.sum(total_hits == 0)}", xy=(0, np.sum(total_hits == 0)),
            fontsize=9, ha="center", va="bottom")

plt.tight_layout()
fpath = OUTPUT_DIR / "ssd_voxel_hit_histogram.png"
fig.savefig(fpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fpath}")