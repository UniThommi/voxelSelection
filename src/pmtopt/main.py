#!/usr/bin/env python3
"""
PMT Position Optimization for Neutron Capture Detection
========================================================

Unified entry point for:
  1. Greedy voxel selection with stochastic ratio scaling (default)
  2. HDF5 file writing with ratio-adjusted hits (--write-hdf5)
  3. Sensitivity analysis for ratio perturbation (--sensitivity)

All modes use stochastic rounding: floor(hits/ratio) + Bernoulli(frac).

Usage examples:
    # Standard greedy (NC mode, M=1)
    python main.py data.hdf5 -N 300 --optimize nc -M 1 -m 1

    # With custom ratios
    python main.py data.hdf5 -N 300 --optimize nc -M 1 -m 1 \\
        --pit 2.07 --bot 2.38 --top 2.20 --wall 1.88

    # Write ratio-adjusted HDF5 + run greedy + sensitivity
    python main.py data.hdf5 -N 300 --optimize nc -M 1 -m 1 \\
        --write-hdf5 --sensitivity --output-dir results/

    # Muon-Ge77 mode with muon weighting
    python main.py data.hdf5 -N 300 --optimize muon-ge77 -W 1 -m 1 \\
        --muon-weight 4.34

    # No ratio scaling (equivalent to ratio=1 for all areas)
    python main.py data.hdf5 -N 300 --optimize nc -M 1 -m 1 \\
        --pit 1 --bot 1 --top 1 --wall 1

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from pmtopt.geometry import (
    PMT_RADIUS, DEFAULT_AREA_RATIOS, compute_per_area_N,
)
from pmtopt.data_loading import (
    load_and_binarize, load_muon_data, build_muon_index, load_nc_muon_ids,
)
from pmtopt.greedy import greedy_select_nc, greedy_select_muon
from pmtopt.plotting import (
    plot_selected_voxels, plot_muon_nc_histogram,
)
from pmtopt.ratio_scaling import write_ratio_hdf5, fmt_ratio_filename
from pmtopt.sensitivity import run_sensitivity


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="PMT position optimization via greedy voxel selection "
                    "with stochastic ratio scaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    parser.add_argument("hdf5_file", type=str,
                        help="Path to the HDF5 data file.")

    # --- Optimization parameters ---
    parser.add_argument("-N", type=int, default=None,
                        help="Number of voxels to select. "
                             "Required when --optimize is set.")
    parser.add_argument("--optimize", type=str, default=None,
                        choices=["nc", "muon-ge77"],
                        help="Optimization target. If not set, no greedy "
                             "is run (requires --write-hdf5).")
    parser.add_argument("-M", type=int, default=1,
                        help="Multiplicity threshold for NC detection. "
                             "Ignored in muon-ge77 mode (forced to 1).")
    parser.add_argument("-m", type=int, default=1,
                        help="Hit threshold per voxel.")
    parser.add_argument("-W", type=int, default=None,
                        help="Muon coincidence threshold. "
                             "Required for muon-ge77 mode.")

    # --- Area ratios ---
    parser.add_argument("--pit", type=float, default=None,
                        help=f"SSD/PMT ratio for PIT "
                             f"(default: {DEFAULT_AREA_RATIOS['pit']}).")
    parser.add_argument("--bot", type=float, default=None,
                        help=f"SSD/PMT ratio for BOT "
                             f"(default: {DEFAULT_AREA_RATIOS['bot']}).")
    parser.add_argument("--top", type=float, default=None,
                        help=f"SSD/PMT ratio for TOP "
                             f"(default: {DEFAULT_AREA_RATIOS['top']}).")
    parser.add_argument("--wall", type=float, default=None,
                        help=f"SSD/PMT ratio for WALL "
                             f"(default: {DEFAULT_AREA_RATIOS['wall']}).")

    # --- Greedy options ---
    parser.add_argument("--no-spacing", action="store_true",
                        help="Disable minimum spacing constraint.")
    parser.add_argument("--per-area", action="store_true",
                        help="Optimize each area independently.")
    parser.add_argument("--muon-weight", type=float, default=None, metavar="K",
                        help="Muon-level diminishing-returns weighting "
                             "with saturation constant k.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stochastic rounding.")

    # --- Optional modes ---
    parser.add_argument("--write-hdf5", action="store_true",
                        help="Write ratio-adjusted HDF5 file to disk.")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis with perturbed ratios.")
    parser.add_argument("--deltas", type=str, default=None,
                        help="Comma-separated delta values for sensitivity "
                             "(e.g. '-0.20,-0.10,-0.05,+0.05,+0.10,+0.20').")

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for all output files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")

    args = parser.parse_args(argv)
    verbose = not args.quiet
    min_spacing = 0.0 if args.no_spacing else 2 * PMT_RADIUS

    # --- Determine run mode ---
    run_greedy = args.optimize is not None
    run_write_hdf5 = args.write_hdf5
    do_sensitivity = args.sensitivity

    # --- Validate: must do at least something ---
    if not run_greedy and not run_write_hdf5:
        parser.error(
            "No action specified. Either set --optimize (nc/muon-ge77) "
            "to run greedy, or set --write-hdf5 to write a ratio-adjusted "
            "HDF5 file, or both."
        )

    # --- Validate: greedy-only parameters require --optimize ---
    if not run_greedy:
        greedy_only_errors = []
        if args.N is not None:
            greedy_only_errors.append("-N")
        if args.M != 1:
            greedy_only_errors.append("-M")
        if args.m != 1:
            greedy_only_errors.append("-m")
        if args.W is not None:
            greedy_only_errors.append("-W")
        if args.no_spacing:
            greedy_only_errors.append("--no-spacing")
        if args.per_area:
            greedy_only_errors.append("--per-area")
        if args.muon_weight is not None:
            greedy_only_errors.append("--muon-weight")
        if do_sensitivity:
            greedy_only_errors.append("--sensitivity")
        if args.deltas is not None:
            greedy_only_errors.append("--deltas")
        if greedy_only_errors:
            parser.error(
                f"The following arguments require --optimize to be set: "
                f"{', '.join(greedy_only_errors)}"
            )

    # --- Validate greedy-specific arguments ---
    if run_greedy:
        if args.N is None:
            parser.error("--optimize requires -N argument.")

        if args.optimize == "muon-ge77":
            if args.W is None:
                parser.error("--optimize muon-ge77 requires -W argument.")
            if args.M != 1:
                if verbose:
                    print(f"Warning: muon-ge77 mode forces M=1 "
                          f"(was M={args.M}).")
                args.M = 1

        if args.muon_weight is not None and args.per_area:
            parser.error("--muon-weight and --per-area cannot be combined.")

        if args.muon_weight is not None and args.muon_weight <= 0:
            parser.error("--muon-weight must be a positive float.")

    # --- Build ratios ---
    area_ratios = dict(DEFAULT_AREA_RATIOS)
    if args.pit is not None:
        area_ratios["pit"] = args.pit
    if args.bot is not None:
        area_ratios["bot"] = args.bot
    if args.top is not None:
        area_ratios["top"] = args.top
    if args.wall is not None:
        area_ratios["wall"] = args.wall

    for area, ratio in area_ratios.items():
        if ratio <= 0:
            parser.error(f"Ratio for {area} must be > 0 (got {ratio})")

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-generate output filename (only for greedy mode) ---
    base_name = None
    if run_greedy:
        spacing_tag = ("nospacing" if args.no_spacing
                       else f"spacing{int(min_spacing)}mm")
        perarea_tag = "_perarea" if args.per_area else ""
        mw_tag = (f"_mw{args.muon_weight:.2f}"
                  if args.muon_weight is not None else "")
        ratio_tag = f"_ratios_{fmt_ratio_filename(area_ratios)}"

        if args.optimize == "nc":
            base_name = (f"greedy_N{args.N}_M{args.M}_m{args.m}_"
                         f"{spacing_tag}{ratio_tag}{perarea_tag}{mw_tag}")
        else:
            base_name = (f"greedy_muon_N{args.N}_W{args.W}_m{args.m}_"
                         f"{spacing_tag}{ratio_tag}{perarea_tag}{mw_tag}")

    # =====================================================================
    # PHASE 1: Write ratio-adjusted HDF5 (optional)
    # =====================================================================
    if args.write_hdf5:
        if verbose:
            print("=" * 65)
            print("Phase 1: Writing ratio-adjusted HDF5")
            print("=" * 65)

        input_path = Path(args.hdf5_file)
        hdf5_output = (input_path.parent /
                       f"{input_path.stem}_ratios_"
                       f"{fmt_ratio_filename(area_ratios)}.hdf5")

        write_ratio_hdf5(
            input_path=str(input_path),
            output_path=str(hdf5_output),
            ratios=area_ratios,
            seed=args.seed,
            verbose=verbose,
        )

    # If no greedy requested, we're done after writing HDF5
    if not run_greedy:
        if verbose:
            print("\nNo --optimize set. Done.")
        return

    # =====================================================================
    # PHASE 2: Greedy voxel selection
    # =====================================================================
    if verbose:
        print("\n" + "=" * 65)
        print("Phase 2: Greedy Voxel Selection")
        print("=" * 65)
        print(f"  Mode:     {args.optimize}")
        print(f"  N:        {args.N}")
        print(f"  M:        {args.M}, m: {args.m}")
        if args.optimize == "muon-ge77":
            print(f"  W:        {args.W}")
        print(f"  Ratios:   {area_ratios}")
        print(f"  Seed:     {args.seed}")
        print(f"  Spacing:  {min_spacing:.0f} mm")
        print(f"  Per-area: {args.per_area}")
        if args.muon_weight is not None:
            print(f"  Muon weight: k = {args.muon_weight:.4f}")

    t_greedy_start = time.time()

    B, voxel_ids, centers, layers, num_primaries = load_and_binarize(
        args.hdf5_file, m=args.m, area_ratios=area_ratios,
        seed=args.seed, verbose=verbose,
    )

    # Load muon data if needed
    nc_muon_weight_data = None
    if args.optimize == "nc" and args.muon_weight is not None:
        nc_to_muon_local_nc, num_muons_nc, _ = load_nc_muon_ids(
            args.hdf5_file, num_ncs=B.shape[0], verbose=verbose,
        )
        nc_muon_weight_data = {
            "nc_to_muon_local": nc_to_muon_local_nc,
            "num_muons": num_muons_nc,
        }

    nc_to_muon_local = None
    eligible_nc_mask = None
    num_ge77_muons = 0
    ge77_muon_global_ids = None
    muon_nc_counts = None
    if args.optimize == "muon-ge77":
        global_muon_id, nc_time_ns, nc_flag_ge77 = load_muon_data(
            args.hdf5_file, num_ncs=B.shape[0], verbose=verbose,
        )
        (nc_to_muon_local, muon_nc_counts, ge77_muon_global_ids,
         eligible_nc_mask, num_ge77_muons) = build_muon_index(
            global_muon_id, nc_time_ns, nc_flag_ge77, verbose=verbose,
        )
        if verbose:
            max_ncs = int(muon_nc_counts.max()) if num_ge77_muons > 0 else 0
            print(f"  Max eligible NCs per Ge77 muon: {max_ncs}")
            if args.W is not None:
                n_feasible = int((muon_nc_counts >= args.W).sum())
                print(f"  Ge77 muons with ≥ W={args.W} eligible NCs: "
                      f"{n_feasible:,} / {num_ge77_muons:,}")

    # --- Run greedy ---
    if args.per_area:
        allocation = compute_per_area_N(args.N, verbose=verbose)
        all_selected_cols: list[int] = []

        if args.optimize == "muon-ge77":
            shared_nc_detected = np.zeros(B.shape[0], dtype=bool)
            shared_muon_counts = np.zeros(num_ge77_muons, dtype=np.int32)

        for area_name in ["pit", "bot", "top", "wall"]:
            n_area = allocation[area_name]
            if n_area == 0:
                if verbose:
                    print(f"\nSkipping {area_name}: 0 PMTs allocated.")
                continue

            area_mask = (layers == area_name)
            area_indices = np.where(area_mask)[0]
            n_voxels_area = len(area_indices)

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Per-area: {area_name} "
                      f"(N={n_area}, {n_voxels_area} voxels)")
                print(f"{'=' * 60}")

            if n_voxels_area == 0:
                if verbose:
                    print(f"  No valid voxels in {area_name}, skipping.")
                continue

            B_area = B[:, area_indices]
            centers_area = centers[area_indices]
            layers_area = layers[area_indices]

            if args.optimize == "nc":
                sel_local, _, _, _ = greedy_select_nc(
                    B_area, N=n_area, M=args.M,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing, verbose=verbose,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected_cols.extend(sel_global)

            elif args.optimize == "muon-ge77":
                sel_local, _, nc_det_area, muon_det_area = greedy_select_muon(
                    B_area, N=n_area, W=args.W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing,
                    muon_weight_k=args.muon_weight,
                    verbose=verbose,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected_cols.extend(sel_global)
                shared_nc_detected |= nc_det_area
                shared_muon_counts += muon_det_area

        selected_cols = all_selected_cols

        # Compute final metrics
        if args.optimize == "nc":
            coverage_counts = np.zeros(B.shape[0], dtype=np.int16)
            for col in selected_cols:
                s, e = B.indptr[col], B.indptr[col + 1]
                coverage_counts[B.indices[s:e]] += 1
            num_detected = int(np.sum(coverage_counts >= args.M))
            final_eff = num_detected / B.shape[0]
            efficiencies = [final_eff]
            muon_detected_counts = None
            nc_detected = None
        else:
            num_muons_detected = int(np.sum(shared_muon_counts >= args.W))
            final_eff = (num_muons_detected / num_ge77_muons
                         if num_ge77_muons > 0 else 0.0)
            efficiencies = [final_eff]
            muon_detected_counts = shared_muon_counts
            nc_detected = shared_nc_detected
            coverage_counts = None

    else:
        # Global optimization
        if args.optimize == "nc":
            mw_kwargs = {}
            if nc_muon_weight_data is not None:
                mw_kwargs = {
                    "muon_weight_k": args.muon_weight,
                    "nc_to_muon_local": nc_muon_weight_data["nc_to_muon_local"],
                    "num_muons": nc_muon_weight_data["num_muons"],
                }
            selected_cols, efficiencies, coverage_counts, muon_det_nc = (
                greedy_select_nc(
                    B, N=args.N, M=args.M,
                    centers=centers, layers=layers,
                    min_spacing=min_spacing, verbose=verbose,
                    **mw_kwargs,
                )
            )
            final_eff = efficiencies[-1]
            muon_detected_counts = muon_det_nc
            nc_detected = None

        elif args.optimize == "muon-ge77":
            selected_cols, efficiencies, nc_detected, muon_detected_counts = (
                greedy_select_muon(
                    B, N=args.N, W=args.W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    centers=centers, layers=layers,
                    min_spacing=min_spacing,
                    muon_weight_k=args.muon_weight,
                    verbose=verbose,
                )
            )
            final_eff = efficiencies[-1]

    t_greedy = time.time() - t_greedy_start
    selected_voxel_ids = voxel_ids[selected_cols]

    if verbose:
        print(f"\n{'=' * 60}")
        if args.optimize == "nc":
            print(f"Final NC detection efficiency: {final_eff:.4%}")
            if coverage_counts is not None:
                n_det = int(np.sum(coverage_counts >= args.M))
                print(f"Detected NCs: {n_det:,} / {B.shape[0]:,}")
        else:
            n_muon_det = int(np.sum(muon_detected_counts >= args.W))
            print(f"Final Ge77 muon detection efficiency: {final_eff:.4%}")
            print(f"Detected Ge77 muons: {n_muon_det:,} / {num_ge77_muons:,}")
            if nc_detected is not None:
                n_nc_det = int(nc_detected.sum())
                print(f"NCs detected (any): {n_nc_det:,} / {B.shape[0]:,}")
        print(f"Total primaries: {num_primaries:,}")
        print(f"Greedy time: {t_greedy:.1f}s")

    # --- Save results ---
    # NPZ
    npz_path = str(output_dir / f"{base_name}.npz")
    npz_data = {
        "selected_columns": np.array(selected_cols),
        "selected_voxel_ids": selected_voxel_ids,
        "efficiencies": np.array(efficiencies),
        "N": args.N, "M": args.M, "m": args.m,
        "optimize": args.optimize,
        "min_spacing": min_spacing,
        "num_ncs": B.shape[0],
        "num_primaries": num_primaries,
        "seed": args.seed,
        "area_ratios": json.dumps(area_ratios),
    }
    if coverage_counts is not None:
        npz_data["coverage_counts"] = coverage_counts
    if nc_detected is not None:
        npz_data["nc_detected"] = nc_detected
    if muon_detected_counts is not None:
        npz_data["muon_detected_counts"] = muon_detected_counts
    if ge77_muon_global_ids is not None:
        npz_data["ge77_muon_global_ids"] = ge77_muon_global_ids
    if args.optimize == "muon-ge77":
        npz_data["W"] = args.W
        npz_data["num_ge77_muons"] = num_ge77_muons
    if args.per_area:
        npz_data["per_area"] = True
        npz_data["allocation"] = json.dumps(
            compute_per_area_N(args.N, verbose=False)
        )

    np.savez(npz_path, **npz_data)
    if verbose:
        print(f"\nResults saved to {npz_path}")

    # JSON (selected voxels with geometry)
    json_path = output_dir / f"{base_name}.json"
    selected_voxels_json = []

    with h5py.File(args.hdf5_file, "r") as f:
        for col_idx in selected_cols:
            vid = voxel_ids[col_idx]
            center = f[f"voxels/{vid}/center"][:].tolist()
            corners_x = f[f"voxels/{vid}/corners/x"][:].tolist()
            corners_y = f[f"voxels/{vid}/corners/y"][:].tolist()
            corners_z = f[f"voxels/{vid}/corners/z"][:].tolist()
            corners = [[x, y, z]
                       for x, y, z in zip(corners_x, corners_y, corners_z)]
            layer_raw = f[f"voxels/{vid}/layer"][()]
            layer = (layer_raw.decode() if isinstance(layer_raw, bytes)
                     else str(layer_raw))
            selected_voxels_json.append({
                "index": vid,
                "center": center,
                "corners": corners,
                "layer": layer,
            })

    json_data = {
        "config": {
            "optimize": args.optimize,
            "N": args.N, "M": args.M, "m": args.m,
            "W": args.W,
            "area_ratios": area_ratios,
            "seed": args.seed,
            "min_spacing": min_spacing,
            "per_area": args.per_area,
            "muon_weight_k": args.muon_weight,
        },
        "efficiency": final_eff,
        "num_ncs": B.shape[0],
        "num_primaries": num_primaries,
        "selected_voxels": selected_voxels_json,
    }
    with open(json_path, "w") as jf:
        json.dump(json_data, jf, indent=2)
    if verbose:
        print(f"JSON saved to {json_path}")

    # Print selected voxels
    if verbose:
        print(f"\nSelected voxels (in selection order):")
        for rank, (col, vid) in enumerate(
            zip(selected_cols, selected_voxel_ids)
        ):
            if len(efficiencies) > rank:
                eff_val = efficiencies[rank]
                eff_delta = eff_val - (
                    efficiencies[rank - 1] if rank > 0 else 0.0
                )
                print(f"  {rank+1:>3}. Voxel {vid} (col {col}), "
                      f"cumulative eff = {eff_val:.4%}, "
                      f"Δeff = {eff_delta:.4%}")
            else:
                print(f"  {rank+1:>3}. Voxel {vid} (col {col})")

    # Plots
    if args.muon_weight is not None and muon_detected_counts is not None:
        hist_path = output_dir / f"{base_name}_muon_hist.png"
        title_info = f"mode={args.optimize}, k={args.muon_weight:.2f}"
        if args.optimize == "muon-ge77":
            title_info += f", W={args.W}"
        plot_muon_nc_histogram(muon_detected_counts, hist_path, title_info)

    plot_centers = centers[selected_cols]
    plot_layers = layers[selected_cols]
    plot_ids = [str(voxel_ids[c]) for c in selected_cols]
    plot_path = output_dir / f"{base_name}.png"

    title_extra = f"mode={args.optimize}"
    if args.optimize == "nc":
        title_extra += f", M={args.M}, m={args.m}"
    else:
        title_extra += f", W={args.W}, m={args.m}"
    if args.per_area:
        title_extra += ", per-area"

    plot_selected_voxels(
        plot_centers, plot_layers, plot_ids,
        output_path=plot_path, title_extra=title_extra,
    )

    # Free B before sensitivity
    del B

    # =====================================================================
    # PHASE 3: Sensitivity analysis (optional)
    # =====================================================================
    if args.sensitivity:
        if verbose:
            print("\n" + "=" * 65)
            print("Phase 3: Sensitivity Analysis")
            print("=" * 65)

        deltas = None
        if args.deltas is not None:
            deltas = [float(d.strip()) for d in args.deltas.split(",")]

        sens_output_dir = str(output_dir / "sensitivity")

        run_sensitivity(
            filepath=args.hdf5_file,
            N=args.N, m=args.m,
            area_ratios=area_ratios,
            optimize=args.optimize,
            M=args.M,
            W=args.W if args.W is not None else 1,
            min_spacing=min_spacing,
            per_area=args.per_area,
            muon_weight_k=args.muon_weight,
            seed=args.seed,
            deltas=deltas,
            output_dir=sens_output_dir,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()