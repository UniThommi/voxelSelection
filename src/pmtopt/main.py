#!/usr/bin/env python3
"""
PMT Position Optimization — unified entry point
================================================

Subcommands:
  greedy           Greedy voxel selection from HDF5 simulation data (NC / muon-Ge77 modes)
  homogeneous      Generate all voxels or select N homogeneously distributed voxels
  rotate           Rotate an existing voxel selection by an azimuthal angle
  plot             Generate 3D plots for existing JSON voxel selection files
  plot-w2          W2 explanation visualizations (plots A–F) from JSON config files
  sample-w2-range  Generate configurations spanning the W2 homogeneity range

Usage examples:
    # Greedy: NC mode
    python -m pmtopt.main greedy data.hdf5 -N 300 --optimize nc -M 1 -m 1

    # Greedy: muon-Ge77 mode
    python -m pmtopt.main greedy data.hdf5 -N 300 --optimize muon-ge77 -W 6 -m 1

    # Greedy: write ratio-adjusted HDF5 + sensitivity
    python -m pmtopt.main greedy data.hdf5 -N 300 --optimize nc -M 1 -m 1 \\
        --write-hdf5 --sensitivity --output-dir results/

    # Homogeneous: generate all voxels
    python -m pmtopt.main homogeneous --mode generate --output-dir ./output

    # Homogeneous: select 300 PMTs across all areas
    python -m pmtopt.main homogeneous --all-voxels all_valid.json --mode select -N 300 --output-dir ./output

    # Rotate: single angle
    python -m pmtopt.main rotate --all-voxels all.json --selected greedy.json --angle 0.25 \\
        --output-dir ./output

    # Rotate: explore all valid angles
    python -m pmtopt.main rotate --all-voxels all.json --selected greedy.json \\
        --output-dir ./output

    # Plot: generate 3D PNG for one or more existing JSON selections
    python -m pmtopt.main plot greedy_N300.json setup_*.json

    # Sample W2 range: 50 geometry-driven configs spanning W2 space
    python -m pmtopt.main sample-w2-range --hdf5 data.hdf5 -N 300 --n-configs 50 \\
        --output-dir w2_setups/

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
    load_raw_sparse, binarize_from_raw,
    load_muon_data, build_muon_index, load_nc_muon_ids,
    load_invalid_voxel_data,
)
from pmtopt.greedy import greedy_select_nc, greedy_select_muon
from pmtopt.plotting import (
    plot_selected_voxels, plot_selected_voxels_grid, plot_muon_nc_histogram,
    plot_hit_heatmap, plot_marginal_gain_heatmap, plot_ssd_voxels_3d,
)
from pmtopt.ratio_scaling import write_ratio_hdf5, fmt_ratio_filename
from pmtopt.sensitivity import run_sensitivity


def run_greedy(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Greedy PMT voxel selection with stochastic ratio scaling.",
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

    # --- Area filter ---
    parser.add_argument("--areas", nargs="+", default=None,
                        choices=["pit", "bot", "top", "wall"], metavar="AREA",
                        help="Restrict PMT placement to these areas. "
                             "Choices: pit bot top wall. Default: all four. "
                             "With --per-area, N is redistributed proportionally "
                             "over the selected surfaces only.")

    # --- Greedy options ---
    parser.add_argument("--no-spacing", action="store_true",
                        help="Disable minimum spacing constraint.")
    parser.add_argument("--per-area", action="store_true",
                        help="Optimize each area independently.")
    parser.add_argument("--muon-weight", type=float, default=None, metavar="K",
                        help="Muon-level diminishing-returns weighting "
                             "with saturation constant k.")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic fallback instead of priority logic. "
                             "muon-ge77: sweeps W high→low at each M level "
                             "(high→low). nc: sweeps M high→low.")
    parser.add_argument("--worst", action="store_true",
                        help="Worst-case benchmark: select voxels with "
                             "minimal gain, avoiding M/W promotions. "
                             "Cannot be combined with --dynamic.")
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
    do_greedy = args.optimize is not None
    run_write_hdf5 = args.write_hdf5
    do_sensitivity = args.sensitivity

    # --- Validate: must do at least something ---
    if not do_greedy and not run_write_hdf5:
        parser.error(
            "No action specified. Either set --optimize (nc/muon-ge77) "
            "to run greedy, or set --write-hdf5 to write a ratio-adjusted "
            "HDF5 file, or both."
        )

    # --- Validate: greedy-only parameters require --optimize ---
    if not do_greedy:
        greedy_only_errors = []
        if args.N is not None:
            greedy_only_errors.append("-N")
        if args.M != 1:
            greedy_only_errors.append("-M")
        if args.m != 1:
            greedy_only_errors.append("-m")
        if args.W is not None:
            greedy_only_errors.append("-W")
        if args.areas is not None:
            greedy_only_errors.append("--areas")
        if args.no_spacing:
            greedy_only_errors.append("--no-spacing")
        if args.per_area:
            greedy_only_errors.append("--per-area")
        if args.muon_weight is not None:
            greedy_only_errors.append("--muon-weight")
        if do_sensitivity:
            greedy_only_errors.append("--sensitivity")
        if args.dynamic:
            greedy_only_errors.append("--dynamic")
        if args.worst:
            greedy_only_errors.append("--worst")
        if args.deltas is not None:
            greedy_only_errors.append("--deltas")
        if greedy_only_errors:
            parser.error(
                f"The following arguments require --optimize to be set: "
                f"{', '.join(greedy_only_errors)}"
            )

    # --- Validate greedy-specific arguments ---
    if do_greedy:
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

        if args.muon_weight is not None and args.worst:
            parser.error("--muon-weight and --worst cannot be combined.")

        if args.muon_weight is not None and args.muon_weight <= 0:
            parser.error("--muon-weight must be a positive float.")

        if args.dynamic and args.worst:
            parser.error("--dynamic and --worst cannot be combined.")

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

    # --- Normalise selected areas ---
    _area_order = ["pit", "bot", "top", "wall"]
    selected_areas: list[str] | None = (
        sorted(set(args.areas), key=_area_order.index)
        if args.areas is not None else None
    )

    # --- Auto-generate output filename (only for greedy mode) ---
    base_name = None
    if do_greedy:
        spacing_tag = ("nospacing" if args.no_spacing
                       else f"spacing{int(min_spacing)}mm")
        perarea_tag = "_perarea" if args.per_area else ""
        worst_tag = "_worst" if args.worst else ""
        mw_tag = (f"_mw{args.muon_weight:.2f}"
                  if args.muon_weight is not None else "")
        ratio_tag = f"_ratios_{fmt_ratio_filename(area_ratios)}"
        areas_tag = (
            "_areas_" + "".join(a[0] for a in selected_areas)
            if selected_areas is not None else ""
        )

        if args.optimize == "nc":
            base_name = (f"greedy_N{args.N}_M{args.M}_m{args.m}_"
                         f"{spacing_tag}{ratio_tag}{perarea_tag}"
                         f"{areas_tag}{worst_tag}{mw_tag}")
        else:
            base_name = (f"greedy_muon_N{args.N}_W{args.W}_m{args.m}_"
                         f"{spacing_tag}{ratio_tag}{perarea_tag}"
                         f"{areas_tag}{worst_tag}{mw_tag}")

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
    if not do_greedy:
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
        if selected_areas is not None:
            print(f"  Areas:    {selected_areas}")
        if args.muon_weight is not None:
            print(f"  Muon weight: k = {args.muon_weight:.4f}")

    t_greedy_start = time.time()

    (raw_rows, raw_cols, raw_vals,
     voxel_ids, centers, layers,
     num_ncs, num_primaries) = load_raw_sparse(
        args.hdf5_file, verbose=verbose,
        skip_validity=args.no_spacing,
    )

    # ---- Load invalid voxel data for visualization ----
    # Only needed when the validity filter is active (i.e. --no-spacing not set).
    if not args.no_spacing:
        _inv_centers, _inv_layers, _inv_hits = load_invalid_voxel_data(
            args.hdf5_file, set(voxel_ids), verbose=verbose,
        )
    else:
        _inv_centers = np.empty((0, 3), dtype=np.float64)
        _inv_layers  = np.empty(0, dtype=object)
        _inv_hits    = np.empty(0, dtype=np.float64)

    # ---- Hit distribution heatmap (raw hits, all areas, incl. invalid) ----
    if verbose:
        print("\nGenerating hit distribution heatmap ...")
    _hits_per_voxel = np.bincount(
        raw_cols, weights=raw_vals.astype(float), minlength=len(voxel_ids),
    )
    _areas_hit_data = {}
    for _area in ["pit", "bot", "top", "wall"]:
        _mask = layers == _area
        _mask_inv = _inv_layers == _area
        _vc = centers[_mask] if _mask.any() else np.empty((0, 3))
        _vh = _hits_per_voxel[_mask] if _mask.any() else np.empty(0)
        _ic = _inv_centers[_mask_inv] if _mask_inv.any() else np.empty((0, 3))
        _ih = _inv_hits[_mask_inv] if _mask_inv.any() else np.empty(0)
        _ac = np.vstack([_vc, _ic]) if (len(_vc) or len(_ic)) else np.empty((0, 3))
        _ah = np.concatenate([_vh, _ih]) if (len(_vh) or len(_ih)) else np.empty(0)
        if len(_ac):
            _areas_hit_data[_area] = (_ac, _ah)
    plot_hit_heatmap(
        _areas_hit_data,
        output_dir / f"{base_name}_hit_heatmap.png",
        num_primaries=num_primaries,
        num_ncs=num_ncs,
    )

    # ---- 3D SSD voxel geometry plot ----
    if verbose:
        print("Generating 3D voxel geometry plot ...")
    plot_ssd_voxels_3d(
        centers, layers,
        output_dir / f"{base_name}_ssd_voxels.png",
        invalid_centers=_inv_centers,
        invalid_layers=_inv_layers,
    )

    B = binarize_from_raw(
        raw_rows, raw_cols, raw_vals,
        num_ncs=num_ncs,
        num_voxels=len(voxel_ids),
        layers=layers,
        area_ratios=area_ratios,
        m=args.m,
        seed=args.seed,
    )

    # Save full (unfiltered) arrays for sensitivity — run_sensitivity handles
    # area filtering internally and needs the complete voxel space.
    voxel_ids_full = voxel_ids
    centers_full = centers
    layers_full = layers

    # Filter voxels to selected areas (if --areas was given)
    if selected_areas is not None:
        _area_mask = np.isin(layers, selected_areas)
        _area_indices = np.where(_area_mask)[0]
        B = B[:, _area_indices]
        voxel_ids = voxel_ids[_area_indices]
        centers = centers[_area_indices]
        layers = layers[_area_indices]
        if verbose:
            print(f"  Areas filter: kept {len(_area_indices)} / "
                  f"{len(_area_mask)} voxels "
                  f"(areas: {selected_areas})")

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
        allocation = compute_per_area_N(
            args.N, areas=selected_areas, verbose=verbose,
        )
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
                    min_spacing=min_spacing, dynamic=args.dynamic,
                    worst=args.worst, verbose=verbose,
                )
                sel_global = [int(area_indices[i]) for i in sel_local]
                all_selected_cols.extend(sel_global)

            elif args.optimize == "muon-ge77":
                sel_local, _, nc_det_area, muon_det_area, _ = greedy_select_muon(
                    B_area, N=n_area, W=args.W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    M=args.M,
                    centers=centers_area, layers=layers_area,
                    min_spacing=min_spacing,
                    muon_weight_k=args.muon_weight,
                    dynamic=args.dynamic,
                    worst=args.worst,
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

            # Marginal gain snapshot at step 10 (after 10 voxels selected)
            _marginal_snapshot: dict = {}

            def _snapshot_cb(gains_raw, avail, sel):
                _marginal_snapshot["gains"] = gains_raw
                _marginal_snapshot["available"] = avail
                _marginal_snapshot["selected"] = sel

            _snap_step = 10 if args.N > 10 else None

            selected_cols, efficiencies, coverage_counts, muon_det_nc = (
                greedy_select_nc(
                    B, N=args.N, M=args.M,
                    centers=centers, layers=layers,
                    min_spacing=min_spacing, dynamic=args.dynamic,
                    worst=args.worst, verbose=verbose,
                    on_step_snapshot=_snapshot_cb,
                    snapshot_step=_snap_step,
                    **mw_kwargs,
                )
            )
            final_eff = efficiencies[-1]
            muon_detected_counts = muon_det_nc
            nc_detected = None

            if _marginal_snapshot:
                _mg_path = output_dir / f"{base_name}_marginal_gain_step10.png"
                if verbose:
                    print("\nGenerating marginal gain heatmap (step 10) ...")
                plot_marginal_gain_heatmap(
                    _marginal_snapshot["gains"],
                    _marginal_snapshot["available"],
                    _marginal_snapshot["selected"],
                    centers, layers,
                    output_path=_mg_path,
                    step=10,
                    invalid_centers=_inv_centers,
                    invalid_layers=_inv_layers,
                )

        elif args.optimize == "muon-ge77":
            selected_cols, efficiencies, nc_detected, muon_detected_counts, _ = (
                greedy_select_muon(
                    B, N=args.N, W=args.W,
                    nc_to_muon_local=nc_to_muon_local,
                    eligible_nc_mask=eligible_nc_mask,
                    num_ge77_muons=num_ge77_muons,
                    M=args.M,
                    centers=centers, layers=layers,
                    min_spacing=min_spacing,
                    muon_weight_k=args.muon_weight,
                    dynamic=args.dynamic,
                    worst=args.worst,
                    verbose=verbose,
                )
            )
            final_eff = efficiencies[-1]
            coverage_counts = None

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
        npz_data["allocation"] = json.dumps(allocation)

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
            "areas": selected_areas,
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

    _plot_w2: "float | None" = None
    if len(plot_centers) >= 2:
        from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
        _plot_w2 = compute_wasserstein_homogeneity(
            plot_centers, reference=get_w2_ref()
        )["w2"]

    plot_selected_voxels(
        plot_centers, plot_layers, plot_ids,
        output_path=plot_path, title_extra=title_extra, w2=_plot_w2,
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
            areas=selected_areas,
            muon_weight_k=args.muon_weight,
            seed=args.seed,
            deltas=deltas,
            output_dir=sens_output_dir,
            dynamic=args.dynamic,
            worst=args.worst,
            baseline_selected=selected_cols,
            baseline_eff=final_eff,
            raw_rows=raw_rows,
            raw_cols=raw_cols,
            raw_vals=raw_vals,
            voxel_ids=voxel_ids_full,
            centers=centers_full,
            layers=layers_full,
            num_ncs=num_ncs,
            num_primaries=num_primaries,
            verbose=verbose,
        )


def run_plot(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D plots for existing JSON voxel selection files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_files", nargs="+", type=str,
        help="One or more paths to JSON voxel selection files. "
             "A single file produces one PNG next to the JSON; two or more "
             "files are tiled into combined grid figure(s) (4 / 2 / 1 panels "
             "per figure).",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Custom title suffix for the single-file plot. If not set, uses "
             "config metadata from the JSON (N, efficiency, etc.). "
             "Ignored when multiple files are given (use --labels).",
    )
    parser.add_argument(
        "--labels", nargs="*", default=None, metavar="LABEL",
        help="Display labels for the combined grid panels (same order as the "
             "JSON files). Missing labels fall back to the file stem.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for the combined grid figure(s). "
             "Default: the directory of the first JSON file.",
    )
    parser.add_argument(
        "--name", type=str, default="combined_grid",
        help="Base filename for the combined grid figure(s). Multiple "
             "figures are suffixed _01, _02, ...",
    )
    args = parser.parse_args(argv)

    # --- Multiple files: combined grid figure(s) ---
    if len(args.json_files) >= 2:
        _run_plot_grid(args)
        return

    for json_path_str in args.json_files:
        json_path = Path(json_path_str)
        if not json_path.exists():
            print(f"ERROR: file not found: {json_path}", file=sys.stderr)
            continue

        with open(json_path) as jf:
            data = json.load(jf)

        selected_voxels = data.get("selected_voxels", [])
        if not selected_voxels:
            print(f"WARNING: no selected_voxels in {json_path.name}, skipping.")
            continue

        centers_list = [v["center"] for v in selected_voxels]
        layers_list = [v["layer"] for v in selected_voxels]
        ids_list = [str(v["index"]) for v in selected_voxels]

        centers = np.array(centers_list, dtype=float)
        layers = np.array(layers_list)

        if args.title is not None:
            title_extra = args.title
        else:
            cfg = data.get("config", {})
            eff = data.get("efficiency")
            parts = [f"N={len(selected_voxels)}"]
            if cfg.get("optimize"):
                parts.append(f"mode={cfg['optimize']}")
            if cfg.get("M") is not None:
                parts.append(f"M={cfg['M']}")
            if cfg.get("m") is not None:
                parts.append(f"m={cfg['m']}")
            if eff is not None:
                parts.append(f"eff={eff:.4%}")
            title_extra = "  ".join(parts)

        _plot_w2: "float | None" = None
        if len(centers) >= 2:
            from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref
            _plot_w2 = compute_wasserstein_homogeneity(
                centers, reference=get_w2_ref()
            )["w2"]

        png_path = json_path.with_suffix(".png")
        plot_selected_voxels(centers, layers, ids_list,
                             output_path=png_path, title_extra=title_extra,
                             w2=_plot_w2)
        print(f"  -> {png_path}")


def _run_plot_grid(args) -> None:
    """Load ≥2 JSON selections and tile them into combined grid figure(s)."""
    from pmtopt.homogeneous import compute_wasserstein_homogeneity, get_w2_ref

    if args.labels and len(args.labels) != len(args.json_files):
        print(f"WARNING: got {len(args.labels)} labels for "
              f"{len(args.json_files)} files; missing labels use the file stem.")

    setups: list[dict] = []
    for i, json_path_str in enumerate(args.json_files):
        json_path = Path(json_path_str)
        if not json_path.exists():
            print(f"ERROR: file not found: {json_path}", file=sys.stderr)
            continue

        with open(json_path) as jf:
            data = json.load(jf)

        selected_voxels = data.get("selected_voxels", [])
        if not selected_voxels:
            print(f"WARNING: no selected_voxels in {json_path.name}, skipping.")
            continue

        centers = np.array([v["center"] for v in selected_voxels], dtype=float)
        layers = np.array([v["layer"] for v in selected_voxels])
        label = (args.labels[i]
                 if args.labels and i < len(args.labels)
                 else json_path.stem)

        w2 = None
        if len(centers) >= 2:
            w2 = compute_wasserstein_homogeneity(
                centers, reference=get_w2_ref())["w2"]

        setups.append({"centers": centers, "layers": layers,
                       "label": label, "w2": w2})

    if not setups:
        print("ERROR: no valid selections to plot.", file=sys.stderr)
        return

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(args.json_files[0]).resolve().parent)
    plot_selected_voxels_grid(setups, output_dir, name=args.name)


def run_plot_hits(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Heatmap of total photon hits per voxel across all NCs, split by area.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_file", type=str,
                        help="Path to the voxelSelection HDF5 file (target_matrix).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the PNG (default: same as HDF5 file).")
    args = parser.parse_args(argv)

    import h5py
    from collections import defaultdict

    hdf5_path = Path(args.hdf5_file)
    output_dir = Path(args.output_dir) if args.output_dir else hdf5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {hdf5_path} ...")
    with h5py.File(hdf5_path, "r") as f:
        print("  Loading target_matrix ...")
        target = f["target_matrix"][:]           # (num_NCs, num_voxels)
        columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        ]
        num_ncs_phits = int(target.shape[0])
        num_primaries_phits = int(f["primaries"][()])
        print(f"  {num_ncs_phits:,} NCs × {target.shape[1]:,} voxels")
        print(f"  {num_primaries_phits:,} primaries")

        print("  Loading voxel geometry ...")
        center_map: dict[str, np.ndarray] = {}
        layer_map:  dict[str, str]        = {}
        for voxel_id in f["voxels"].keys():
            vg = f["voxels"][voxel_id]
            center_map[voxel_id] = vg["center"][:]
            raw_layer = vg["layer"][()]
            layer_map[voxel_id] = (
                raw_layer.decode() if isinstance(raw_layer, bytes) else str(raw_layer)
            )

    hits_per_voxel = target.sum(axis=0)   # (num_voxels,)

    area_centers: dict[str, list] = defaultdict(list)
    area_hits:    dict[str, list] = defaultdict(list)
    for col_idx, voxel_id in enumerate(columns):
        if voxel_id not in center_map:
            continue
        area = layer_map[voxel_id]
        area_centers[area].append(center_map[voxel_id])
        area_hits[area].append(float(hits_per_voxel[col_idx]))

    areas_data = {
        area: (np.array(area_centers[area]), np.array(area_hits[area]))
        for area in area_centers
    }

    output_path = output_dir / f"{hdf5_path.stem}_hit_heatmap.png"
    plot_hit_heatmap(
        areas_data, output_path,
        num_primaries=num_primaries_phits, num_ncs=num_ncs_phits,
    )


def run_plot_w2(argv: Optional[list[str]] = None) -> None:
    """Generate W2 explanation visualizations (plots A–F) from JSON config files."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate W2 homogeneity explanation plots (A–F) from PMT JSON config files.\n"
            "Plots: A=3D scatter, B=CDF marginals, C=transport arrows, "
            "D=cost heatmap, E=density diff, F=N-scaling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_files", nargs="+", metavar="JSON",
        help="One or more PMT JSON config files (voxel selections).",
    )
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--labels", nargs="*", metavar="LABEL",
        help="Optional display labels for each JSON file (same order).",
    )
    args = parser.parse_args(argv)

    from pmtopt.w2_plot_helpers import plot_all_w2_explanation
    plot_all_w2_explanation(
        json_paths=args.json_files,
        output_dir=args.output_dir,
        labels=args.labels,
    )


_SUBCOMMANDS = ("greedy", "homogeneous", "rotate", "plot", "plot-hits", "plot-w2", "sample-w2-range")


def main(argv: Optional[list[str]] = None) -> None:
    """Top-level dispatcher: routes to greedy / homogeneous / rotate."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] not in _SUBCOMMANDS:
        print(
            "usage: main.py <subcommand> [args ...]\n"
            f"subcommands: {', '.join(_SUBCOMMANDS)}\n\n"
            "  greedy           Greedy PMT selection from HDF5 simulation data\n"
            "  homogeneous      Generate or select homogeneously distributed voxels\n"
            "  rotate           Rotate an existing voxel selection azimuthally\n"
            "  plot             Plot existing JSON voxel selection file(s)\n"
            "  plot-hits        Heatmap of total photon hits per voxel (light distribution)\n"
            "  plot-w2          W2 explanation visualizations (A–F) from JSON config files\n"
            "  sample-w2-range  Generate configurations spanning the W2 range\n\n"
            "Run 'main.py <subcommand> --help' for subcommand-specific help."
        )
        sys.exit(1 if argv else 0)

    mode, rest = argv[0], argv[1:]

    if mode == "greedy":
        run_greedy(rest)
    elif mode == "homogeneous":
        from pmtopt.homogeneous import main as hom_main
        hom_main(rest)
    elif mode == "rotate":
        from pmtopt.rotate import main as rot_main
        rot_main(rest)
    elif mode == "plot":
        run_plot(rest)
    elif mode == "plot-hits":
        run_plot_hits(rest)
    elif mode == "plot-w2":
        run_plot_w2(rest)
    elif mode == "sample-w2-range":
        from pmtopt.sample_w2_range import main as w2_main
        w2_main(rest)


if __name__ == "__main__":
    main()