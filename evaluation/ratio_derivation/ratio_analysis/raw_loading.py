"""Raw LGDO HDF5 loading for photon detection data.

Loads data directly from the raw LGDO format produced by REMAGE/Geant4:
  - /hit/MyNeutronCaptureOutput  (Sim 1 — NC truth)
  - /hit/optical                 (Sim 2 — photon detection)

Mirrors the loading logic of evaluation/comparePMTCoverage.py but
generalises to multiple run directories and returns outputs suitable
for sparse-matrix construction.
"""

from __future__ import annotations

import gc
import glob
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ──────────────────────────────────────────────────────────────────────
# Constants (same as comparePMTCoverage.py)
# ──────────────────────────────────────────────────────────────────────
FLOAT_TOL_NS: float = -1.0     # tolerance for small negative times (float rounding)
TIME_CUT_NC_NS: float = 200.0  # photon detection window after NC [ns]


# ──────────────────────────────────────────────────────────────────────
# Low-level I/O helpers
# ──────────────────────────────────────────────────────────────────────
def _read_pages(group: h5py.Group, field_name: str) -> np.ndarray:
    """Read a field stored in LGDO column format (pages sub-array)."""
    return group[field_name]["pages"][:]


def _load_nc_from_file(fpath: str) -> pd.DataFrame | None:
    """Load MyNeutronCaptureOutput from one HDF5 file. Returns None if empty."""
    with h5py.File(fpath, "r") as f:
        grp = f["hit"]["MyNeutronCaptureOutput"]
        if int(grp["entries"][()]) == 0:
            return None
        return pd.DataFrame({
            "muon_id":    _read_pages(grp, "evtid"),
            "nc_id":      _read_pages(grp, "nC_track_id"),
            "nc_time_ns": _read_pages(grp, "nC_time_in_ns"),
            "flag_ge77":  _read_pages(grp, "nC_flag_Ge77"),
            "nc_x":       _read_pages(grp, "nC_x_position_in_m"),
            "nc_y":       _read_pages(grp, "nC_y_position_in_m"),
            "nc_z":       _read_pages(grp, "nC_z_position_in_m"),
        })


def _load_optical_from_file(fpath: str) -> pd.DataFrame | None:
    """Load optical hit data from one HDF5 file. Returns None if empty."""
    with h5py.File(fpath, "r") as f:
        grp = f["hit"]["optical"]
        if int(grp["entries"][()]) == 0:
            return None
        return pd.DataFrame({
            "muon_track_id": _read_pages(grp, "muon_track_id"),
            "nC_track_id":   _read_pages(grp, "nC_track_id"),
            "det_uid":       _read_pages(grp, "det_uid"),
            "time_in_ns":    _read_pages(grp, "time_in_ns"),
        })


def _count_vertices_from_file(fpath: str) -> int:
    """Read vertex entry count from one HDF5 file."""
    with h5py.File(fpath, "r") as f:
        return int(f["hit"]["vertices"]["entries"][()])


# ──────────────────────────────────────────────────────────────────────
# Run-directory loaders
# ──────────────────────────────────────────────────────────────────────
def _list_run_dirs(base_dir: str, omit_runs: set[str] | None = None) -> list[str]:
    """Return sorted list of run_NNN subdirectories found in base_dir.

    Directories whose basename appears in omit_runs are silently excluded.
    """
    pattern = os.path.join(base_dir, "run_*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if omit_runs:
        dirs = [d for d in dirs if os.path.basename(d) not in omit_runs]
    if not dirs:
        raise FileNotFoundError(
            f"No run_* subdirectories found in {base_dir!r}"
        )
    return dirs


def _run_id_from_dir(run_dir: str) -> int:
    """Extract integer run ID from a run_NNN directory name (e.g. run_001 → 1)."""
    try:
        return int(os.path.basename(run_dir).split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot extract integer run ID from {run_dir!r}. "
            "Expected directory name of the form run_NNN."
        ) from exc


def _load_nc_from_run_dir(run_dir: str, run_id: int) -> pd.DataFrame:
    """Concatenate NC truth DataFrames from all output_t*.hdf5 in run_dir.

    Raises RuntimeError if the same (muon_id, nc_id) appears in more than
    one output_t file within this run — such duplicates indicate a simulation
    output bug and must not be silently dropped.

    Adds a ``run_id`` column (integer extracted from the directory name) so
    that NCs from different runs can be distinguished after concatenation.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No output_t*.hdf5 in {run_dir!r}")
    frames = [_load_nc_from_file(f) for f in files]
    frames = [df for df in frames if df is not None]
    if not frames:
        raise ValueError(f"No NC entries in {run_dir!r}")
    df = pd.concat(frames, ignore_index=True)

    # Detect cross-file duplicates within this run — must never be silently dropped.
    dup_mask = df.duplicated(subset=["muon_id", "nc_id"], keep=False)
    if dup_mask.any():
        dup_df = df.loc[dup_mask, ["muon_id", "nc_id"]].drop_duplicates()
        n_dup_pairs = len(dup_df)
        sample = dup_df.head(5).to_dict(orient="records")
        raise RuntimeError(
            f"Run {os.path.basename(run_dir)!r} (run_id={run_id}): "
            f"{n_dup_pairs} NC(s) appear in more than one output_t file. "
            f"This is a simulation output bug — NCs must not be silently dropped. "
            f"Sample duplicates (muon_id, nc_id): {sample}"
        )

    df["run_id"] = run_id
    return df


def _load_optical_from_run_dir(run_dir: str, run_id: int) -> pd.DataFrame:
    """Concatenate optical DataFrames from all output_t*.hdf5 in run_dir.

    Adds a ``run_id`` column so that optical hits can be matched to NC truth
    rows by (run_id, muon_track_id, nC_track_id).
    """
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No output_t*.hdf5 in {run_dir!r}")
    frames = [_load_optical_from_file(f) for f in files]
    frames = [df for df in frames if df is not None]
    if not frames:
        return pd.DataFrame(
            columns=["run_id", "muon_track_id", "nC_track_id", "det_uid", "time_in_ns"]
        )
    df = pd.concat(frames, ignore_index=True)
    df["run_id"] = run_id
    return df


def _count_vertices_run_dir(run_dir: str) -> int:
    """Sum vertex entry counts across all output_t*.hdf5 in run_dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
    return sum(_count_vertices_from_file(f) for f in files)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def check_all_files_integrity(
    nc_dir: str,
    sim_dirs: list[str],
    labels: list[str],
    omit_runs: set[str] | None = None,
) -> None:
    """Check all output_t*.hdf5 files for corruption before any analysis.

    For every run_* subdirectory in nc_dir and each sim_dir, attempts to
    open every HDF5 file. If any file in a run directory is unreadable
    (truncated, corrupt), the entire run is flagged — because all files
    in a run belong to the same simulation and cannot be treated
    individually.

    Prints a full list of defective runs and raises RuntimeError if any
    are found. If all files are intact, prints a confirmation and returns.

    Parameters
    ----------
    nc_dir:
        NC truth simulation directory (Sim 1).
    sim_dirs:
        List of optical simulation directories (Sim 2), one per setup.
    labels:
        Human-readable label for each entry in sim_dirs.
    """
    all_dirs = [(nc_dir, "nc_dir")] + list(zip(sim_dirs, labels))

    # (dir_label, run_label) -> count of corrupt files in that run
    defective_runs: dict[tuple[str, str], int] = {}
    n_runs_checked = 0

    print("Checking HDF5 file integrity ...")
    for base_dir, dir_label in all_dirs:
        try:
            run_dirs = _list_run_dirs(base_dir, omit_runs=omit_runs)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc

        for run_dir in run_dirs:
            n_runs_checked += 1
            run_label = os.path.basename(run_dir)
            files = sorted(glob.glob(os.path.join(run_dir, "output_t*.hdf5")))
            n_corrupt = 0
            for fpath in files:
                try:
                    with h5py.File(fpath, "r"):
                        pass
                except OSError:
                    n_corrupt += 1
            if n_corrupt > 0:
                defective_runs[(dir_label, run_label)] = n_corrupt

    if defective_runs:
        n_bad_runs = len(defective_runs)
        print(f"\n  [ERROR] Found corrupt/truncated file(s) in {n_bad_runs} run(s):\n")
        for (dir_label, run_label), n_corrupt in sorted(defective_runs.items()):
            print(f"    Setup: {dir_label}  |  Run: {run_label}  ({n_corrupt} corrupt file(s))")

        raise RuntimeError(
            f"Integrity check failed: {n_bad_runs} run(s) contain corrupt "
            "HDF5 file(s). Fix or remove the defective runs before proceeding."
        )

    setup_names = "  +  ".join(lbl for _, lbl in all_dirs)
    print(f"  All files intact — setups checked: [{setup_names}]  |  {n_runs_checked} runs total.")


def build_nc_truth(muon_base_dir: str, verbose: bool = True, omit_runs: set[str] | None = None) -> pd.DataFrame:
    """Load NC truth from Sim 1 across all run_NNN subdirs of muon_base_dir.

    Returns a DataFrame with columns:
        run_id, muon_id, nc_id, nc_time_ns, flag_ge77, nc_x, nc_y, nc_z

    ``run_id`` is the integer extracted from the run directory name
    (e.g. run_001 → 1).  The unique NC key across runs is
    (run_id, muon_id, nc_id).

    Within-run duplicates — the same (muon_id, nc_id) appearing in more
    than one output_t file — raise RuntimeError immediately rather than
    being silently dropped.

    Rows are sorted by (run_id, muon_id, nc_id) and carry a sequential
    integer index (reset_index).  This ordering defines the row mapping
    used by build_pmt_matrix(); pass the same DataFrame to every setup so
    that all B matrices share consistent row indices.
    """
    run_dirs = _list_run_dirs(muon_base_dir, omit_runs=omit_runs)
    if verbose:
        print(f"  Loading NC truth from {len(run_dirs)} run(s) in {muon_base_dir!r}")

    skipped: list[tuple[str, str]] = []
    frames = []
    for rd in run_dirs:
        run_id = _run_id_from_dir(rd)
        try:
            frames.append(_load_nc_from_run_dir(rd, run_id))
        except Exception as exc:
            skipped.append((rd, str(exc)))

    if skipped:
        for rd, err in skipped[:3]:
            print(f"  [WARN] Skipped {rd}: {err}")
        if len(skipped) > 3:
            print(f"  [WARN] ... and {len(skipped) - 3} more skipped.")
    if not frames:
        raise RuntimeError(f"No NC truth loaded from {muon_base_dir!r}")

    nc_truth = pd.concat(frames, ignore_index=True)

    nc_truth = (
        nc_truth
        .sort_values(["run_id", "muon_id", "nc_id"])
        .reset_index(drop=True)
    )

    if verbose:
        n_muons = len(nc_truth[["run_id", "muon_id"]].drop_duplicates())
        n_ge77_ncs = int((nc_truth["flag_ge77"] == 1).sum())
        print(
            f"  NC truth: {len(nc_truth):,} NCs, {n_muons:,} muons, "
            f"{n_ge77_ncs:,} Ge77 NCs."
        )
    return nc_truth


def count_vertices_by_run(base_dir: str, omit_runs: set[str] | None = None) -> dict[str, int]:
    """Return vertex count per run label for all run_NNN subdirs of base_dir.

    Returns dict mapping run label (e.g. "run_001") to total vertex count
    summed across all output_t*.hdf5 files in that run directory.
    Used for cross-setup vertex count validation.
    """
    run_dirs = _list_run_dirs(base_dir, omit_runs=omit_runs)
    result = {}
    for rd in run_dirs:
        run_label = os.path.basename(rd)
        result[run_label] = _count_vertices_run_dir(rd)
    return result


def build_pmt_matrix(
    sim_base_dir: str,
    nc_truth: pd.DataFrame,
    m_threshold: int = 1,
    time_cut_ns: float = TIME_CUT_NC_NS,
    float_tol_ns: float = FLOAT_TOL_NS,
    verbose: bool = True,
    omit_runs: set[str] | None = None,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """Build a sparse NC×PMT binary detection matrix from Sim 2 optical data.

    Each row corresponds to one NC (same order as nc_truth; share the
    exact same DataFrame instance across all setups so row indices align).
    Each column corresponds to one unique det_uid found in this setup's
    optical data, sorted ascending.

    det_uid is used as-is as the PMT column identifier — no additional
    layer filter is applied (same convention as comparePMTCoverage.py).

    Optical hits are matched to NC truth rows by (run_id, muon_track_id,
    nC_track_id) so that IDs from different runs are never conflated.

    B[i, j] = 1  if NC i was detected by PMT j with ≥m_threshold photon
              hits within time_cut_ns of the NC time.

    Parameters
    ----------
    sim_base_dir:
        Directory containing run_NNN subdirectories with Sim 2 optical data.
    nc_truth:
        Shared NC truth DataFrame from build_nc_truth().
    m_threshold:
        Minimum photon hit count per PMT per NC to count as a detection.
    time_cut_ns:
        Photon arrival time window after NC time [ns].
    float_tol_ns:
        Tolerance for small negative dt values (float rounding artefacts).
    verbose:
        Print progress information.

    Returns
    -------
    B : scipy.sparse.csr_matrix, shape (n_nc, n_pmts)
    pmt_uids : np.ndarray[int64], shape (n_pmts,) — det_uid per column
    detect_info : dict with boolean arrays of shape (n_nc,):
        "nc_any_photon"   — NC has ≥1 photon hit at any time (no time cut)
        "nc_within_200ns" — NC has ≥1 photon hit within time_cut_ns
        (nc_only_outside_200ns = nc_any_photon & ~nc_within_200ns)
    """
    run_dirs = _list_run_dirs(sim_base_dir, omit_runs=omit_runs)
    if verbose:
        print(
            f"  Loading optical data from {len(run_dirs)} run(s) "
            f"in {sim_base_dir!r}"
        )

    n_nc = len(nc_truth)

    # ── load all optical data ─────────────────────────────────────────
    opt_frames = []
    for rd in run_dirs:
        run_id = _run_id_from_dir(rd)
        try:
            opt_frames.append(_load_optical_from_run_dir(rd, run_id))
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Skipping {rd}: {exc}")

    if not opt_frames:
        raise RuntimeError(f"No optical data loaded from {sim_base_dir!r}")

    optical = pd.concat(opt_frames, ignore_index=True)
    del opt_frames
    gc.collect()

    if verbose:
        print(f"  Optical rows loaded: {len(optical):,}")

    # Empty result for degenerate case
    _empty_detect = {
        "nc_any_photon":   np.zeros(n_nc, dtype=bool),
        "nc_within_200ns": np.zeros(n_nc, dtype=bool),
    }
    if len(optical) == 0:
        return sp.csr_matrix((n_nc, 0), dtype=np.int8), np.array([], dtype=np.int64), _empty_detect

    # ── merge with NC truth to get nc_time_ns (vectorised) ───────────
    # Key: (run_id, muon_track_id, nC_track_id) — run_id prevents conflating
    # IDs from different runs that happen to share the same numeric values.
    nc_time_lookup = pd.DataFrame({
        "run_id":        nc_truth["run_id"].astype(np.int32),
        "muon_track_id": nc_truth["muon_id"].astype(np.int64),
        "nC_track_id":   nc_truth["nc_id"].astype(np.int64),
        "nc_time_ns":    nc_truth["nc_time_ns"].values,
        "row_idx":       np.arange(n_nc, dtype=np.int32),
    })

    optical["run_id"]         = optical["run_id"].astype(np.int32)
    optical["muon_track_id"]  = optical["muon_track_id"].astype(np.int64)
    optical["nC_track_id"]    = optical["nC_track_id"].astype(np.int64)

    optical = optical.merge(
        nc_time_lookup[["run_id", "muon_track_id", "nC_track_id", "nc_time_ns", "row_idx"]],
        on=["run_id", "muon_track_id", "nC_track_id"],
        how="left",
    )

    matched_mask = ~np.isnan(optical["nc_time_ns"].values)

    # ── detectability: nc_any_photon (no time cut) ────────────────────
    # Any NC that has at least one photon hit on any PMT at any matched time
    matched_rows = optical.loc[matched_mask, "row_idx"].dropna().astype(np.int32).values
    nc_any_photon = np.zeros(n_nc, dtype=bool)
    nc_any_photon[np.unique(matched_rows)] = True

    # ── time cut ──────────────────────────────────────────────────────
    dt = optical["time_in_ns"].values - optical["nc_time_ns"].values
    valid_mask = (
        matched_mask
        & (dt >= float_tol_ns)
        & (dt <= time_cut_ns)
    )

    # ── detectability: nc_within_200ns ────────────────────────────────
    within_rows = optical.loc[valid_mask, "row_idx"].dropna().astype(np.int32).values
    nc_within_200ns = np.zeros(n_nc, dtype=bool)
    nc_within_200ns[np.unique(within_rows)] = True

    filtered = optical.loc[
        valid_mask, ["run_id", "muon_track_id", "nC_track_id", "det_uid", "row_idx"]
    ].copy()
    del optical
    gc.collect()

    if verbose:
        pct = 100.0 * valid_mask.sum() / max(valid_mask.size, 1)
        print(
            f"  Photons within time cut: {valid_mask.sum():,} ({pct:.1f}%)  |  "
            f"nc_any_photon={nc_any_photon.sum():,}  "
            f"nc_within_200ns={nc_within_200ns.sum():,}"
        )

    detect_info = {
        "nc_any_photon":   nc_any_photon,
        "nc_within_200ns": nc_within_200ns,
    }

    if len(filtered) == 0:
        return sp.csr_matrix((n_nc, 0), dtype=np.int8), np.array([], dtype=np.int64), detect_info

    # ── aggregate hits per (NC, PMT) and binarise ────────────────────
    hits = (
        filtered
        .groupby(["run_id", "muon_track_id", "nC_track_id", "det_uid"])
        .size()
        .reset_index(name="n_hits")
    )
    del filtered
    gc.collect()

    firing = hits.loc[hits["n_hits"] >= m_threshold].copy()
    del hits
    gc.collect()

    # ── PMT column mapping ────────────────────────────────────────────
    pmt_uids = np.sort(firing["det_uid"].unique()).astype(np.int64)
    n_pmts = len(pmt_uids)

    # ── map NC keys to row indices (vectorised merge) ─────────────────
    nc_row_lookup = nc_time_lookup[["run_id", "muon_track_id", "nC_track_id", "row_idx"]].copy()
    firing["run_id"]         = firing["run_id"].astype(np.int32)
    firing["muon_track_id"]  = firing["muon_track_id"].astype(np.int64)
    firing["nC_track_id"]    = firing["nC_track_id"].astype(np.int64)

    firing_with_rows = firing.merge(
        nc_row_lookup,
        on=["run_id", "muon_track_id", "nC_track_id"],
        how="inner",  # drops photons whose NC is not in truth (validated elsewhere)
    )

    row_arr = firing_with_rows["row_idx"].values.astype(np.int32)
    uid_arr = firing_with_rows["det_uid"].astype(np.int64).values
    # pmt_uids is sorted → searchsorted gives correct column index
    col_arr = np.searchsorted(pmt_uids, uid_arr).astype(np.int32)

    data = np.ones(len(row_arr), dtype=np.int8)
    B = sp.csr_matrix(
        (data, (row_arr, col_arr)),
        shape=(n_nc, n_pmts),
        dtype=np.int8,
    )

    if verbose:
        mem = (B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1e6
        print(
            f"  B: {n_nc:,} NCs × {n_pmts} PMTs, "
            f"nnz={B.nnz:,}, {mem:.2f} MB"
        )

    return B, pmt_uids, detect_info
