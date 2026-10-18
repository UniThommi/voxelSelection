"""
Data loading and sparse matrix construction.

Reads HDF5 simulation data, applies stochastic rounding with area ratios,
and constructs sparse binary matrices for greedy optimisation.

Input is the ML-format HDF5 written by simPostProcessing (``target_matrix``
with photon hits per NC and SSD voxel, ``target_columns``, ``voxels/``,
``phi_matrix``, ``event_ids``, ``primaries``). The hits are SSD hits; the
per-area PMT/SSD ratios convert them to PMT-equivalent hits before the
per-voxel threshold m is applied.

Author: Thomas Bürger, Eberhard Karls Universität Tübingen
"""

import time

import h5py
import numpy as np
from scipy import sparse

from .geometry import is_valid_pmt_position
from .ratio_scaling import build_layer_map, build_ratio_vec, stochastic_round_block


def get_valid_voxel_mask(
    f: h5py.File,
    voxel_keys: list[str],
    verbose: bool = True,
) -> np.ndarray:
    """
    Read voxel centre and layer from HDF5 and determine which voxels
    can physically host a PMT.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    voxel_keys : list[str]
        Sorted list of voxel ID strings.
    verbose : bool
        Print filtering statistics.

    Returns
    -------
    valid_mask : np.ndarray of bool, shape (num_voxels,)
        True for voxels where a PMT can be placed.
    """
    from .geometry import PMT_RADIUS

    num_voxels = len(voxel_keys)
    valid_mask = np.zeros(num_voxels, dtype=bool)
    layer_counts = {"pit": [0, 0], "bot": [0, 0], "top": [0, 0], "wall": [0, 0]}

    for col_idx, vkey in enumerate(voxel_keys):
        center = f[f"voxels/{vkey}/center"][:]
        layer_raw = f[f"voxels/{vkey}/layer"][()]
        layer = layer_raw.decode() if isinstance(layer_raw, bytes) else str(layer_raw)

        if layer in layer_counts:
            layer_counts[layer][0] += 1

        if is_valid_pmt_position(center, layer):
            valid_mask[col_idx] = True
            if layer in layer_counts:
                layer_counts[layer][1] += 1

    if verbose:
        total = num_voxels
        valid = int(valid_mask.sum())
        print(f"PMT placement filter (r_pmt = {PMT_RADIUS} mm):")
        for ly in ["pit", "bot", "top", "wall"]:
            tot, val = layer_counts[ly]
            print(f"  {ly:>4}: {val}/{tot} valid")
        print(f"  Total: {valid}/{total} valid "
              f"({total - valid} filtered out)")

    return valid_mask


def load_muon_data(
    filepath: str,
    num_ncs: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load muon-related fields from the phi group in the HDF5 file.

    ``nC_time_in_ns`` and ``nC_flag_Ge77`` come from ``phi_matrix``; the
    muon identity comes from ``event_ids`` (``run_id``, ``muon_id``),
    because ``muon_id`` restarts in every run and is only unique together
    with ``run_id``.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    num_ncs : int
        Expected number of NCs (for consistency check).
    verbose : bool
        Print progress information.

    Returns
    -------
    global_muon_id : np.ndarray, shape (num_ncs,), dtype int64
        Dense muon index 0..n_muons-1 built from unique (run_id, muon_id).
    nc_time_ns : np.ndarray, shape (num_ncs,), dtype float64
        Capture time relative to the muon (ns).
    nc_flag_ge77 : np.ndarray, shape (num_ncs,), dtype bool
        True for the capture that produced Ge77.
    """
    with h5py.File(filepath, "r") as f:
        phi_columns = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["phi_columns"][:]]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}

        phi_matrix = f["phi_matrix"]
        nc_time_ns   = phi_matrix[:, phi_col_idx["nC_time_in_ns"]].astype(np.float64)
        nc_flag_ge77 = phi_matrix[:, phi_col_idx["nC_flag_Ge77"]].astype(bool)

        # Muon IDs live in event_ids.
        # Build a globally unique muon ID from (run_id, muon_id) pairs so that
        # the ID is unique across runs even when muon_id restarts each run.
        event_id_cols = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["event_id_columns"][:]]
        event_ids = f["event_ids"][:]
        run_col  = event_id_cols.index("run_id")
        muon_col = event_id_cols.index("muon_id")
        pairs = np.stack(
            [event_ids[:, run_col], event_ids[:, muon_col]], axis=1
        )
        _, global_muon_id = np.unique(pairs, axis=0, return_inverse=True)
        global_muon_id = global_muon_id.astype(np.int64)

    if len(global_muon_id) != num_ncs:
        raise ValueError(
            f"Muon data length ({len(global_muon_id)}) != num_ncs ({num_ncs})"
        )

    if verbose:
        n_ge77_ncs = int(nc_flag_ge77.sum())
        n_unique_muons = len(np.unique(global_muon_id))
        ge77_muon_ids = np.unique(global_muon_id[nc_flag_ge77])
        print(f"\nMuon data loaded:")
        print(f"  Total NCs: {num_ncs:,}")
        print(f"  Unique muons: {n_unique_muons:,}")
        print(f"  Ge77 NCs: {n_ge77_ncs:,}")
        print(f"  Ge77 muons: {len(ge77_muon_ids):,}")

    return global_muon_id, nc_time_ns, nc_flag_ge77


def count_hdf5_runs(filepath: str) -> int:
    """Return the number of unique run IDs in the HDF5 event_ids dataset."""
    with h5py.File(filepath, "r") as f:
        event_id_cols = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["event_id_columns"][:]]
        run_col = event_id_cols.index("run_id")
        run_ids = f["event_ids"][:, run_col]
    return int(len(np.unique(run_ids)))


def build_muon_index(
    global_muon_id: np.ndarray,
    nc_time_ns: np.ndarray,
    nc_flag_ge77: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build data structures for muon-level optimisation.

    A muon is a Ge77 muon if any of its NCs carries the Ge77 flag (no time
    restriction). An NC is eligible, i.e. counts towards the muon
    multiplicity W, if it belongs to a Ge77 muon and its capture time lies
    in the muon window [MUON_TIME_WINDOW_MIN_NS, MUON_TIME_WINDOW_MAX_NS]
    = [1 µs, 200 µs]. Ge77-flagged NCs inside the window count as well.

    Parameters
    ----------
    global_muon_id : np.ndarray, shape (num_ncs,)
        Muon index per NC (from ``load_muon_data``).
    nc_time_ns : np.ndarray, shape (num_ncs,)
        Capture time per NC (ns).
    nc_flag_ge77 : np.ndarray, shape (num_ncs,)
        Ge77 flag per NC.
    verbose : bool
        Print a summary.

    Returns
    -------
    nc_to_muon_local : np.ndarray, shape (num_ncs,), dtype int32
        Index into the Ge77-muon arrays for eligible NCs, -1 otherwise.
    muon_nc_counts : np.ndarray, shape (num_ge77_muons,), dtype int32
        Number of eligible NCs per Ge77 muon (upper bound for its W count).
    ge77_muon_global_ids : np.ndarray, shape (num_ge77_muons,), dtype int64
        Sorted ``global_muon_id`` values of the Ge77 muons.
    eligible_nc_mask : np.ndarray, shape (num_ncs,), dtype bool
        True for eligible NCs.
    num_ge77_muons : int
        Number of Ge77 muons.
    """
    from .geometry import MUON_TIME_WINDOW_MIN_NS, MUON_TIME_WINDOW_MAX_NS

    num_ncs = len(global_muon_id)

    ge77_muon_global_ids = np.unique(global_muon_id[nc_flag_ge77])
    num_ge77_muons = len(ge77_muon_global_ids)

    in_time_window = (
        (nc_time_ns >= MUON_TIME_WINDOW_MIN_NS)
        & (nc_time_ns <= MUON_TIME_WINDOW_MAX_NS)
    )

    belongs_to_ge77_muon = np.isin(global_muon_id, ge77_muon_global_ids)
    # Every NC inside the muon window counts towards W, including the
    # Ge77-flagged capture itself: the tagger cannot tell which capture
    # produced Ge77, and the PMT-side evaluation
    # (ratio_analysis.coverage_analysis.evaluate_muon) counts it as well.
    eligible_nc_mask = belongs_to_ge77_muon & in_time_window

    nc_to_muon_local = np.full(num_ncs, -1, dtype=np.int32)
    eligible_indices = np.where(eligible_nc_mask)[0]

    eligible_global_ids = global_muon_id[eligible_indices]
    local_ids = np.searchsorted(ge77_muon_global_ids, eligible_global_ids)
    nc_to_muon_local[eligible_indices] = local_ids.astype(np.int32)

    muon_nc_counts = np.bincount(local_ids, minlength=num_ge77_muons).astype(np.int32)

    if verbose:
        n_eligible = int(eligible_nc_mask.sum())
        n_muons_with_any = int((muon_nc_counts >= 1).sum())
        print(f"\nMuon index built:")
        print(f"  Ge77 muons: {num_ge77_muons:,}")
        print(f"  Eligible NCs (in time window): {n_eligible:,}")
        print(f"  Ge77 muons with ≥1 eligible NC: {n_muons_with_any:,}")
        print(f"  Time window: [{MUON_TIME_WINDOW_MIN_NS/1e3:.0f} µs, "
              f"{MUON_TIME_WINDOW_MAX_NS/1e3:.0f} µs]")

    return (nc_to_muon_local, muon_nc_counts, ge77_muon_global_ids,
            eligible_nc_mask, num_ge77_muons)


def load_nc_muon_ids(
    filepath: str,
    num_ncs: int,
    verbose: bool = True,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Load global muon IDs for all NCs (no time filter).
    Used for NC-mode muon-aware weighting.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    num_ncs : int
        Expected number of NCs.
    verbose : bool
        Print info.

    Returns
    -------
    nc_to_muon_local : np.ndarray, shape (num_ncs,), dtype int32
    num_muons : int
    unique_muon_ids : np.ndarray, dtype int64
    """
    with h5py.File(filepath, "r") as f:
        # Muon IDs live in event_ids, not phi_matrix.
        event_id_cols = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in f["event_id_columns"][:]]
        event_ids = f["event_ids"][:]
        run_col  = event_id_cols.index("run_id")
        muon_col = event_id_cols.index("muon_id")
        pairs = np.stack(
            [event_ids[:, run_col], event_ids[:, muon_col]], axis=1
        )
        _, nc_global_muon_id = np.unique(pairs, axis=0, return_inverse=True)
        nc_global_muon_id = nc_global_muon_id.astype(np.int64)

    if len(nc_global_muon_id) != num_ncs:
        raise RuntimeError(
            f"Muon data length ({len(nc_global_muon_id)}) != "
            f"num_ncs ({num_ncs})"
        )

    unique_muon_ids = np.unique(nc_global_muon_id)
    num_muons = len(unique_muon_ids)
    nc_to_muon_local = np.searchsorted(
        unique_muon_ids, nc_global_muon_id
    ).astype(np.int32)

    if verbose:
        print(f"\nMuon IDs loaded (NC mode):")
        print(f"  Unique muons: {num_muons:,}")

    return nc_to_muon_local, num_muons, unique_muon_ids

def load_raw_sparse(
    filepath: str,
    verbose: bool = True,
    skip_validity: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load raw (unscaled) hit values from HDF5 as sparse COO arrays.
    By default only valid voxels (PMT-placeable) are kept; pass
    ``skip_validity=True`` to include all voxels.

    Keeping the raw counts (instead of a binarised matrix) lets callers
    re-binarise with other ratios or seeds (sensitivity analysis,
    evaluation) without reading the HDF5 file again.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    verbose : bool
        Print progress.
    skip_validity : bool
        If True, keep all voxels instead of only those where a PMT fits
        (``geometry.is_valid_pmt_position``).

    Returns
    -------
    raw_rows : np.ndarray, dtype int64
        NC indices of non-zero entries.
    raw_cols : np.ndarray, dtype int32
        Valid-voxel column indices of non-zero entries.
    raw_vals : np.ndarray, dtype int32
        Hit counts (before any scaling).
    voxel_ids : np.ndarray of str
        Voxel ID strings for valid voxels.
    centers : np.ndarray, shape (num_valid, 3)
        Voxel centres.
    layers : np.ndarray of str
        Layer labels per valid voxel.
    num_ncs : int
        Total number of NCs.
    num_primaries : int
        Total number of primaries.
    """
    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )

        if skip_validity:
            valid_keys = voxel_keys
            num_voxels = len(valid_keys)
            if verbose:
                print(f"PMT placement filter skipped: "
                      f"using all {num_voxels} voxels")
        else:
            valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=verbose)
            valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
            num_voxels = len(valid_keys)

        centers = np.empty((num_voxels, 3), dtype=np.float64)
        layers = np.empty(num_voxels, dtype=object)
        for i, vkey in enumerate(valid_keys):
            centers[i] = f[f"voxels/{vkey}/center"][:]
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers[i] = (layer_raw.decode() if isinstance(layer_raw, bytes)
                         else str(layer_raw))

        target_columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        ]
        target_col_to_idx = {c: i for i, c in enumerate(target_columns)}
        valid_col_indices_arr = np.array(
            [target_col_to_idx[k] for k in valid_keys]
        )

        num_ncs = f["target_matrix"].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"Loading raw sparse data: {num_ncs} NCs, "
                  f"{num_voxels} valid voxels")

        BATCH_SIZE = 1000
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        target_dset = f["target_matrix"]
        total_batches = (num_ncs - 1) // BATCH_SIZE + 1
        t_start = time.time()
        t_last = t_start

        for batch_idx, row_start in enumerate(range(0, num_ncs, BATCH_SIZE)):
            row_end = min(row_start + BATCH_SIZE, num_ncs)
            block = target_dset[row_start:row_end, :]
            block_valid = block[:, valid_col_indices_arr]

            nc_idx, col_idx = np.nonzero(block_valid)
            if len(nc_idx) > 0:
                rows_list.append(nc_idx.astype(np.int64) + row_start)
                cols_list.append(col_idx.astype(np.int32))
                vals_list.append(
                    block_valid[nc_idx, col_idx].astype(np.int32)
                )

            t_now = time.time()
            if verbose and (t_now - t_last >= 5.0
                            or batch_idx == total_batches - 1):
                elapsed = t_now - t_start
                frac = (batch_idx + 1) / total_batches
                eta = (elapsed / frac - elapsed) if frac > 0.01 else 0.0
                print(f"  Batch {batch_idx+1}/{total_batches} "
                      f"({frac:.1%}) | "
                      f"elapsed {elapsed:.1f}s | "
                      f"ETA {eta:.1f}s", end="\r")
                t_last = t_now

        elapsed = time.time() - t_start
        if verbose:
            print(f"\n  Raw data loaded in {elapsed:.1f}s")

        if len(rows_list) > 0:
            raw_rows = np.concatenate(rows_list)
            raw_cols = np.concatenate(cols_list)
            raw_vals = np.concatenate(vals_list)
        else:
            raw_rows = np.array([], dtype=np.int64)
            raw_cols = np.array([], dtype=np.int32)
            raw_vals = np.array([], dtype=np.int32)

        if verbose:
            mem_mb = (raw_rows.nbytes + raw_cols.nbytes
                      + raw_vals.nbytes) / 1e6
            print(f"  Non-zero entries: {len(raw_vals):,}")
            print(f"  Sparse RAM: {mem_mb:.1f} MB")

        voxel_ids = np.array(valid_keys)

    return (raw_rows, raw_cols, raw_vals,
            voxel_ids, centers, layers, num_ncs, num_primaries)


def load_invalid_voxel_data(
    filepath: str,
    valid_voxel_ids: set,
    verbose: bool = False,
) -> tuple:
    """Load centers, layers, and total hit sums for geometrically invalid voxels.

    Invalid voxels are those present in the HDF5 but absent from valid_voxel_ids.

    Returns
    -------
    centers : np.ndarray, shape (n_invalid, 3)
    layers  : np.ndarray of str, shape (n_invalid,)
    hits    : np.ndarray, shape (n_invalid,) — summed photon hits over all NCs
    """
    with h5py.File(filepath, "r") as f:
        all_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )
        invalid_keys = [k for k in all_keys if k not in valid_voxel_ids]

        if not invalid_keys:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=object),
                np.empty(0, dtype=np.float64),
            )

        n_invalid = len(invalid_keys)
        centers = np.empty((n_invalid, 3), dtype=np.float64)
        layers = np.empty(n_invalid, dtype=object)
        for i, key in enumerate(invalid_keys):
            centers[i] = f[f"voxels/{key}/center"][:]
            lr = f[f"voxels/{key}/layer"][()]
            layers[i] = lr.decode() if isinstance(lr, bytes) else str(lr)

        col_to_idx = {k: i for i, k in enumerate(all_keys)}
        invalid_col_indices = np.array([col_to_idx[k] for k in invalid_keys], dtype=np.int32)

        num_ncs = f["target_matrix"].shape[0]
        hits = np.zeros(n_invalid, dtype=np.float64)
        BATCH = 2000
        if verbose:
            print(f"  Loading hit sums for {n_invalid} invalid voxels ...")
        for row_start in range(0, num_ncs, BATCH):
            row_end = min(row_start + BATCH, num_ncs)
            block = f["target_matrix"][row_start:row_end, :][:, invalid_col_indices]
            hits += block.sum(axis=0)

    return centers, layers, hits


def binarize_from_raw(
    raw_rows: np.ndarray,
    raw_cols: np.ndarray,
    raw_vals: np.ndarray,
    num_ncs: int,
    num_voxels: int,
    layers: np.ndarray,
    area_ratios: dict[str, float],
    m: int,
    seed: int = 42,
) -> sparse.csc_matrix:
    """
    Apply stochastic rounding to raw hits and binarise.

    Each SSD hit count n is multiplied by the PMT/SSD ratio r of its area
    and rounded stochastically: floor(n·r) plus 1 with probability equal to
    the fractional part. The expected value is exactly n·r, while the result
    stays an integer photon count that can be compared with the per-voxel
    threshold m. ``B[i, j] = 1`` iff the rounded count of NC i on voxel j
    is >= m. The fixed seed makes B reproducible.

    Parameters
    ----------
    raw_rows, raw_cols, raw_vals : np.ndarray
        Sparse COO arrays of raw hit counts.
    num_ncs : int
        Total number of NCs (rows).
    num_voxels : int
        Number of valid voxels (columns).
    layers : np.ndarray of str
        Layer labels per valid voxel.
    area_ratios : dict[str, float]
        PMT/SSD ratios per area (SSD hits are multiplied by these).
    m : int
        Hit threshold for binarisation.
    seed : int
        Random seed for stochastic rounding.

    Returns
    -------
    B : sparse.csc_matrix
        Binary matrix (NCs x valid_voxels).
    """
    from .ratio_scaling import build_ratio_vec

    rng = np.random.default_rng(seed)
    ratio_vec = build_ratio_vec(layers, area_ratios)

    # Adapt SSD hits to PMT-equivalent hits. Ratio convention is PMT/SSD
    # (ratio = n_PMT / n_SSD), so MULTIPLY each entry by its column's ratio:
    # n_PMT = n_SSD * ratio.
    ratios_per_entry = ratio_vec[raw_cols]
    scaled = raw_vals.astype(np.float64) * ratios_per_entry

    # Stochastic rounding
    floor_vals = np.floor(scaled).astype(np.int32)
    fractional = scaled - floor_vals
    coin = rng.random(len(fractional)) < fractional
    rounded = floor_vals + coin.astype(np.int32)

    # Binarise: keep only entries where rounded >= m
    keep = rounded >= m
    b_rows = raw_rows[keep]
    b_cols = raw_cols[keep]
    b_data = np.ones(keep.sum(), dtype=np.int8)

    B = sparse.coo_matrix(
        (b_data, (b_rows, b_cols)),
        shape=(num_ncs, num_voxels),
        dtype=np.int8,
    ).tocsc()

    return B