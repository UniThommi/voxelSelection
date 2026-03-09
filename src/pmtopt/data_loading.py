"""
Data loading and sparse matrix construction.

Reads HDF5 simulation data, applies stochastic rounding with area ratios,
and constructs sparse binary matrices for greedy optimization.
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
    Read voxel center and layer from HDF5 and determine which voxels
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


def load_and_binarize(
    filepath: str,
    m: int,
    area_ratios: dict[str, float],
    seed: int = 42,
    verbose: bool = True,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load HDF5 data, apply stochastic rounding with area ratios,
    and construct sparse binary matrix B.

    B[i, j] = 1 iff the stochastically rounded hit count for voxel j
    and NC i is >= m.

    Only voxels where a PMT can physically be placed are included.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    m : int
        Minimum number of (rounded) hits per voxel for detection.
    area_ratios : dict[str, float]
        SSD/PMT ratios per area.
    seed : int
        Random seed for stochastic rounding.
    verbose : bool
        Print progress information.

    Returns
    -------
    B : sparse.csc_matrix
        Binary (NCs x valid_voxels) matrix in CSC format.
    voxel_ids : np.ndarray
        Array of voxel ID strings, mapping column index -> voxel name.
    centers : np.ndarray, shape (num_valid_voxels, 3)
        Voxel center coordinates (x, y, z) in mm.
    layers : np.ndarray of str, shape (num_valid_voxels,)
        Layer label per valid voxel.
    num_primaries : int
        Total number of primary events.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(filepath, "r") as f:
        voxel_keys = sorted(
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        )

        # PMT placement filter
        valid_mask = get_valid_voxel_mask(f, voxel_keys, verbose=verbose)
        valid_keys = [k for k, v in zip(voxel_keys, valid_mask) if v]
        num_voxels = len(valid_keys)

        # Read centers and layers for valid voxels
        centers = np.empty((num_voxels, 3), dtype=np.float64)
        layers = np.empty(num_voxels, dtype=object)
        for i, vkey in enumerate(valid_keys):
            centers[i] = f[f"voxels/{vkey}/center"][:]
            layer_raw = f[f"voxels/{vkey}/layer"][()]
            layers[i] = (layer_raw.decode() if isinstance(layer_raw, bytes)
                         else str(layer_raw))

        # Map valid voxel keys to matrix columns
        target_columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["target_columns"][:]
        ]
        target_col_to_idx = {c: i for i, c in enumerate(target_columns)}
        valid_col_indices_arr = np.array(
            [target_col_to_idx[k] for k in valid_keys]
        )

        # Build ratio vector for valid voxels only
        ratio_vec = np.array(
            [area_ratios.get(layers[c], 1.0) for c in range(num_voxels)],
            dtype=np.float64,
        )

        num_ncs = f["target_matrix"].shape[0]
        num_primaries = int(f["primaries"][()])

        if verbose:
            print(f"Loading data: {num_ncs} NCs, {num_voxels} valid voxels, "
                  f"{num_primaries} primaries")
            print(f"Binarization threshold m = {m}")
            print(f"Area ratios: {area_ratios}")
            print(f"Stochastic rounding seed: {seed}")

        # Row-block reading — aligned to chunk layout (1000, 9583)
        BATCH_SIZE = 1000
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []

        target_dset = f["target_matrix"]
        total_batches = (num_ncs - 1) // BATCH_SIZE + 1
        t_load_start = time.time()
        t_last_report = t_load_start

        for batch_idx, row_start in enumerate(range(0, num_ncs, BATCH_SIZE)):
            row_end = min(row_start + BATCH_SIZE, num_ncs)

            # Read full chunk row slice, then extract valid columns
            block = target_dset[row_start:row_end, :]
            block_valid = block[:, valid_col_indices_arr]

            # Stochastic rounding + binarization
            rounded = stochastic_round_block(block_valid, ratio_vec, rng)
            mask = rounded >= m

            nc_idx, col_idx = np.nonzero(mask)
            if len(nc_idx) > 0:
                rows_list.append(nc_idx.astype(np.int64) + row_start)
                cols_list.append(col_idx.astype(np.int32))

            # Progress report every ~5 seconds
            t_now = time.time()
            if verbose and (t_now - t_last_report >= 5.0
                            or batch_idx == total_batches - 1):
                elapsed = t_now - t_load_start
                frac = (batch_idx + 1) / total_batches
                eta = (elapsed / frac - elapsed) if frac > 0.01 else 0.0
                rows_per_sec = row_end / elapsed if elapsed > 0 else 0.0
                print(f"  Batch {batch_idx+1}/{total_batches} "
                      f"({frac:.1%}) | "
                      f"{rows_per_sec:,.0f} rows/s | "
                      f"elapsed {elapsed:.1f}s | "
                      f"ETA {eta:.1f}s", end="\r")
                t_last_report = t_now

        t_load_elapsed = time.time() - t_load_start
        if verbose:
            print(f"\n  Target matrix loaded and binarized in "
                  f"{t_load_elapsed:.1f}s "
                  f"({num_ncs / t_load_elapsed:,.0f} rows/s)")

        if len(rows_list) > 0:
            all_rows = np.concatenate(rows_list)
            all_cols = np.concatenate(cols_list)
        else:
            all_rows = np.array([], dtype=np.int64)
            all_cols = np.array([], dtype=np.int32)
        all_data = np.ones(len(all_rows), dtype=np.int8)

        B = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(num_ncs, num_voxels),
            dtype=np.int8,
        ).tocsc()

        nnz = B.nnz
        density = nnz / (num_ncs * num_voxels) * 100 if num_ncs * num_voxels > 0 else 0
        mem_mb = (B.data.nbytes + B.indices.nbytes + B.indptr.nbytes) / 1e6

        if verbose:
            print(f"Sparse matrix: {num_ncs} x {num_voxels}, "
                  f"nnz = {nnz:,} ({density:.3f}%), "
                  f"memory = {mem_mb:.1f} MB")

        voxel_ids = np.array(valid_keys)

    return B, voxel_ids, centers, layers, num_primaries


def load_muon_data(
    filepath: str,
    num_ncs: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load muon-related fields from the phi group in the HDF5 file.

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
    nc_time_ns : np.ndarray, shape (num_ncs,), dtype float64
    nc_flag_ge77 : np.ndarray, shape (num_ncs,), dtype bool
    """
    with h5py.File(filepath, "r") as f:
        phi_columns = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["phi_columns"][:]]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}

        phi_matrix = f["phi_matrix"]
        global_muon_id = phi_matrix[:, phi_col_idx["global_muon_id"]].astype(np.int64)
        nc_time_ns = phi_matrix[:, phi_col_idx["nC_time_in_ns"]].astype(np.float64)
        nc_flag_ge77 = phi_matrix[:, phi_col_idx["nC_flag_Ge77"]].astype(bool)

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


def build_muon_index(
    global_muon_id: np.ndarray,
    nc_time_ns: np.ndarray,
    nc_flag_ge77: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build data structures for muon-level optimization.

    Parameters
    ----------
    global_muon_id : np.ndarray, shape (num_ncs,)
    nc_time_ns : np.ndarray, shape (num_ncs,)
    nc_flag_ge77 : np.ndarray, shape (num_ncs,)
    verbose : bool

    Returns
    -------
    nc_to_muon_local : np.ndarray, shape (num_ncs,), dtype int32
    muon_nc_counts : np.ndarray, shape (num_ge77_muons,), dtype int32
    ge77_muon_global_ids : np.ndarray, shape (num_ge77_muons,), dtype int64
    eligible_nc_mask : np.ndarray, shape (num_ncs,), dtype bool
    num_ge77_muons : int
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
    eligible_nc_mask = belongs_to_ge77_muon & in_time_window & (~nc_flag_ge77)

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
        print(f"  Eligible NCs (non-Ge77, in time window): {n_eligible:,}")
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
        phi_columns = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in f["phi_columns"][:]]
        phi_col_idx = {name: i for i, name in enumerate(phi_columns)}
        phi_matrix = f["phi_matrix"]
        nc_global_muon_id = phi_matrix[:, phi_col_idx["global_muon_id"]].astype(np.int64)

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