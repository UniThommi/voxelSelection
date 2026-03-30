"""PMT position data loading and UID mapping."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class PMTInfo:
    """Single PMT information."""
    index: str
    center: np.ndarray
    layer: str

    @property
    def r(self) -> float:
        return np.sqrt(self.center[0]**2 + self.center[1]**2)

    @property
    def z(self) -> float:
        return self.center[2]

    @property
    def phi(self) -> float:
        return np.arctan2(self.center[1], self.center[0])


def load_pmt_data(json_path: Path) -> List[PMTInfo]:
    """Load PMT positions from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    pmts = []
    for entry in data:
        pmts.append(PMTInfo(
            index=entry['index'],
            center=np.array(entry['center']),
            layer=entry['layer'].lower()
        ))
    return pmts


def get_pmts_by_layer(pmts: List[PMTInfo]) -> Dict[str, List[PMTInfo]]:
    """Group PMTs by layer."""
    grouped: Dict[str, List[PMTInfo]] = {'pit': [], 'bot': [], 'top': [], 'wall': []}
    for pmt in pmts:
        if pmt.layer in grouped:
            grouped[pmt.layer].append(pmt)
    return grouped


def pmt_index_to_uid(index: str) -> int:
    """
    Convert PMT JSON index to detector UID.

    Normal:   '10' + index → 10XXXXXX (8 digits)
    Overflow: '1' + index  → 1XXXXXXX (8 digits, wall PMTs where '10'+index > 8 digits)

    Both produce 8-digit UIDs. Overflow is detected when '10'+index exceeds 8 digits.
    """
    uid_normal = '10' + index
    if len(uid_normal) == 8:
        return int(uid_normal)
    uid_overflow = '1' + index
    if len(uid_overflow) == 8:
        return int(uid_overflow)
    raise ValueError(f"Cannot convert PMT index '{index}' to 8-digit UID")


def uid_to_pmt_index(det_uid: int) -> str:
    """
    Convert detector UID to PMT JSON index.

    Normal:   10XXXXXX → strip '10' → 6-digit index
    Overflow: 1XXXXXXX (2nd digit != '0') → strip '1' → 7-digit index
    """
    s = str(det_uid)
    if len(s) != 8:
        return ''
    if s[:2] == '10':
        return s[2:]
    elif s[0] == '1' and s[1] != '0':
        return s[1:]
    return ''


def build_uid_to_pmt_map(pmts: List[PMTInfo]) -> Dict[int, PMTInfo]:
    """Build mapping from detector UID to PMT info using forward mapping."""
    uid_map: Dict[int, PMTInfo] = {}
    for pmt in pmts:
        uid = pmt_index_to_uid(pmt.index)
        if uid in uid_map:
            print(f"  ⚠️  Duplicate UID {uid} for indices {uid_map[uid].index} and {pmt.index}")
        uid_map[uid] = pmt
    return uid_map


def crosscheck_uids(
    uid_to_pmt: Dict[int, PMTInfo],
    observed_uids: set,
    setup_name: str
) -> None:
    """Crosscheck: all observed PMT UIDs should map to a JSON entry and vice versa."""
    json_uids = set(uid_to_pmt.keys())
    in_data_not_json = observed_uids - json_uids
    in_json_not_data = json_uids - observed_uids

    print(f"\n  UID Crosscheck ({setup_name}):")
    print(f"    JSON PMTs: {len(json_uids)}, Observed UIDs: {len(observed_uids)}")

    if in_data_not_json:
        print(f"    ⚠️  {len(in_data_not_json)} UIDs in data but NOT in JSON: "
              f"{sorted(in_data_not_json)[:10]}...")
    if in_json_not_data:
        print(f"    ℹ️  {len(in_json_not_data)} PMTs in JSON but not observed in data "
              f"(normal if no photons hit them)")
    if not in_data_not_json:
        print(f"    ✅ All observed UIDs found in JSON.")
