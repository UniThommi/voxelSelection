"""Export voxel-selection JSON files as LaTeX longtables.

Takes the same JSON files accepted by ``pmtopt.main plot`` (either the
greedy output dict with a ``selected_voxels`` list, or a plain list of
voxel records) and writes one ``<stem>_table.tex`` per input file. Each
table lists x/y/z position (mm, one decimal) and the capitalized layer
name, in the order the voxels appear in the JSON (selection order for
greedy outputs).

The emitted longtable uses ``booktabs``, ``longtable`` and ``siunitx``
(S columns), so the including document needs::

    \\usepackage{booktabs}
    \\usepackage{longtable}
    \\usepackage{siunitx}

Usage:
    python src/latex_export/voxel_table.py greedy_N300.json [more.json ...] \\
        [--output-dir DIR]

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import sys
from pathlib import Path

# Make src/pmtopt importable when running as a standalone script.
_pmtopt_src = Path(__file__).resolve().parents[1]
if _pmtopt_src.is_dir() and str(_pmtopt_src) not in sys.path:
    sys.path.insert(0, str(_pmtopt_src))

from pmtopt.main import _parse_selection_json


def _escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]
    for char, repl in replacements:
        text = text.replace(char, repl)
    return text


def _table_format(voxels: list[dict]) -> str:
    """siunitx table-format for the coordinate columns, e.g. '-5.1'."""
    coords = [c for v in voxels for c in v["center"]]
    int_digits = max(len(f"{abs(c):.1f}".split(".")[0]) for c in coords)
    sign = "-" if any(c < 0 for c in coords) else ""
    return f"{sign}{int_digits}.1"


def render_longtable(voxels: list[dict], caption: str, label: str) -> str:
    """Render voxel records as a complete LaTeX longtable string."""
    fmt = _table_format(voxels)
    col_spec = f"S[table-format={fmt}] " * 3 + "l"
    header = r"{$x$ [mm]} & {$y$ [mm]} & {$z$ [mm]} & Layer \\"

    lines = [
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"  \\caption{{{caption}}}\\label{{{label}}}\\\\",
        "  \\toprule",
        f"  {header}",
        "  \\midrule",
        "  \\endfirsthead",
        f"  \\caption[]{{{caption} (continued)}}\\\\",
        "  \\toprule",
        f"  {header}",
        "  \\midrule",
        "  \\endhead",
        "  \\midrule",
        r"  \multicolumn{4}{r}{\emph{Continued on next page}}\\",
        "  \\endfoot",
        "  \\bottomrule",
        "  \\endlastfoot",
    ]
    for v in voxels:
        x, y, z = v["center"]
        layer = _escape_latex(str(v["layer"]).capitalize())
        lines.append(f"  {x:.1f} & {y:.1f} & {z:.1f} & {layer} \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write voxel-selection JSON files as LaTeX longtables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_files", nargs="+", metavar="JSON",
        help="Voxel selection JSON file(s), as accepted by 'pmtopt plot'.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for the .tex files (default: next to each JSON).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for json_path_str in args.json_files:
        json_path = Path(json_path_str)
        if not json_path.exists():
            print(f"ERROR: file not found: {json_path}", file=sys.stderr)
            continue

        with open(json_path) as jf:
            data = json.load(jf)
        voxels, _cfg, _eff = _parse_selection_json(data)
        if not voxels:
            print(f"WARNING: no selected_voxels in {json_path.name}, "
                  f"skipping.", file=sys.stderr)
            continue

        stem = json_path.stem
        caption = (f"Selected PMT positions "
                   f"({_escape_latex(stem)}, $N={len(voxels)}$)")
        label = f"tab:{stem}"
        tex = render_longtable(voxels, caption, label)

        out_path = (out_dir or json_path.parent) / f"{stem}_table.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"{json_path.name}: {len(voxels)} voxels -> {out_path}")


if __name__ == "__main__":
    main()
