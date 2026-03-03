"""Shared utilities for arxiv-compatible notebook results output.

Usage in notebooks:
    from results_utils import setup_results, save_figure, save_table, save_data

    RESULTS = setup_results("00_mc_thermalization_validation")
    # ... create fig ...
    save_figure(fig, "thermalization_curves", RESULTS)
    save_table({"beta": [1,2], "plaq": [0.24, 0.43]}, "mc_vs_exact", RESULTS)
    save_data({"beta": betas, "mc": mc_vals}, "raw_measurements", RESULTS)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# ArXiv-compatible matplotlib defaults
# ---------------------------------------------------------------------------
ARXIV_RC = {
    # Type 1 / OpenType fonts (no bitmap Type 3)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,  # don't require LaTeX install; mathtext is fine
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    # Sizes suitable for single-column (3.4 in) or double-column (7 in)
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Line/marker defaults
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # Clean style
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # High-quality output
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
}

# Standard figure widths for arxiv papers (inches)
FIG_WIDTH_SINGLE = 3.4   # single-column
FIG_WIDTH_DOUBLE = 7.0   # double-column / full-width
FIG_WIDTH_DEFAULT = 7.0


def apply_arxiv_style() -> None:
    """Apply arxiv-compatible matplotlib rc settings."""
    mpl.rcParams.update(ARXIV_RC)


def setup_results(notebook_name: str) -> Path:
    """Set up results directory and apply arxiv plotting style.

    Args:
        notebook_name: Notebook stem (without .ipynb), e.g.
            "00_mc_thermalization_validation".

    Returns:
        Path to the results directory for this notebook.
    """
    apply_arxiv_style()

    results_dir = Path(__file__).parent / "results" / notebook_name
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "data").mkdir(parents=True, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"  figures/  tables/  data/")
    return results_dir


# ---------------------------------------------------------------------------
# Figure saving — PDF (arxiv primary) + PNG (notebook preview)
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    name: str,
    results_dir: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
    close: bool = False,
) -> dict[str, Path]:
    """Save figure in arxiv-compatible formats.

    Saves to ``results_dir/figures/{name}.{ext}`` for each format.

    Args:
        fig: Matplotlib figure.
        name: Filename stem (no extension).
        results_dir: Base results directory for this notebook.
        formats: Output formats. Default: PDF (arxiv) + PNG (preview).
        dpi: Resolution for raster formats.
        close: Whether to close the figure after saving.

    Returns:
        Dict mapping format to saved path.
    """
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for fmt in formats:
        path = fig_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        paths[fmt] = path
        print(f"  Saved: {path.relative_to(results_dir.parent)}")

    if close:
        plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# Table saving — LaTeX .tex (arxiv) + CSV (data)
# ---------------------------------------------------------------------------

def save_table(
    data: dict[str, list] | list[dict],
    name: str,
    results_dir: Path,
    *,
    caption: str = "",
    label: str = "",
    fmt: str = ".4f",
) -> dict[str, Path]:
    r"""Save table as LaTeX .tex and CSV.

    The .tex file contains a standalone ``tabular`` environment
    (no ``\begin{table}`` wrapper) so it can be ``\input{}``-ed
    into a paper's table float.

    Args:
        data: Column-oriented dict {col_name: [values]} or
            row-oriented list of dicts [{col: val}, ...].
        name: Filename stem.
        results_dir: Base results directory.
        caption: Optional caption comment in the .tex file.
        label: Optional LaTeX label.
        fmt: Default numeric format string.

    Returns:
        Dict mapping format to saved path.
    """
    tbl_dir = results_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to column-oriented dict
    if isinstance(data, list):
        cols: dict[str, list] = {}
        for row in data:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
        data = cols

    headers = list(data.keys())
    n_rows = len(next(iter(data.values())))
    rows = []
    for i in range(n_rows):
        row = []
        for h in headers:
            v = data[h][i]
            if isinstance(v, float):
                row.append(f"{v:{fmt}}")
            else:
                row.append(str(v))
        rows.append(row)

    paths = {}

    # --- CSV ---
    csv_path = tbl_dir / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    paths["csv"] = csv_path

    # --- LaTeX tabular ---
    tex_path = tbl_dir / f"{name}.tex"
    col_spec = " ".join(["r"] * len(headers))
    lines = []
    if caption:
        lines.append(f"% {caption}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_tex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if label:
        lines.append(f"% \\label{{{label}}}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    paths["tex"] = tex_path

    for fmt_key, p in paths.items():
        print(f"  Saved: {p.relative_to(results_dir.parent)}")

    return paths


# ---------------------------------------------------------------------------
# Raw data saving — CSV + JSON
# ---------------------------------------------------------------------------

def save_data(
    data: dict[str, Any],
    name: str,
    results_dir: Path,
    *,
    fmt: str = ".6f",
) -> dict[str, Path]:
    """Save raw numerical data as CSV and JSON.

    Args:
        data: Column-oriented dict. Values can be lists, numpy arrays,
            or torch tensors.
        name: Filename stem.
        results_dir: Base results directory.
        fmt: Float format for CSV.

    Returns:
        Dict mapping format to saved path.
    """
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Convert arrays/tensors to lists
    clean = {}
    for k, v in data.items():
        if hasattr(v, "numpy"):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            v = v.tolist()
        clean[k] = v

    paths = {}

    # JSON (full precision)
    json_path = data_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=_json_default)
    paths["json"] = json_path

    # CSV (if all values are equal-length lists)
    values = list(clean.values())
    if all(isinstance(v, list) for v in values):
        lengths = [len(v) for v in values]
        if len(set(lengths)) == 1:
            csv_path = data_dir / f"{name}.csv"
            headers = list(clean.keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for i in range(lengths[0]):
                    row = []
                    for h in headers:
                        val = clean[h][i]
                        if isinstance(val, float):
                            row.append(f"{val:{fmt}}")
                        else:
                            row.append(str(val))
                    writer.writerow(row)
            paths["csv"] = csv_path

    for fmt_key, p in paths.items():
        print(f"  Saved: {p.relative_to(results_dir.parent)}")

    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters in table headers."""
    for char in ("_", "%", "&", "#", "$"):
        s = s.replace(char, f"\\{char}")
    return s


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
