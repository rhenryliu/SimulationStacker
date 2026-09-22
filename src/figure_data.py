"""Opt-in export of the data drawn in a figure, for publication data releases.

Plotting scripts call :func:`record` (or :func:`record_array`) next to each
plotting call, passing the values as drawn but without any purely visual
x-offset, and call :func:`save` straight after ``savefig``.  Every function is a
no-op unless the environment variable ``FIGURE_DATA_DIR`` is set, so ordinary
runs are unaffected.

With ``FIGURE_DATA_DIR`` set, :func:`save` writes ``<figure stem>.csv`` or
``<figure stem>.npz`` into that directory and clears the buffer, so a script
that saves several figures produces one file per figure.  A CSV has one row
per plotted point: ``panel`` and ``series`` columns, then the recorded columns
in the order they were first seen, left empty where a row has no value.

This module is duplicated verbatim in SimulationStacker/src so that neither
repository depends on the other; keep the two copies identical.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

ENV_VAR = "FIGURE_DATA_DIR"

_rows: list[dict[str, Any]] = []
_arrays: dict[str, np.ndarray] = {}


def enabled() -> bool:
    """Return True when figure data should be recorded."""
    return bool(os.environ.get(ENV_VAR))


def plain(label: str) -> str:
    """Convert a Matplotlib TeX legend label to plain text for a series name.

    Args:
        label: Legend label, e.g. ``r'DESI $\\times$ ACT (z$\\sim$0.51)'``.

    Returns:
        The label with ``$\\times$`` as ``x``, ``$\\sim$`` as ``~`` and any
        remaining ``$`` removed, e.g. ``'DESI x ACT (z~0.51)'``.
    """
    label = label.replace(r"$\times$", "x").replace(r"$\sim$", "~")
    return " ".join(label.replace("$", "").split())


def record(panel: str, series: str, **columns: Any) -> None:
    """Buffer one plotted series as CSV rows, one row per point.

    Args:
        panel: Panel title, or '' for a panel without one.
        series: Plain-text series name.
        **columns: Column name mapped to 1-D values.  Arrays must share one
            length; a scalar is repeated on every row and None leaves the
            column empty.

    Raises:
        ValueError: If the non-scalar columns differ in length.
    """
    if not enabled():
        return
    values = {
        name: np.ravel(np.asarray(value, dtype=float))
        for name, value in columns.items()
        if value is not None
    }
    lengths = {array.size for array in values.values() if array.size != 1}
    if len(lengths) > 1:
        raise ValueError(
            f"Columns for series {series!r} have different lengths: {sorted(lengths)}"
        )
    n_rows = lengths.pop() if lengths else 1
    for index in range(n_rows):
        row: dict[str, Any] = {"panel": panel, "series": series}
        for name, array in values.items():
            row[name] = float(array[index] if array.size == n_rows else array[0])
        _rows.append(row)


def record_array(name: str, array: Any) -> None:
    """Buffer a numerical array, such as a correlation matrix, for an NPZ file.

    Args:
        name: Key of the array in the NPZ file.
        array: Numerical array as drawn.
    """
    if not enabled():
        return
    _arrays[name] = np.asarray(array, dtype=float)


def save(figure_path: str | os.PathLike) -> None:
    """Write the buffered data for the figure just saved, then clear the buffer.

    Args:
        figure_path: Path the figure was saved to; only its stem is used.

    Raises:
        ValueError: If both CSV rows and NPZ arrays were buffered.
    """
    if not enabled():
        return
    stem = Path(figure_path).stem
    output_dir = Path(os.environ[ENV_VAR])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if _rows and _arrays:
            raise ValueError(f"Both CSV rows and NPZ arrays were recorded for {stem}")
        if _arrays:
            path = output_dir / f"{stem}.npz"
            np.savez(path, **_arrays)
        elif _rows:
            path = output_dir / f"{stem}.csv"
            header = list(dict.fromkeys(key for row in _rows for key in row))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, restval="")
                writer.writeheader()
                for row in _rows:
                    writer.writerow(
                        {key: repr(value) if isinstance(value, float) else value
                         for key, value in row.items()}
                    )
        else:
            print(f"Figure data: nothing recorded for {stem}")
            return
        print(f"Figure data: wrote {path}")
    finally:
        _rows.clear()
        _arrays.clear()
