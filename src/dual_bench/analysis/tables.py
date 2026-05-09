# -*- coding: utf-8 -*-
"""LaTeX booktabs table generation for optimization results."""
from __future__ import annotations

import math


def to_latex(
    data: list[dict], columns: list[str], caption: str = "", label: str = "",
    float_fmt: str = ".4f", highlight_best: bool = True,
) -> str:
    """Convert tabular data to a LaTeX booktabs table string.

    When *highlight_best* is ``True`` the minimum value in each numeric
    column is wrapped in ``\\textbf``.
    """
    col_best: dict[str, float] = {}
    if highlight_best:
        for col in columns:
            nums = [r.get(col) for r in data
                    if isinstance(r.get(col), (int, float))
                    and not math.isnan(r.get(col))]
            if nums:
                col_best[col] = min(nums)

    col_fmt = "l" + "c" * (len(columns) - 1)
    header = " & ".join(columns) + " \\\\"

    rows_lines: list[str] = []
    for row in data:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    cell = "--"
                else:
                    cell = f"{val:{float_fmt}}"
                    if col in col_best and val == col_best[col]:
                        cell = f"\\textbf{{{cell}}}"
            else:
                cell = str(val)
            cells.append(cell)
        rows_lines.append(" & ".join(cells) + " \\\\")

    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [
        "\\small", f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule",
        header, "\\midrule", *rows_lines,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def summary_table(
    results: list[dict], metric: str = "hv_mean",
    rows: str = "formulation", cols: str = "algorithm",
) -> str:
    """Create a pivot-style LaTeX summary table from result dicts.

    *rows*/*cols* specify the grouping keys; *metric* is the cell value.
    Minimum per column is bolded.
    """
    row_labels: list[str] = []
    col_labels: list[str] = []
    seen_r: set[str] = set()
    seen_c: set[str] = set()
    for r in results:
        rv, cv = str(r[rows]), str(r[cols])
        if rv not in seen_r:
            row_labels.append(rv); seen_r.add(rv)
        if cv not in seen_c:
            col_labels.append(cv); seen_c.add(cv)

    lookup = {(str(r[rows]), str(r[cols])): r.get(metric, float("nan")) for r in results}

    col_best: dict[str, float] = {}
    for c in col_labels:
        nums = [lookup.get((rv, c), float("nan")) for rv in row_labels]
        nums = [v for v in nums if isinstance(v, (int, float)) and not math.isnan(v)]
        if nums:
            col_best[c] = min(nums)

    col_fmt = "l" + "c" * len(col_labels)
    header = " & ".join([rows] + col_labels) + " \\\\"
    body: list[str] = []
    for rv in row_labels:
        cells = [rv]
        for cv in col_labels:
            val = lookup.get((rv, cv), float("nan"))
            if isinstance(val, float) and math.isnan(val):
                cell = "--"
            else:
                cell = f"{val:.4f}"
                if cv in col_best and val == col_best[cv]:
                    cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        body.append(" & ".join(cells) + " \\\\")

    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\small",
        f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule",
        header, "\\midrule", *body,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)
