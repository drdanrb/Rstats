"""Detailed descriptive statistics without model fitting."""

import numpy as np
import pandas as pd
from scipy import stats

from ._data import alpha_value, dataframe, names
from .result import AnalysisResult


def descriptives(*, data, columns=None, by=None, alpha=0.05):
    """Summarize selected columns, optionally within groups.

    Missingness is per variable. SD/variance use n-1, skewness is bias-corrected,
    and kurtosis is bias-corrected excess kurtosis. Mean CIs use Student's t.
    """
    alpha_value(alpha)
    dataframe(data, [])  # Validate the container before reading its columns.
    groups = names(by)
    columns = (
        names(columns) if columns is not None else [c for c in data.columns if c not in groups]
    )
    if not columns:
        raise ValueError("Select at least one column to summarize.")
    if set(groups) & set(columns):
        raise ValueError("Grouping columns must not also be summarized columns.")
    frame = dataframe(data, [*groups, *columns])
    numeric_rows, frequency_rows = [], []
    chunks = (
        frame.groupby(groups, observed=True, dropna=False, sort=False) if groups else [((), frame)]
    )
    for key, chunk in chunks:
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(groups, key))
        for col in columns:
            values = chunk[col].dropna()
            n = len(values)
            base = {**labels, "variable": col, "n": n, "missing": int(chunk[col].isna().sum())}
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                if np.isinf(values.to_numpy(dtype=float)).any():
                    raise ValueError(f"{col!r} contains infinite values.")
                x = values.to_numpy(dtype=float)
                sd = np.std(x, ddof=1) if n > 1 else np.nan
                mean = np.mean(x) if n else np.nan
                margin = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
                q = np.quantile(x, [0.25, 0.5, 0.75]) if n else [np.nan] * 3
                numeric_rows.append(
                    {
                        **base,
                        "mean": mean,
                        "median": q[1],
                        "modes": ", ".join(map(str, values.mode().tolist())),
                        "sd": sd,
                        "variance": sd**2,
                        "min": np.min(x) if n else np.nan,
                        "max": np.max(x) if n else np.nan,
                        "range": np.ptp(x) if n else np.nan,
                        "q1": q[0],
                        "q3": q[2],
                        "iqr": q[2] - q[0],
                        "mad": stats.median_abs_deviation(x) if n else np.nan,
                        "skewness": stats.skew(x, bias=False) if n > 2 and sd > 0 else np.nan,
                        "excess_kurtosis": stats.kurtosis(x, bias=False)
                        if n > 3 and sd > 0
                        else np.nan,
                        "mean_ci_low": mean - margin,
                        "mean_ci_high": mean + margin,
                    }
                )
            else:
                counts = values.value_counts(sort=False, dropna=True)
                for level, count in counts.items():
                    if count:
                        frequency_rows.append(
                            {
                                **base,
                                "level": level,
                                "count": int(count),
                                "percent": 100 * count / n,
                            }
                        )
                if not n:
                    frequency_rows.append(
                        {**base, "level": "(no valid observations)", "count": 0, "percent": np.nan}
                    )
    tables = {}
    if numeric_rows:
        tables["descriptives"] = pd.DataFrame(numeric_rows)
    if frequency_rows:
        tables["frequencies"] = pd.DataFrame(frequency_rows)
    return AnalysisResult(
        "Descriptive statistics",
        tables,
        [
            "Available values are used per variable; missing grouping values form a separate group.",
            f"Sample SD/variance use n−1. Mean CIs are {100 * (1 - alpha):g}% t intervals; MAD is unscaled and kurtosis is excess kurtosis.",
            "Modes list all tied most-frequent values. Descriptives do not establish causal effects.",
        ],
        {"alpha": alpha, "by": groups},
    )
