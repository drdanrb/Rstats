"""Frequency tables and categorical tests."""

import numpy as np
import pandas as pd
from scipy import stats

from ._data import complete, dataframe, parse
from .result import AnalysisResult


def _variables(formula):
    if isinstance(formula, str) and "~" not in formula:
        return [formula.strip().strip("`")]
    spec = parse(formula)
    if len(spec.predictors) != 1 or any(c in spec.expression for c in "*:+-()"):
        raise ValueError("Use 'row ~ column' for a two-way categorical table.")
    return [spec.outcome, spec.predictors[0]]


def table(formula, *, data, percent="valid", missing="drop"):
    """Counts and percentages: ``table('group ~ passed', data=df, percent='row')``.

    Missing values are excluded and counted separately. ``percent='valid'``
    uses the valid grand total; row/column/total apply to two-way tables.
    """
    if percent not in {"valid", "row", "column", "total"}:
        raise ValueError("percent must be 'valid', 'row', 'column', or 'total'.")
    cols = _variables(formula)
    raw = dataframe(data, cols)
    frame, notes = complete(data, cols, missing)
    if len(cols) == 1:
        if percent not in {"valid", "total"}:
            raise ValueError("One-way tables use percent='valid' or 'total'.")
        counts = frame[cols[0]].value_counts(sort=False).rename("count").to_frame()
        counts = counts.loc[counts["count"] > 0]
        percentages = (100 * counts / len(frame)).rename(columns={"count": "percent"})
        denominator = len(frame)
    else:
        counts = pd.crosstab(frame[cols[0]], frame[cols[1]])
        denominator = (
            counts.sum(axis=1)
            if percent == "row"
            else counts.sum(axis=0)
            if percent == "column"
            else counts.to_numpy().sum()
        )
        percentages = counts.div(denominator, axis=0 if percent == "row" else 1) * 100
    notes.append(
        f"Percentages use {'the valid grand total' if percent in {'valid', 'total'} else percent + ' totals'} as denominator."
    )
    return AnalysisResult(
        "Frequency table",
        {
            "counts": counts,
            "percentages": percentages,
            "missing": raw.isna().sum().rename("missing").to_frame(),
        },
        notes,
        {"percent": percent, "denominator": denominator},
    )


def chisq(formula, *, data=None, expected=None, method="pearson", correction=False, missing="drop"):
    """Pearson independence/GOF test, or explicit Fisher exact for a 2×2 table.

    ``chisq('group ~ passed', data=df)`` tests independence.
    ``chisq([20, 30, 50], expected=[.25, .25, .5])`` tests count proportions.
    """
    if method not in {"pearson", "fisher"}:
        raise ValueError("method must be 'pearson' or 'fisher'.")
    if not isinstance(correction, bool):
        raise TypeError(
            "correction must be True or False (Yates correction for 2×2 Pearson tests)."
        )
    if isinstance(formula, str):
        cols = _variables(formula)
        if len(cols) != 2:
            raise ValueError(
                "For independence use 'group ~ outcome'; for goodness of fit pass a count vector."
            )
        if expected is not None:
            raise ValueError("expected proportions apply only to goodness-of-fit count vectors.")
        frame, notes = complete(data, cols, missing)
        observed = pd.crosstab(frame[cols[0]], frame[cols[1]])
        if min(observed.shape) < 2:
            raise ValueError(
                "An independence test needs at least two observed levels per variable."
            )
        if correction and observed.shape != (2, 2):
            raise ValueError("Yates correction is supported only for 2×2 tables.")
        chi2, p, dof, exp = stats.chi2_contingency(observed, correction=correction)
        residuals = (observed.to_numpy() - exp) / np.sqrt(exp)
        if method == "fisher":
            if observed.shape != (2, 2) or correction:
                raise ValueError("Fisher's test needs a 2×2 table and correction=False.")
            fisher = stats.fisher_exact(observed.to_numpy())
            summary = {"odds_ratio": fisher.statistic, "p": fisher.pvalue, "n": len(frame)}
            title = "Fisher exact test (two-sided)"
        else:
            summary = {
                "chi2": chi2,
                "df": dof,
                "p": p,
                "n": len(frame),
                "cramers_v": np.sqrt(
                    stats.chi2_contingency(observed, correction=False)[0]
                    / (len(frame) * min(np.array(observed.shape) - 1))
                ),
            }
            title = "Pearson chi-square test of independence"
        tables = {
            "test": pd.DataFrame([summary]),
            "observed": observed,
            "expected": pd.DataFrame(exp, index=observed.index, columns=observed.columns),
            "pearson_residuals": pd.DataFrame(
                residuals, index=observed.index, columns=observed.columns
            ),
        }
        notes.append(
            f"Continuity correction: {'Yates' if correction else 'none'}. Expected counts and residuals describe the independence model."
        )
    else:
        if method != "pearson" or correction:
            raise ValueError("Goodness of fit uses method='pearson', correction=False.")
        obs = np.asarray(formula, dtype=float)
        if (
            obs.ndim != 1
            or len(obs) < 2
            or not np.isfinite(obs).all()
            or (obs < 0).any()
            or obs.sum() <= 0
            or not np.equal(obs, np.floor(obs)).all()
        ):
            raise ValueError(
                "Supply at least two nonnegative integer counts with a positive total."
            )
        proportions = (
            np.ones(len(obs)) / len(obs) if expected is None else np.asarray(expected, dtype=float)
        )
        if (
            proportions.shape != obs.shape
            or not np.isfinite(proportions).all()
            or (proportions <= 0).any()
            or not np.isclose(proportions.sum(), 1)
        ):
            raise ValueError(
                "expected must contain positive proportions summing to 1, one per category."
            )
        exp = proportions * obs.sum()
        test = stats.chisquare(obs, exp)
        tables = {
            "test": pd.DataFrame(
                [
                    {
                        "chi2": test.statistic,
                        "df": len(obs) - 1,
                        "p": test.pvalue,
                        "n": int(obs.sum()),
                        "cohens_w": np.sqrt(test.statistic / obs.sum()),
                    }
                ]
            ),
            "counts": pd.DataFrame(
                {"observed": obs, "expected": exp, "pearson_residual": (obs - exp) / np.sqrt(exp)}
            ),
        }
        title, notes = (
            "Chi-square goodness-of-fit test",
            [
                "Expected proportions must be specified independently of these observations; fitted-probability degrees of freedom are not supported."
            ],
        )
    if (exp < 1).any() or (exp < 5).mean() > 0.2:
        notes.append(
            "Sparse expected cells: chi-square asymptotic inference may be unreliable. For a 2×2 table consider method='fisher'."
        )
    notes.append(
        "Independent observations are required; these tests are not for paired categorical measurements."
    )
    return AnalysisResult(title, tables, notes, {"method": method, "correction": correction})
