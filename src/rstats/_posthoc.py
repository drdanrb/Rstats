"""Omnibus-gated comparisons with explicit test and adjustment families."""

from itertools import combinations, product

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ._data import levels

ADJUSTMENTS = {
    "holm": "holm",
    "bonferroni": "bonferroni",
    "sidak": "sidak",
    "holm-sidak": "holm-sidak",
    "fdr-bh": "fdr_bh",
    "fdr-by": "fdr_by",
}
SPECIAL = {"tukey", "games-howell", "dunnett", "scheffe"}


def method_name(value):
    if value is False:
        return None
    if value is True:
        return "tukey"
    if not isinstance(value, str) or value.lower() not in ADJUSTMENTS.keys() | SPECIAL:
        raise ValueError(
            "posthocifsig must be False, True (Tukey), or one of: "
            + ", ".join(sorted(ADJUSTMENTS.keys() | SPECIAL))
        )
    return value.lower()


def validate_method(method, *, factors, covariates, repeated, control, frame):
    if method is None:
        if control is not None:
            raise ValueError("control= is used only with posthocifsig='dunnett'.")
        return
    if method in SPECIAL and (len(factors) != 1 or covariates or repeated):
        raise ValueError(
            f"{method!r} is supported for one independent grouping factor without covariates. "
            "Use posthocifsig='holm' for model-adjusted or repeated/mixed comparisons."
        )
    if method == "dunnett":
        if control is None or control not in levels(frame[factors[0]]):
            raise ValueError("Dunnett needs control= set to an observed group level.")
    elif control is not None:
        raise ValueError("control= applies only to posthocifsig='dunnett'.")
    if method == "games-howell" and frame.groupby(factors[0], observed=True).size().min() < 2:
        raise ValueError("Games–Howell needs at least two observations in each group.")


def _special(method, frame, outcome, factor, model, alpha, control):
    lev = levels(frame[factor])
    samples = [frame.loc[frame[factor] == value, outcome].to_numpy(dtype=float) for value in lev]
    if method == "games-howell" and any(np.var(x, ddof=1) <= 0 for x in samples):
        raise ValueError("Games–Howell needs positive variance in each group.")
    rows = []
    if method in {"tukey", "games-howell"}:
        result = stats.tukey_hsd(*samples, equal_var=method == "tukey")
        ci = result.confidence_interval(1 - alpha)
        for i, j in combinations(range(len(lev)), 2):
            rows.append(
                {
                    "group_a": lev[i],
                    "group_b": lev[j],
                    "estimate": result.statistic[i, j],
                    "p_adjusted": result.pvalue[i, j],
                    "ci_low": ci.low[i, j],
                    "ci_high": ci.high[i, j],
                }
            )
    elif method == "dunnett":
        index = lev.index(control)
        treatment_indices = [i for i in range(len(lev)) if i != index]
        result = stats.dunnett(
            *[samples[i] for i in treatment_indices],
            control=samples[index],
            rng=np.random.default_rng(0),
        )
        ci = result.confidence_interval(1 - alpha)
        for k, i in enumerate(treatment_indices):
            rows.append(
                {
                    "group_a": lev[i],
                    "group_b": control,
                    "estimate": np.mean(samples[i]) - np.mean(samples[index]),
                    "statistic": result.statistic[k],
                    "p_adjusted": result.pvalue[k],
                    "ci_low": ci.low[k],
                    "ci_high": ci.high[k],
                }
            )
    elif method == "scheffe":
        rank = len(lev) - 1
        critical = np.sqrt(rank * stats.f.ppf(1 - alpha, rank, model.df_resid))
        for i, j in combinations(range(len(lev)), 2):
            estimate = np.mean(samples[i]) - np.mean(samples[j])
            se = np.sqrt(model.mse_resid * (1 / len(samples[i]) + 1 / len(samples[j])))
            t = estimate / se
            rows.append(
                {
                    "group_a": lev[i],
                    "group_b": lev[j],
                    "estimate": estimate,
                    "std_error": se,
                    "df": model.df_resid,
                    "p_raw": 2 * stats.t.sf(abs(t), model.df_resid),
                    "p_adjusted": stats.f.sf(t * t / rank, rank, model.df_resid),
                    "ci_low": estimate - critical * se,
                    "ci_high": estimate + critical * se,
                }
            )
    for row in rows:
        row.update(
            {
                "family": factor,
                "target": factor,
                "condition": "all observations",
                "test": method,
                "adjustment": method,
                "ci_coverage": "simultaneous",
                "significant": row["p_adjusted"] < alpha,
            }
        )
    return rows


def _slices(effect, frame):
    """Main effects are marginal; interactions produce all simple-effect slices."""
    for target in effect:
        conditions = [col for col in effect if col != target]
        for values in product(*[levels(frame[c]) for c in conditions]):
            yield target, dict(zip(conditions, values))


def _ols_rows(effect, frame, outcome, factors, covariates, model, alpha):
    # Equal-weight marginal means at the analyzed-sample covariate means.
    grid = pd.DataFrame(product(*[levels(frame[col]) for col in factors]), columns=factors)
    for col in covariates:
        grid[col] = frame[col].mean()
    info = getattr(model.model.data, "design_info", None)
    if info is None:
        info = model.model.data.model_spec
    matrix = np.asarray(build_design_matrices([info], grid)[0])
    rows = []
    for target, condition in _slices(effect, frame):
        mask = np.ones(len(grid), dtype=bool)
        for col, value in condition.items():
            mask &= grid[col].to_numpy() == value
        means = {
            value: matrix[mask & (grid[target].to_numpy() == value)].mean(axis=0)
            for value in levels(frame[target])
        }
        for a, b in combinations(means, 2):
            contrast = means[a] - means[b]
            test = model.t_test(contrast)
            ci = np.asarray(test.conf_int(alpha=alpha)).ravel()
            rows.append(
                {
                    "target": target,
                    "condition": ", ".join(f"{c}={v}" for c, v in condition.items()) or "marginal",
                    "group_a": a,
                    "group_b": b,
                    "estimate": float(np.asarray(test.effect).item()),
                    "std_error": float(np.asarray(test.sd).item()),
                    "df": model.df_resid,
                    "t": float(np.asarray(test.tvalue).item()),
                    "p_raw": float(np.asarray(test.pvalue).item()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "test": "OLS marginal-mean contrast",
                    "ci_coverage": "pointwise",
                }
            )
    return rows


def _repeated_rows(effect, frame, outcome, subject, within, alpha):
    rows = []
    for target, condition in _slices(effect, frame):
        chunk = frame
        for col, value in condition.items():
            chunk = chunk.loc[chunk[col] == value]
        # Subject means marginalize over other within factors before comparisons.
        subject_means = (
            chunk.groupby([subject, target], observed=True)[outcome].mean().reset_index()
        )
        for a, b in combinations(levels(frame[target]), 2):
            aa = subject_means.loc[subject_means[target] == a].set_index(subject)[outcome]
            bb = subject_means.loc[subject_means[target] == b].set_index(subject)[outcome]
            paired = target in within
            if paired:
                matched = pd.concat([aa.rename("a"), bb.rename("b")], axis=1).dropna()
                delta = matched.a - matched.b
                df = len(delta) - 1
                estimate = delta.mean()
                se = delta.std(ddof=1) / np.sqrt(len(delta))
                label = "Paired subject-mean t test"
            else:
                estimate = aa.mean() - bb.mean()
                va, vb = aa.var(ddof=1) / len(aa), bb.var(ddof=1) / len(bb)
                se = np.sqrt(va + vb)
                df = (va + vb) ** 2 / (va**2 / (len(aa) - 1) + vb**2 / (len(bb) - 1))
                label = "Welch subject-mean t test"
            if not np.isfinite(se) or se <= 0 or not np.isfinite(df) or df <= 0:
                raise ValueError(
                    "A requested comparison has insufficient observations or zero contrast variance."
                )
            t = estimate / se
            margin = stats.t.ppf(1 - alpha / 2, df) * se
            rows.append(
                {
                    "target": target,
                    "condition": ", ".join(f"{c}={v}" for c, v in condition.items()) or "marginal",
                    "group_a": a,
                    "group_b": b,
                    "estimate": estimate,
                    "std_error": se,
                    "df": df,
                    "t": t,
                    "p_raw": 2 * stats.t.sf(abs(t), df),
                    "ci_low": estimate - margin,
                    "ci_high": estimate + margin,
                    "test": label,
                    "ci_coverage": "pointwise",
                }
            )
    return rows


def comparisons(
    method,
    effects,
    *,
    frame,
    outcome,
    factors,
    covariates,
    model,
    alpha,
    control=None,
    subject=None,
    within=(),
):
    if method is None:
        return None
    if not effects:
        return pd.DataFrame(
            columns=["family", "target", "group_a", "group_b", "estimate", "p_adjusted"]
        )
    if method in SPECIAL:
        return pd.DataFrame(_special(method, frame, outcome, factors[0], model, alpha, control))
    rows = []
    for effect in effects:
        current = (
            _repeated_rows(effect, frame, outcome, subject, within, alpha)
            if subject
            else _ols_rows(effect, frame, outcome, factors, covariates, model, alpha)
        )
        if current:
            adjusted = multipletests(
                [v["p_raw"] for v in current], alpha=alpha, method=ADJUSTMENTS[method]
            )[1]
            for row, p in zip(current, adjusted):
                row.update(
                    {
                        "family": ":".join(effect),
                        "adjustment": method,
                        "p_adjusted": p,
                        "significant": p < alpha,
                    }
                )
            rows.extend(current)
    return pd.DataFrame(rows)
