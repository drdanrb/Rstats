"""ANOVA/ANCOVA reports with explicit designs and sums of squares."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from patsy import ModelDesc
from scipy import stats
from statsmodels.stats.anova import anova_lm

from ._data import (
    alpha_value,
    clean_term,
    complete,
    dataframe,
    factorize,
    levels,
    names,
    numeric,
    parse,
    sample_table,
)
from ._posthoc import comparisons, method_name, validate_method
from .regression import coding_table, fit_ols, ols_diagnostics
from .result import AnalysisResult


def _group_descriptives(frame, outcome, factors, alpha):
    rows = []
    for keys, chunk in frame.groupby(factors, observed=True, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        x = chunk[outcome].to_numpy(dtype=float)
        n, mean = len(x), np.mean(x)
        sd = np.std(x, ddof=1) if n > 1 else np.nan
        margin = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(
            {
                **dict(zip(factors, keys)),
                "n": n,
                "mean": mean,
                "sd": sd,
                "median": np.median(x),
                "mean_ci_low": mean - margin,
                "mean_ci_high": mean + margin,
            }
        )
    return pd.DataFrame(rows)


def _prepare_repeated(data, spec, within, between, subject, missing):
    if subject in spec.predictors or subject == spec.outcome:
        raise ValueError("subject identifies people and must not appear in the model formula.")
    columns = [spec.outcome, *spec.predictors, subject]
    raw = dataframe(data, columns)
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'drop' or 'raise'.")
    numeric(raw[spec.outcome], spec.outcome)
    if np.isinf(raw[spec.outcome].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("The outcome contains infinite values.")
    identifiable = raw.dropna(subset=[subject])
    if identifiable.duplicated([subject, *within]).any():
        raise ValueError(
            "Duplicate subject-condition cells: aggregate deliberate replicates before aov()."
        )
    for col in between:
        if identifiable.groupby(subject, observed=True)[col].nunique(dropna=True).gt(1).any():
            raise ValueError(f"Between-subject factor {col!r} changes within a subject.")
    expected_cells = int(np.prod([len(levels(raw[col])) for col in within]))
    if not expected_cells or any(len(levels(raw[col])) < 2 for col in within):
        raise ValueError("Each within-subject factor needs at least two observed levels.")
    valid = []
    for sid, chunk in identifiable.groupby(subject, observed=True):
        if len(chunk) == expected_cells and not chunk.isna().any().any():
            valid.append(sid)
    frame = identifiable.loc[identifiable[subject].isin(valid)].copy().reset_index(drop=True)
    excluded_subjects = identifiable[subject].nunique() - len(valid)
    if missing == "raise" and (excluded_subjects or raw.isna().any().any()):
        raise ValueError(
            "Incomplete subjects or missing repeated cells; resolve them or use missing='drop'."
        )
    if len(valid) < 3:
        raise ValueError("Repeated/mixed ANOVA needs at least three complete subjects.")
    if between and frame.groupby(between, observed=True)[subject].nunique().min() < 2:
        raise ValueError(
            "Mixed ANOVA needs at least two complete subjects per between-subject group."
        )
    notes = [
        f"Analyzed {len(valid)} complete subjects ({len(frame)} rows); excluded {excluded_subjects} incomplete subjects and {int(raw[subject].isna().sum())} rows without subject IDs.",
        "Complete-subject deletion can discard substantial information. Consider a mixed-effects model for incomplete longitudinal data.",
    ]
    return frame, notes, len(valid), int(excluded_subjects)


def aov(
    formula,
    *,
    data,
    covariates=None,
    type=None,
    method="standard",
    subject=None,
    within=None,
    posthocifsig=False,
    control=None,
    alpha=0.05,
    correction="gg",
    missing="drop",
):
    """ANOVA/ANCOVA with factor predictors and optional continuous covariates.

    ``type`` is sums of squares (I/II/III), not design. The ordinary default is
    II. ``within`` plus ``subject`` declares repeated/mixed designs. Numeric
    formula predictors are groups; use ``covariates=['age']`` for ANCOVA.

    ``posthocifsig=True`` means Tukey; a method string selects another procedure.
    Incompatible requests raise an explanation rather than changing the test.
    """
    alpha_value(alpha)
    if method not in {"standard", "welch"}:
        raise ValueError("method must be 'standard' or 'welch'.")
    if correction not in {"gg", "none"}:
        raise ValueError("correction must be 'gg' or 'none'.")
    if type is not None and (isinstance(type, bool) or type not in {1, 2, 3}):
        raise ValueError("type must be 1, 2, or 3 (sums of squares).")
    spec = parse(formula)
    factors, covars, within = spec.predictors, names(covariates), names(within)
    if not factors or spec.outcome in factors:
        raise ValueError("ANOVA needs at least one grouping predictor distinct from the outcome.")
    if len(set(covars)) != len(covars) or len(set(within)) != len(within):
        raise ValueError("covariates and within must not contain duplicate names.")
    if set(covars) & {spec.outcome, *factors}:
        raise ValueError(
            "Covariates must be separate from the outcome and grouping predictors in the formula."
        )
    if (subject is None) != (not within):
        raise ValueError("Repeated designs require both subject='id' and within='time'.")
    if set(within) - set(factors):
        raise ValueError("within must name grouping predictors in the formula.")
    desc = ModelDesc.from_formula("y ~ " + spec.expression)
    if not any(not t.factors for t in desc.rhs_termlist):
        raise ValueError("aov requires an intercept; use lm for no-intercept models.")
    posthoc = method_name(posthocifsig)
    if method == "welch" and (len(factors) != 1 or covars or within):
        raise ValueError("Welch ANOVA supports one independent grouping factor and no covariates.")
    if method == "welch" and type is not None:
        raise ValueError("Welch ANOVA does not use Type I/II/III sums of squares; omit type=.")
    if method == "welch" and posthoc in {"tukey", "scheffe", "dunnett"}:
        raise ValueError(
            "That post hoc assumes equal variances. After Welch use posthocifsig='games-howell'."
        )
    if method == "welch" and posthoc is not None and posthoc != "games-howell":
        raise ValueError("Welch ANOVA currently supports posthocifsig='games-howell'.")
    between = [f for f in factors if f not in within]
    model = None
    excluded_subjects = 0
    if within:
        if (
            covars
            or type is not None
            or len(within) > 2
            or len(between) > 1
            or (between and len(within) != 1)
        ):
            raise ValueError(
                "Supported repeated designs: 1–2 within factors, or 1 within + 1 between; omit covariates and type=."
            )
        expected_terms = {
            frozenset(combo)
            for size in range(1, len(factors) + 1)
            for combo in combinations(spec.tokens, size)
        }
        actual_terms = {
            frozenset(f.name() for f in term.factors) for term in desc.rhs_termlist if term.factors
        }
        if actual_terms != expected_terms:
            raise ValueError(
                "Repeated/mixed ANOVA requires the full factorial formula, e.g. 'score ~ time * group'."
            )
        frame, notes, n_subjects, excluded_subjects = _prepare_repeated(
            data, spec, within, between, subject, missing
        )
    else:
        frame, notes = complete(data, [spec.outcome, *factors, *covars], missing)
    numeric(frame[spec.outcome], spec.outcome)
    if frame[spec.outcome].nunique() < 2:
        raise ValueError("The outcome must vary to compute ANOVA.")
    for col in covars:
        numeric(frame[col], col)
    coding = factorize(frame, factors)
    if within or method == "welch" or type == 3:
        for value in coding.values():
            value["reference"] = (
                "not applicable (sum contrasts)" if type == 3 else "not applicable (effect tests)"
            )
    validate_method(
        posthoc,
        factors=factors,
        covariates=covars,
        repeated=bool(within),
        control=control,
        frame=frame,
    )
    tables = {
        "sample": sample_table(len(data), len(frame)),
        "group_descriptives": _group_descriptives(frame, spec.outcome, factors, alpha),
        "coding": coding_table(coding),
    }
    effect_map = {}
    if within:
        if between:
            raw = pg.mixed_anova(
                data=frame,
                dv=spec.outcome,
                within=within[0],
                between=between[0],
                subject=subject,
                correction=True,
                effsize="np2",
            )
            effect_map = {
                between[0]: [between[0]],
                within[0]: [within[0]],
                "Interaction": [between[0], within[0]],
            }
            notes.append(
                "Mixed-design effect tests use the backend's Type II convention for unequal between-group sample sizes; this can differ from Type III software defaults."
            )
            design = "mixed"
        else:
            raw = pg.rm_anova(
                data=frame,
                dv=spec.outcome,
                within=within,
                subject=subject,
                detailed=True,
                correction=True,
                effsize="np2",
            )
            effect_map = {name: name.split(" * ") for name in raw.Source if name != "Error"}
            design = "repeated"
        tests = raw.rename(
            columns={
                "Source": "effect",
                "SS": "sum_sq",
                "MS": "mean_sq",
                "DF": "df_numerator",
                "DF1": "df_numerator",
                "ddof1": "df_numerator",
                "DF2": "df_denominator",
                "ddof2": "df_denominator",
                "p_unc": "p_uncorrected",
                "p_GG_corr": "p_gg",
                "np2": "partial_eta_squared",
            }
        )
        if "df_denominator" not in tests:
            # Pingouin's detailed one-factor RM table puts the error df on its
            # own row; multi-factor designs carry a denominator on each effect.
            error_df = tests.loc[tests.effect == "Error", "df_numerator"].iloc[0]
            tests["df_denominator"] = np.where(tests.effect == "Error", np.nan, error_df)
        tests["p"] = (
            tests["p_gg"].fillna(tests["p_uncorrected"])
            if correction == "gg" and "p_gg" in tests
            else tests["p_uncorrected"]
        )
        tests["primary_inference"] = np.where(
            tests.get("p_gg", pd.Series(np.nan, index=tests.index)).notna() & (correction == "gg"),
            "Greenhouse–Geisser",
            "uncorrected",
        )
        if "eps" in tests and "df_denominator" in tests:
            tests["df_numerator_gg"] = tests["df_numerator"] * tests["eps"]
            tests["df_denominator_gg"] = tests["df_denominator"] * tests["eps"]
        tables["subjects"] = pd.DataFrame(
            [{"analyzed_subjects": n_subjects, "excluded_subjects": excluded_subjects}]
        )
        notes.extend(
            [
                "Greenhouse–Geisser corrected inference is primary where available unless correction='none'; both p values are shown. No diagnostic-driven switching is performed.",
                "Cell mean CIs are pointwise descriptive intervals, not within-subject contrast intervals.",
            ]
        )
    elif method == "welch":
        factor = factors[0]
        if (
            frame.groupby(factor, observed=True)[spec.outcome]
            .agg(["count", "var"])
            .pipe(lambda t: (t["count"] < 2).any() or (t["var"] <= 0).any())
        ):
            raise ValueError(
                "Welch ANOVA needs at least two observations and positive variance in each group."
            )
        tests = pg.welch_anova(data=frame, dv=spec.outcome, between=factor).rename(
            columns={
                "Source": "effect",
                "ddof1": "df_numerator",
                "ddof2": "df_denominator",
                "p_unc": "p",
                "np2": "partial_eta_squared",
            }
        )
        effect_map = {factor: [factor]}
        design = "one-way Welch"
        notes.append(
            "Welch ANOVA permits unequal group variances; its denominator degrees of freedom are approximate."
        )
    else:
        fit_spec = parse(formula + "".join(f" + `{c}`" for c in covars))
        backend = fit_spec.backend(factors, sum_contrasts=type == 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                model = fit_ols(backend, frame)
                raw = anova_lm(model, typ=type or 2)
            except RuntimeWarning as exc:
                raise ValueError(
                    "ANOVA inference is undefined; inspect empty cells or a zero residual variance."
                ) from exc
        tests = raw.rename(columns={"PR(>F)": "p", "df": "df_numerator"}).copy()
        tests["mean_sq"] = tests.sum_sq / tests.df_numerator
        tests["df_denominator"] = model.df_resid
        tests["partial_eta_squared"] = tests.sum_sq / (tests.sum_sq + model.ssr)
        tests.loc["Residual", ["partial_eta_squared", "df_denominator"]] = np.nan
        for term in raw.index:
            if term not in {"Intercept", "Residual"}:
                members = [f for f in factors if f"Q({f!r})" in term]
                if members:
                    effect_map[clean_term(term, fit_spec)] = members
        tests.index = [clean_term(str(term), fit_spec) for term in tests.index]
        tests = tests.rename_axis("effect").reset_index()
        tables["model"] = pd.DataFrame(
            [
                {
                    "n": int(model.nobs),
                    "F": model.fvalue,
                    "df_model": model.df_model,
                    "df_residual": model.df_resid,
                    "p": model.f_pvalue,
                    "r_squared": model.rsquared,
                    "adjusted_r_squared": model.rsquared_adj,
                }
            ]
        )
        diagnostic_table, diagnostic_notes = ols_diagnostics(model)
        tables["diagnostics"] = diagnostic_table
        notes.extend(diagnostic_notes)
        notes.append(
            f"Type {type or 2} sums of squares; {'sum' if type == 3 else 'treatment'} contrasts. Group descriptives are unadjusted; ANCOVA comparisons use covariate-adjusted marginal means."
        )
        if type == 1:
            notes.append("Type I tests are sequential and depend on formula term order.")
        if any(len(t.factors) > 1 for t in desc.rhs_termlist):
            notes.append(
                "Interpret main effects cautiously when interactions are present. Type II main effects do not adjust for interactions containing that effect."
            )
        design = "ANCOVA" if covars else "between-subjects"
    effect_tests = tests.loc[tests.effect.isin(effect_map)]
    if not np.isfinite(effect_tests["F"]).all() or not np.isfinite(effect_tests["p"]).all():
        raise ValueError(
            "ANOVA returned undefined effect tests; inspect degenerate variances or insufficient replication."
        )
    tables["anova"] = tests
    significant = [effect_map[row.effect] for row in effect_tests.itertuples() if row.p < alpha]
    posthoc_table = comparisons(
        posthoc,
        significant,
        frame=frame,
        outcome=spec.outcome,
        factors=factors,
        covariates=covars,
        model=model,
        alpha=alpha,
        control=control,
        subject=subject,
        within=within,
    )
    if posthoc_table is not None:
        tables["posthoc"] = posthoc_table
        notes.append(
            f"Post hoc method: {posthoc}; comparisons run only for relevant omnibus p < {alpha:g}. Adjustment families include all comparisons generated for each triggering effect."
        )
        if not significant:
            notes.append(
                "No relevant omnibus effect was significant; no post hoc comparisons were run."
            )
        if posthoc in {"fdr-bh", "fdr-by"}:
            notes.append(
                "This is exploratory false-discovery-rate control, not familywise error control. Omnibus gating is an additional selection step, not a guarantee of global error control."
            )
        if posthoc in {"sidak", "holm-sidak"}:
            notes.append(
                "Šidák procedures rely on independence or suitable dependence assumptions; Holm is a safer general-purpose choice under arbitrary dependence."
            )
        if posthoc == "dunnett":
            notes.append(
                "Dunnett probabilities/CIs use numerical multivariate-t integration with a fixed seed; small numerical differences are possible."
            )
        if posthoc not in {"tukey", "games-howell", "dunnett", "scheffe"}:
            notes.append(
                "Post hoc confidence intervals are pointwise; significance uses adjusted p values. Interaction families contain simple effects in each conditioning stratum."
            )
    notes.append(
        "Mean differences are not causal effects unless justified by the study design. Assumption diagnostics never automatically replace the requested test."
    )
    return AnalysisResult(
        f"ANOVA — {design}",
        tables,
        notes,
        {
            "formula": formula,
            "design": design,
            "type": (type or 2) if not within and method != "welch" else None,
            "method": method,
            "alpha": alpha,
            "posthoc": posthoc,
            "coding": coding,
            "covariates": covars,
        },
        model,
    )
