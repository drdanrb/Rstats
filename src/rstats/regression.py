"""Linear and binary logistic regression with complete reports."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import ModelDesc
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

from ._data import (
    alpha_value,
    clean_term,
    complete,
    factorize,
    names,
    numeric,
    parse,
    sample_table,
)
from .result import AnalysisResult


def prepare_regression(formula, data, categorical, reference, missing):
    spec = parse(formula)
    if spec.outcome in spec.predictors:
        raise ValueError("The outcome cannot also be a predictor.")
    explicit = names(categorical) + spec.forced
    if set(explicit) - set(spec.predictors):
        raise ValueError("categorical must name predictors in the formula.")
    frame, notes = complete(data, [spec.outcome, *spec.predictors], missing)
    factors = list(
        dict.fromkeys(
            explicit
            + [
                col
                for col in spec.predictors
                if not pd.api.types.is_numeric_dtype(frame[col])
                or pd.api.types.is_bool_dtype(frame[col])
            ]
        )
    )
    coding = factorize(frame, factors, reference)
    return spec, frame, factors, coding, notes


def validate_model(model):
    x = model.model.exog
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError(
            "The model is rank deficient: remove redundant predictors or empty interaction cells."
        )
    if model.df_resid <= 0:
        raise ValueError(
            "The model has no residual degrees of freedom; add observations or simplify it."
        )
    if (
        not np.isfinite(np.asarray(model.params)).all()
        or not np.isfinite(np.asarray(model.cov_params())).all()
    ):
        raise ValueError(
            "Model estimates or uncertainty are not finite; check for a degenerate model."
        )


def fit_ols(formula, frame):
    unfitted = smf.ols(formula, data=frame)
    if np.linalg.matrix_rank(unfitted.exog) < unfitted.exog.shape[1]:
        raise ValueError(
            "The model is rank deficient: remove redundant predictors or empty interaction cells."
        )
    if len(unfitted.endog) <= unfitted.exog.shape[1]:
        raise ValueError(
            "The model has no residual degrees of freedom; add observations or simplify it."
        )
    model = unfitted.fit()
    validate_model(model)
    if model.ssr <= np.finfo(float).eps * max(1, model.centered_tss):
        raise ValueError(
            "The model has effectively zero residual variance; ordinary inferential tests are undefined."
        )
    return model


def coefficients(model, spec, alpha, logistic=False):
    ci = model.conf_int(alpha=alpha)
    result = pd.DataFrame(
        {
            "estimate": model.params,
            "std_error": model.bse,
            "z" if logistic else "t": model.tvalues,
            "p": model.pvalues,
            "ci_low": ci.iloc[:, 0],
            "ci_high": ci.iloc[:, 1],
        }
    )
    if logistic:
        with np.errstate(over="ignore"):
            result["odds_ratio"] = np.exp(model.params)
            result["or_ci_low"] = np.exp(ci.iloc[:, 0])
            result["or_ci_high"] = np.exp(ci.iloc[:, 1])
    result.index = [clean_term(str(v), spec) for v in result.index]
    result.index.name = "term"
    return result


def coding_table(coding):
    return pd.DataFrame(
        [
            {
                "predictor": col,
                "reference": value["reference"],
                "levels": ", ".join(map(str, value["levels"])),
            }
            for col, value in coding.items()
        ],
        columns=["predictor", "reference", "levels"],
    )


def ols_diagnostics(model):
    notes, rows = [], []
    residuals = np.asarray(model.resid)
    if len(residuals) >= 3 and np.std(residuals) > 0:
        jb = stats.jarque_bera(residuals)
        rows.append(
            {
                "diagnostic": "Residual normality (Jarque–Bera)",
                "statistic": jb.statistic,
                "p": jb.pvalue,
            }
        )
    if "Intercept" in model.model.exog_names and model.model.exog.shape[1] > 1:
        _, _, f, p = het_breuschpagan(residuals, model.model.exog)
        rows.append({"diagnostic": "Constant variance (Breusch–Pagan F)", "statistic": f, "p": p})
    if model.model.exog.shape[1] > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            vifs = [
                variance_inflation_factor(model.model.exog, i)
                for i, name in enumerate(model.model.exog_names)
                if name != "Intercept"
            ]
        if vifs and max(vifs) > 5:
            notes.append("Some predictor VIFs exceed 5; inspect collinearity and factor coding.")
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = model.get_influence().cooks_distance[0]
    count = int(np.sum(cooks > 4 / model.nobs))
    notes.append(f"{count} observations have Cook's distance > 4/n; inspect influential cases.")
    notes.append(
        "Diagnostics do not change the fitted method. Independence follows from the study design."
    )
    return pd.DataFrame(rows, columns=["diagnostic", "statistic", "p"]), notes


def lm(formula, *, data, categorical=None, reference=None, alpha=0.05, missing="drop"):
    """Fit simple/multiple OLS regression and display a complete teaching report.

    Numeric-coded groups: ``lm('score ~ age + group', data=df,
    categorical=['group'], reference={'group': 'control'})``.
    """
    alpha_value(alpha)
    spec, frame, factors, coding, notes = prepare_regression(
        formula, data, categorical, reference, missing
    )
    numeric(frame[spec.outcome], spec.outcome)
    if frame[spec.outcome].nunique() < 2:
        raise ValueError("The outcome must vary to fit a regression.")
    model = fit_ols(spec.backend(factors, reference), frame)
    diag, extra = ols_diagnostics(model)
    notes.extend(extra)
    notes.append(
        f"OLS with {100 * (1 - alpha):g}% confidence intervals. Coefficients are adjusted for the other predictors."
    )
    if "Intercept" not in model.model.exog_names:
        notes.append(
            "This model has no intercept; R² is uncentered and is not directly comparable to ordinary centered R²."
        )
    tables = {
        "sample": sample_table(len(data), len(frame)),
        "model": pd.DataFrame(
            [
                {
                    "n": int(model.nobs),
                    "F": model.fvalue,
                    "df_model": model.df_model,
                    "df_residual": model.df_resid,
                    "p": model.f_pvalue,
                    "r_squared": model.rsquared,
                    "adjusted_r_squared": model.rsquared_adj,
                    "residual_sd": np.sqrt(model.mse_resid),
                }
            ]
        ),
        "coefficients": coefficients(model, spec, alpha),
        "descriptives": frame.select_dtypes(include="number").describe().T,
        "diagnostics": diag,
    }
    if coding:
        tables["coding"] = coding_table(coding)
    return AnalysisResult(
        "Linear regression (OLS)",
        tables,
        notes,
        {"formula": formula, "method": "OLS", "alpha": alpha, "coding": coding},
        model,
    )


def logmodel(
    formula, *, data, categorical=None, reference=None, event=None, alpha=0.05, missing="drop"
):
    """Fit binary logistic regression, with odds ratios and an overall LR test.

    Numeric 0/1 outcomes model 1 by default. For other binary outcomes, specify
    ``event='passed'``. Reference groups refer to predictors, not the outcome.
    """
    alpha_value(alpha)
    spec, frame, factors, coding, notes = prepare_regression(
        formula, data, categorical, reference, missing
    )
    observed = frame[spec.outcome].unique().tolist()
    if len(observed) != 2:
        raise ValueError("logmodel needs exactly two observed outcome values.")
    if event is None:
        if not pd.api.types.is_numeric_dtype(frame[spec.outcome]) or set(observed) != {0, 1}:
            raise ValueError("Specify event= for a binary outcome other than numeric 0/1.")
        event = 1
    if event not in observed:
        raise ValueError(f"event={event!r} is not an observed outcome value.")
    baseline = next(v for v in observed if v != event)
    frame[spec.outcome] = (frame[spec.outcome] == event).astype(int)
    if not any(
        len(t.factors) == 0 for t in ModelDesc.from_formula("y ~ " + spec.expression).rhs_termlist
    ):
        raise ValueError("logmodel requires an intercept for its reported null-model comparison.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", RuntimeWarning)
            model = smf.logit(spec.backend(factors, reference), data=frame).fit(
                disp=False, maxiter=200
            )
    except (
        PerfectSeparationWarning,
        PerfectSeparationError,
        np.linalg.LinAlgError,
        RuntimeWarning,
    ) as exc:
        raise ValueError(
            "Logistic inference failed: inspect separation, redundant predictors, and sparse groups."
        ) from exc
    if not model.mle_retvals.get("converged", False):
        raise ValueError(
            "Logistic regression did not converge; inspect separation and predictor scales."
        )
    validate_model(model)
    notes.extend(
        [
            f"Models P({spec.outcome} = {event!r}); the other outcome is {baseline!r}.",
            f"Odds ratios compare a one-unit change, or a category against its reference, holding other predictors constant. {100 * (1 - alpha):g}% Wald confidence intervals.",
            "McFadden pseudo-R² is not the proportion of variance explained. Predictions are in model.predict().",
        ]
    )
    if np.max(np.abs(model.params)) > 15 or np.max(model.bse) > 100:
        notes.append(
            "Large estimates or standard errors suggest possible quasi-separation; interpret with caution."
        )
    tables = {
        "sample": sample_table(len(data), len(frame)),
        "outcome_counts": pd.DataFrame(
            {
                "outcome": [baseline, event],
                "n": [int((frame[spec.outcome] == 0).sum()), int(frame[spec.outcome].sum())],
            }
        ),
        "model": pd.DataFrame(
            [
                {
                    "n": int(model.nobs),
                    "LR_chi2": model.llr,
                    "df": model.df_model,
                    "p": model.llr_pvalue,
                    "mcfadden_pseudo_r_squared": model.prsquared,
                    "AIC": model.aic,
                    "converged": True,
                }
            ]
        ),
        "coefficients": coefficients(model, spec, alpha, logistic=True),
    }
    if coding:
        tables["coding"] = coding_table(coding)
    return AnalysisResult(
        "Binary logistic regression",
        tables,
        notes,
        {"formula": formula, "method": "logit", "event": event, "alpha": alpha, "coding": coding},
        model,
    )
