"""Shared validation and a deliberately small, non-evaluating formula language."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from patsy import ModelDesc


def names(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def dataframe(data, columns):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame. Load your data with pandas first.")
    if not data.columns.is_unique:
        raise ValueError("Dataframe column names must be unique.")
    absent = [c for c in columns if c not in data.columns]
    if absent:
        raise ValueError(f"Columns not found: {absent}.")
    return data.loc[:, list(dict.fromkeys(columns))].copy(deep=True)


def complete(data, columns, missing="drop"):
    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'.")
    out = dataframe(data, columns)
    for col in out.select_dtypes(include="number"):
        if np.isinf(out[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{col!r} contains infinite values; resolve them before analysis.")
    excluded = int(out.isna().any(axis=1).sum())
    if excluded and missing == "raise":
        raise ValueError(
            f"{excluded} rows have missing analysis values; resolve them or use missing='drop'."
        )
    out = out.dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError("No complete observations remain for this analysis.")
    return out, [f"Analyzed {len(out)} of {len(data)} rows; excluded {excluded} incomplete rows."]


def numeric(series, label):
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise ValueError(f"{label!r} must contain numeric values.")


def alpha_value(alpha):
    if isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def levels(series):
    observed = series.dropna().unique().tolist()
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [v for v in series.cat.categories if v in observed]
    try:
        return sorted(observed)
    except TypeError:
        return sorted(observed, key=lambda v: (type(v).__name__, str(v)))


@dataclass
class Formula:
    outcome: str
    predictors: list[str]
    forced: list[str]
    expression: str
    tokens: dict[str, str]

    def backend(self, factors=(), reference=None, sum_contrasts=False):
        reference = reference or {}
        replacements = {}
        for token, name in self.tokens.items():
            quoted = f"Q({name!r})"
            if name in factors:
                contrast = (
                    "Sum"
                    if sum_contrasts
                    else (
                        f"Treatment(reference={reference[name]!r})"
                        if name in reference
                        else "Treatment"
                    )
                )
                quoted = f"C({quoted}, {contrast})"
            replacements[token] = quoted
        expression = re.sub(r"\bv\d+\b", lambda m: replacements[m[0]], self.expression)
        return f"Q({self.outcome!r}) ~ {expression}"


def parse(formula):
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise ValueError("Use a quoted formula such as 'score ~ group + age'.")
    lhs, rhs = [v.strip() for v in formula.split("~")]
    atom = r"(?:`[^`]+`|[A-Za-z_]\w*)"
    if not re.fullmatch(atom, lhs):
        raise ValueError("Use one outcome column; put names with spaces in backticks.")
    token_pattern = re.compile(rf"(?:factor|C)\(\s*({atom})\s*\)|({atom})|([01])|([+*:\-()])")
    cursor, tokens, predictors, forced, expression = 0, {}, [], [], []
    for match in token_pattern.finditer(rhs):
        if rhs[cursor : match.start()].strip():
            raise ValueError(
                "Supported formulas use column names, factor(name), +, *, :, and intercept 0/1."
            )
        cursor = match.end()
        if match[1] or match[2]:
            name = (match[1] or match[2]).strip("`")
            if name not in predictors:
                predictors.append(name)
            token = f"v{predictors.index(name)}"
            tokens[token] = name
            expression.append(token)
            if match[1] and name not in forced:
                forced.append(name)
        else:
            expression.append(match[3] or match[4])
    if rhs[cursor:].strip() or not expression:
        raise ValueError("Unsupported formula. Example: 'score ~ group * method + age'.")
    expr = " ".join(expression)
    if re.search(r"-\s*(?!1\b)", expr):
        raise ValueError("Only '- 1' is supported for removing an intercept.")
    try:
        ModelDesc.from_formula("y ~ " + expr)
    except Exception as exc:
        raise ValueError(f"Invalid formula: {formula!r}.") from exc
    return Formula(lhs.strip("`"), predictors, forced, expr, tokens)


def factorize(data, candidates, reference=None):
    reference = reference or {}
    unknown = set(reference) - set(candidates)
    if unknown:
        raise ValueError(
            f"Reference groups supplied for non-categorical predictors: {sorted(unknown)}."
        )
    coding = {}
    for col in candidates:
        lev = levels(data[col])
        if len(lev) < 2:
            raise ValueError(f"Categorical predictor {col!r} needs at least two observed levels.")
        if col in reference:
            if reference[col] not in lev:
                raise ValueError(
                    f"Reference {reference[col]!r} is not an observed level of {col!r}."
                )
            lev = [reference[col]] + [v for v in lev if v != reference[col]]
        data[col] = pd.Categorical(data[col].to_numpy(), categories=lev)
        coding[col] = {"levels": lev, "reference": lev[0]}
    return coding


def sample_table(total, analyzed):
    return pd.DataFrame(
        [{"input_rows": total, "analyzed_rows": analyzed, "excluded_rows": total - analyzed}]
    )


def clean_term(term, spec):
    for col in sorted(spec.predictors + [spec.outcome], key=len, reverse=True):
        term = term.replace(f"Q({col!r})", col)
    term = re.sub(r"C\(([^,]+), (?:Sum|Treatment(?:\(reference=.*?\))?)\)", r"\1", term)
    return term
