# API reference

All inference is two-sided where applicable. `alpha=.05` determines confidence
levels, significance labels, and post hoc gates. All analysis functions return
`AnalysisResult`; they do not generate figures.

## Shared conventions

- `data=` must be a pandas dataframe with unique column names.
- Formula: a quoted `"outcome ~ predictors"`. Supports `+`, `*`, `:`, parentheses,
  and `0`/`-1` intercept removal for `lm` only. A single outcome is required.
- Backticks quote names with spaces: ``"`exam score` ~ group"``.
- `missing="drop"` explicitly reports deletion; `missing="raise"` rejects it.
  Infinite analysis values are rejected.
- Formulas are a restricted column/operator language; arbitrary Python/R
  expressions, transformations, and function evaluation are not supported.
  Prepare transformed columns with pandas first.
- `result.tables` contains full-precision pandas dataframes. `result.metadata`
  records choices, `result.diagnostics` contains notes, and `result.model`
  exposes an underlying fitted statsmodels result when one exists.
- `result.show()` displays the report; `print(result)` prints a text version.
  Notebook cells display a rich HTML report automatically.

## `descriptives(*, data, columns=None, by=None, alpha=.05)`

`columns` defaults to all nongrouping columns. `by` takes a name or list of names.
Available values are used independently per column; missing grouping values
form their own group. Numeric output includes valid/missing n, mean, median,
modes, sample SD/variance, min/max/range, Q1/Q3/IQR, unscaled MAD, bias-corrected
skewness/excess kurtosis, and t-based mean CIs. Categorical output contains
counts, valid percentages, and missing n. Undefined quantities remain missing.

## `table(formula, *, data, percent="valid", missing="drop")`

`"group"` produces a one-way table; `"group ~ passed"` produces a two-way
table with the left side as rows and right side as columns.
`percent="valid"`/`"total"` uses the valid grand total; `"row"` and `"column"`
use their respective two-way totals. Counts, percentages, and variable-specific
missingness are separate reusable tables.

## `chisq(formula, *, data=None, expected=None, method="pearson", correction=False, missing="drop")`

Use a two-way formula for independence. Pearson reports χ², df, p, uncorrected
Cramér's V, expected counts, and Pearson residuals. Set `correction=True` for
Yates correction on a 2×2 table; the association effect size remains uncorrected.
`method="fisher"` requests a two-sided 2×2 Fisher exact test with sample odds
ratio. Fisher does not use a continuity correction.

For goodness of fit, pass counts directly and fixed expected probabilities:

```python
rs.chisq([20, 30, 50], expected=[.25, .25, .5])
```

Omitted probabilities default to equal proportions. Counts must be nonnegative
integers; probabilities must be positive and sum to one. Reports Cohen's w.

## `aov(formula, *, data, covariates=None, type=None, method="standard", subject=None, within=None, posthocifsig=False, control=None, alpha=.05, correction="gg", missing="drop")`

Formula predictors are categorical, including numeric-coded groups. `covariates`
adds continuous additive adjustments; those variables must not also appear as
grouping predictors. `type=None` selects Type II for ordinary OLS designs.
Use integers 1, 2, or 3 explicitly for supported sums-of-squares types.
`method="welch"` is for one independent group factor without covariates.

Repeated/mixed designs require `subject` and `within` together, long-format
data, and a full factorial formula. `within` takes one name or a two-name list.
There may be one between factor only when there is exactly one within factor.
Do not supply `type` or `covariates` for these designs.

`correction="gg"` makes Greenhouse–Geisser inference primary where available;
`"none"` makes uncorrected inference primary. Both results are retained. For
ordinary/Welch designs this argument has no effect.

See [ANOVA and post hoc methods](anova.md) for compatibility and interpretation.

## `lm(formula, *, data, categorical=None, reference=None, alpha=.05, missing="drop")`

Ordinary least squares for continuous outcomes. Text/category predictors are
categorical automatically; `categorical=["group"]` marks numeric-coded groups.
`factor(group)` in the formula is an alternative. Use
`reference={"group": "control"}` to choose a baseline.

By default, the first observed level in declared pandas category order, or the
first sorted observed value, is the reference. Unused categories are removed.
Reports overall F, df, p, R²/adjusted R², residual SD, coefficient estimates,
SEs, t tests, confidence intervals, numeric descriptives, and diagnostic tests.
For models without an intercept, statsmodels reports uncentered R²; interpret
it separately from the usual centered R². No overall F exists for intercept-only
models; it is missing rather than a significance claim.

## `logmodel(formula, *, data, categorical=None, reference=None, event=None, alpha=.05, missing="drop")`

Binary logistic regression. Category/reference handling matches `lm`. Numeric
0/1 outcomes model 1; otherwise specify the event explicitly. Exactly two
observed outcomes and an intercept are required.

Reports the overall likelihood-ratio χ² against the intercept-only model,
McFadden pseudo-R², AIC, outcome counts, coefficients, z tests, odds ratios and
Wald CIs. Nonconvergence, detected separation, invalid covariance, and redundant
predictors raise explanatory errors. Large finite coefficients/SEs produce a
quasi-separation note, not a claim that every separation case can be detected.

## `vis(formula, *, data, type=None, categorical=None, hue=None, palette=None, colors="default", xlabel=None, ylabel=None, title=None, estimator=None, normalize=None, units=None, ax=None, **kwargs)`

Standalone plotting; returns Matplotlib Axes. No statistical model is fit by
this command. Complete cases are used for the plotted columns.

| Type | Input / additional requirement |
| --- | --- |
| `scatter` | `"numeric_y ~ numeric_x"` |
| `box`, `violin` | Numeric and grouping variable; explicit type treats numeric-coded groups as categories |
| `hist`, `density` | One numeric column name |
| `count` | One column, or two-way formula with outcome as hue |
| `proportion` | Two-way formula and `normalize="row"`, `"column"`, or `"total"`; row means within RHS group |
| `heatmap` | Two-way contingency counts |
| `line` | `estimator="mean"`/`"median"`, or `units="student_id"` for individual profiles |

Automatic defaults: numeric/numeric → scatter; numeric/categorical → box;
categorical/categorical → contingency heatmap; one numeric → histogram;
one categorical → count. Use `categorical=` to override numeric-coded groups.

`colors="default"` uses a colorblind-friendly palette. A Seaborn palette can be
specified with `palette=` or `colors=`. For continuous hue use a suitable
sequential palette explicitly. `**kwargs` passes plot-specific Seaborn options,
such as `bins=20`, `alpha=.6`, or `errorbar=None`. Heatmaps use `cmap=`.
Seaborn aggregate line intervals use its own bootstrap defaults; they are not
ANOVA-adjusted or repeated-measures contrast intervals. Set `seed=` for
reproducible bootstrap plots.
