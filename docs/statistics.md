# Statistical conventions and limitations

## Defaults are visible

Rstats does not select tests from normality/variance p values. It shows methods,
coding, exclusions, uncertainty, and relevant diagnostics. `alpha=.05` is a
default decision threshold, not a measure of scientific importance. Statistical
significance alone does not establish causation or practical importance.

## Descriptives

SD and variance use n−1. Quartiles use NumPy's default linear interpolation.
MAD is the unscaled median absolute deviation. Skewness and excess kurtosis
are bias-corrected, and undefined constant/small-sample quantities are missing.
Mean confidence intervals use Student's t; they rely on the sampling model and
are not universal distribution-free intervals. Modes include all ties.
Numeric-coded categories should be converted to pandas category dtype before
descriptives if frequency summaries are desired.

## Samples and denominators

Descriptives use available observations per variable. Frequency percentages use
valid counts unless another explicit denominator is requested. Inferential
analyses use complete cases across analysis variables. Repeated/mixed analyses
use complete subjects, including absent condition rows. This can reduce power
or introduce bias when missingness is informative. No automatic imputation is
performed. Plotting independently uses complete cases for plotted variables;
use the same prefiltered dataframe if exact analytic-sample alignment matters.

## Categorical tests

Pearson independence χ² is uncorrected by default, matching an explicit
`chisq.test(..., correct=FALSE)` in R. Cramér's V is uncorrected and not bias
adjusted. Pearson residuals are `(observed−expected)/sqrt(expected)`, not
standardized adjusted residuals. Sparse-cell notes trigger if any expected count
is below 1 or more than 20% are below 5. Fisher is explicit and only 2×2.
Goodness-of-fit uses k−1 df and externally fixed expected probabilities; models
with probabilities estimated from the same sample are unsupported.

## Regression

OLS uses conventional, nonrobust covariance. Reports include Jarque–Bera
residual normality and Breusch–Pagan F tests when defined, a VIF>5 note when
applicable, and counts of Cook's distances exceeding 4/n. VIF interpretation
with interactions and categorical coding requires care. Independence, functional
form, omitted confounding, and causal identification cannot be checked by these
diagnostics. No automatic robust covariance, transformations, or outlier removal.

Logistic inference is maximum likelihood with Wald coefficient/odds-ratio CIs
and an overall likelihood-ratio test against an intercept-only model. McFadden
pseudo-R² is not OLS R². AUC/accuracy are not included as apparent in-sample
performance claims. Separation detection is imperfect; large finite estimates
produce a note. Rescale extreme predictors and investigate design/sparse cells
when estimation fails. This release supports binary outcomes only.

## ANOVA

Partial eta-squared is an effect-size point estimate; no effect-size CI is
claimed. Group/cell mean CIs are raw pointwise t intervals, including in ANCOVA
and repeated reports; they are not adjusted means or within-subject contrast
intervals. OLS post hoc contrasts use equal-weight model marginal means.
The [ANOVA guide](anova.md) explains design-specific corrections, sums of squares,
comparison families, and specialized-method restrictions.

## Validation limits

Checked-in R 4.5.0 fixtures verify core numerical results independently; see
[the reference record](../tests/reference/README.md). Tests also exercise failure
paths and report/plot behavior. Passing fixtures is not a proof of correctness
for every dataset. Dunnett is also checked against R's independent mvtnorm
engine, with numerical integration/CI-inversion tolerance documented in the tests.
No claim of external academic peer review is made.

Python-version test results are recorded in [validation.md](validation.md).
Colab and Windows/macOS workflows must be reported as unverified until run;
installation compatibility must not be inferred merely from declared metadata.
