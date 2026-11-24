# ANOVA designs, types, and post hoc comparisons

## Choose the study design first

| Design | One-line call |
| --- | --- |
| Independent one-way | `rs.aov("score ~ group", data=df)` |
| Factorial | `rs.aov("score ~ group * method", data=df)` |
| ANCOVA | `rs.aov("score ~ group", data=df, covariates=["age"])` |
| Unequal-variance one-way | `rs.aov("score ~ group", data=df, method="welch")` |
| Repeated | `rs.aov("score ~ time", data=df, subject="id", within="time")` |
| Two within factors | `rs.aov("score ~ time * task", data=df, subject="id", within=["time", "task"])` |
| Mixed | `rs.aov("score ~ time * group", data=df, subject="id", within="time")` |

Predictors in ANOVA are grouping variables. Numeric group codes do not need
`C()`/`factor()`. Continuous covariates belong in `covariates=`, not among the
formula's groups. ANCOVA here assumes common covariate slopes; covariate-by-group
interactions and repeated-measures ANCOVA are not supported.

Repeated data must have one row per subject-condition combination. Within-factor
cells must form the complete Cartesian product. Incomplete subjects are excluded
and counted, including implicitly absent rows. Replicate cells must be
deliberately aggregated by the analyst first. A between factor must remain
constant within each subject. Arbitrary random effects and nested designs are
not covered by these commands.

## Sums of squares are a different choice

| `type` | Meaning | When to consider it |
| --- | --- | --- |
| `1` | Sequential: each term adjusts for preceding terms | Prespecified hierarchical questions; order matters |
| `2` (ordinary default) | Each main effect adjusts for other terms except interactions containing it | Additive models; interpret main effects carefully if interactions exist |
| `3` | Each term adjusts for all other model terms | Explicit full-model hypotheses, including interactions; uses sum contrasts |

There is no universally correct sums-of-squares type. In balanced additive
designs they often agree; in unbalanced designs they answer different questions.
Type II here intentionally differs from R `anova(lm(...))`'s sequential default.
Use `type=1` when matching that output. Python's formula engine may reorder main
effects ahead of interactions; inspect the returned table to verify the tested
sequence. Type III uses sum contrasts so hypotheses do not accidentally become
reference-cell tests. Empty-cell/rank-deficient designs are rejected.

Do not supply `type=` to Welch or repeated/mixed ANOVA. Welch uses a different
variance/df calculation. Repeated/mixed designs use effect-specific error strata.
The mixed backend uses Type II conventions when between-subject group sizes
differ, which can differ from Type III defaults in other software.

Ordinary models include overall F, p, and R². The omnibus and effect-specific
tables are separate. Repeated/mixed designs do not have a single artificial
overall F. Corrected df and p values accompany the uncorrected values;
Greenhouse–Geisser is primary where available unless `correction="none"`.

## Request comparisons explicitly

```python
rs.aov("score ~ group", data=df, posthocifsig=True)  # Tukey
rs.aov("score ~ group", data=df, method="welch", posthocifsig="games-howell")
rs.aov("score ~ group", data=df, posthocifsig="dunnett", control="control")
rs.aov("score ~ group", data=df, covariates=["age"], posthocifsig="holm")
```

`False` is the default. `True` means Tukey, not an automatically selected method.
A named procedure runs only when its corresponding omnibus effect has primary
`p < alpha`. This gate is a convenience, not an additional global multiplicity
guarantee. Invalid method/design requests raise an error even if the omnibus
test would not have been significant.

| Method | Purpose / error control | Supported context |
| --- | --- | --- |
| `tukey` | All pairwise comparisons; familywise; Tukey–Kramer for unequal n | One independent factor, common variance, no covariates |
| `games-howell` | All pairwise comparisons with unequal variances and approximate df | One independent factor; the supported follow-up to Welch |
| `dunnett` | Each treatment against explicit `control=`; familywise | One independent factor, common variance, no covariates |
| `scheffe` | Conservative familywise coverage over arbitrary contrasts; reports pairwise contrasts | One independent factor, common variance, no covariates |
| `holm` | Step-down Bonferroni; familywise under arbitrary dependence | Ordinary model-adjusted or subject-aware comparisons |
| `bonferroni` | Single-step conservative familywise adjustment | Same |
| `sidak` | Single-step familywise adjustment under independence/suitable dependence | Same; correlated contrasts require care |
| `holm-sidak` | Step-down Šidák adjustment | Same dependence caveat |
| `fdr-bh` | Exploratory Benjamini–Hochberg false discovery rate | Independent/appropriate positively dependent comparison families |
| `fdr-by` | Exploratory Benjamini–Yekutieli false discovery rate | Arbitrary dependence; typically more conservative |

Holm/Bonferroni/Šidák/FDR are **adjustments**, not standalone tests. Reports
identify both the underlying test and adjustment. Welch currently permits only
Games–Howell as an automatic post hoc follow-up.

### Which comparisons are generated?

- A significant main effect generates all its marginal pairwise comparisons.
- A significant interaction generates simple-effect comparisons for each factor,
  conditioning on every level combination of the other interacting factors.
- Each triggering effect is a separate adjustment family. For an interaction,
  **all** generated simple effects belong to that family, not separate small
  families per slice. Distinct omnibus effects are not jointly adjusted.
- OLS comparisons use fitted marginal-mean contrasts, equal weighting across
  other factor levels, covariates at analyzed-sample means, and model covariance
  and residual df. This can differ from raw means in unbalanced/adjusted models.
- Repeated/mixed follow-ups first average over remaining repeated factors within
  each subject. Within-subject contrasts use paired t tests; between-subject
  contrasts use Welch t tests on subject means. These are labeled explicitly
  rather than presented as identical to the ANOVA's pooled-error contrasts.

`estimate` is group A minus group B. Specialized procedures return simultaneous
intervals. General adjusted t comparisons return **pointwise** CIs alongside
adjusted p values; the CI and adjusted significance decision need not agree.
No significance p value is attached to a raw group mean itself.

### Assumptions and boundaries

Independence is a study-design assumption. Ordinary ANOVA assumes appropriate
residual behavior and common variance; ANCOVA additionally requires a justified
linear adjustment and common slopes. Welch addresses variance heterogeneity,
not dependence or arbitrary distributional problems. Repeated inference depends
on its covariance assumptions; GG correction does not repair all violations.
Small samples, selection, missingness, and outliers require scientific judgment.

Read the group descriptives, diagnostics, effect sizes, and intervals alongside
p values. Diagnostics never silently switch the requested method.

Engine references: [statsmodels ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html),
[Pingouin repeated ANOVA](https://pingouin-stats.org/generated/pingouin.rm_anova.html),
[Pingouin mixed ANOVA](https://pingouin-stats.org/generated/pingouin.mixed_anova.html),
[SciPy Tukey/Games–Howell](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html),
[SciPy Dunnett](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dunnett.html),
[statsmodels adjustments](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
