# Rstats

**Familiar commands. Complete statistical reports.**

[Open the teaching notebook in Google Colab](https://colab.research.google.com/github/drdanrb/Rstats/blob/main/examples/quickstart.ipynb)

Rstats gives Python students and academics a short path from a pandas dataframe
to detailed, reusable statistical results. One analysis call includes the
tables, uncertainty, coding, sample accounting, and diagnostic notes that would
otherwise require several separate calls. Plot independently with `vis()`.

## Install

Python 3.12+; see CI for tested versions. This is an early research/teaching
release, not a package published on PyPI. **Do not install the unrelated
`rstats` PyPI package.** Install directly from GitHub:

```bash
python -m pip install git+https://github.com/drdanrb/Rstats.git
```

Or from the root of the supplied source checkout:

```bash
python -m pip install -e ".[dev]"
```

The distribution is named `rstats-python`; the import is `rstats`. No R runtime
is required to use the library. In Colab, prefix the installation command with
`%pip` instead of `python -m pip`, or open the teaching notebook above and choose
**Runtime → Run all**. The notebook installs Rstats if necessary and downloads
the synthetic example data automatically; no file uploads are needed.

## One-line analyses

```python
import pandas as pd
import rstats as rs

df = pd.read_csv("https://raw.githubusercontent.com/drdanrb/Rstats/main/examples/students.csv")

rs.descriptives(data=df, columns=["score", "study_hours"], by="group")
rs.table("group ~ passed", data=df, percent="row")
rs.chisq("group ~ passed", data=df)
rs.aov("score ~ group", data=df, posthocifsig=True)
rs.aov("score ~ group", data=df, covariates=["study_hours"])
rs.lm("score ~ study_hours + group", data=df)
rs.logmodel("passed ~ study_hours + group", data=df)
rs.vis("score ~ group", data=df, type="box")
```

In a notebook, leave a result as the cell's last expression to display its full
report. In a script use `print(result)` or `result.show()`.

```python
result = rs.lm("score ~ study_hours + group", data=df)
result.tables["coefficients"].to_csv("coefficients.csv")
result.model.predict(df)  # underlying statsmodels model
result.show()
```

## Small rules that keep things intuitive

- Formulas are strings: `"outcome ~ predictor + another"`.
- `+` adds terms; `*` includes main effects and interactions; `:` adds an
  interaction alone. Backticks quote column names with spaces.
- In **ANOVA**, predictors are groups even if coded 1, 2, 3. Put continuous
  adjustments in `covariates=["age"]`.
- In **regression**, numbers are continuous and text/category columns are
  categorical. Use `categorical=["group"]` for numeric-coded categories.
- Set a regression reference with `reference={"group": "control"}`. Otherwise
  the first declared pandas category, or sorted observed level, is the reference.
- `logmodel` models numeric outcome 1 by default. For text outcomes use
  `event="passed"`; the report states exactly which event is modeled.
- `posthocifsig=True` means **Tukey**, not automatic method selection.
  Use `posthocifsig="games-howell"` after Welch or `"holm"` for factorial,
  covariate-adjusted, or repeated/mixed comparisons. Incompatible requests explain
  the problem rather than silently changing the method.

## ANOVA without guessing

```python
rs.aov("score ~ group * teaching", data=df, type=3, posthocifsig="holm")
rs.aov("score ~ group", data=df, method="welch", posthocifsig="games-howell")
rs.aov("score ~ time * group", data=long_df,
       subject="student_id", within="time", posthocifsig="holm")
```

Study design, sums-of-squares **type**, and variance **method** are different
choices. Ordinary ANOVA defaults to Type II; Type I is sequential and Type III
uses sum contrasts. Repeated/mixed designs use effect-specific tests and report
Greenhouse–Geisser inference where applicable. They do not get an invented
overall model F. See the [ANOVA and post hoc guide](docs/anova.md).

## Independent plotting

```python
rs.vis("score ~ study_hours", data=df)  # numeric/numeric → scatter
rs.vis("score ~ group", data=df)        # numeric/category → box
rs.vis("score", data=df)                # numeric → histogram
rs.vis("group", data=df)                # category → counts
rs.vis("score ~ group", data=df, type="violin", palette="colorblind")
rs.vis("passed ~ group", data=df, type="proportion", normalize="row")
rs.vis("score ~ time", data=long_df, type="line", estimator="mean")
```

Each plot returns a Matplotlib Axes: `ax.figure.savefig("figure.png")`.
No global Seaborn theme is changed. Plot data are complete cases for the plotted
variables; analytic sample counts remain in statistical reports.

## Documentation and teaching

- [API reference](docs/api.md)
- [ANOVA types, assumptions, and comparison methods](docs/anova.md)
- [Statistical conventions and limitations](docs/statistics.md)
- [Five-slide PowerPoint](docs/Rstats-quickstart.pptx)
- [Teaching notebook](examples/quickstart.ipynb)
- [Example ANOVA report](examples/anova-report.html) and [regression report](examples/regression-report.html)
- [Validation and R reference fixtures](tests/reference/README.md)

The included datasets are deterministic synthetic teaching data, not research
evidence. Tests are two-sided unless a method states otherwise. Missingness,
study design, independence, model assumptions, and causal interpretation remain
the analyst's responsibility; reports surface relevant details and never switch
methods based on assumption tests.

## Development

```bash
python -m pytest
python -m ruff check src tests tools
python -m build
```

Python-version CI installs and tests on 3.12–3.15; platform/Colab smoke checks are
tracked separately. Do not infer passing environments from a workflow definition
alone. MANOVA/MANCOVA, arbitrary random effects, repeated-measures ANCOVA,
multinomial/ordinal logistic models, and SEM are outside this release.
