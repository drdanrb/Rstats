import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

import rstats as rs


def test_full_precision_display_escape_and_input_integrity(students):
    original = students.copy(deep=True)
    result = rs.lm("score ~ study_hours + group", data=students)
    before = result.tables["coefficients"].copy(deep=True)
    result.title = "<script>unsafe</script>"
    result.diagnostics.append("<b>literal text</b>")
    html = result._repr_html_()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;b&gt;literal text" in html
    assert "Coefficients" in str(result)
    assert_frame_equal(result.tables["coefficients"], before)
    rs.aov("score ~ group", data=students, posthocifsig="holm")
    rs.logmodel("passed ~ study_hours + group", data=students)
    assert_frame_equal(students, original)


def test_numeric_categorical_and_explicit_reference(students):
    numbered = students.assign(
        group=students.group.map({"control": 1, "practice": 2, "tutoring": 3})
    )
    result = rs.lm(
        "score ~ study_hours + group", data=numbered, categorical=["group"], reference={"group": 2}
    )
    assert result.metadata["coding"]["group"]["reference"] == 2
    assert len(result.tables["coefficients"]) == 4
    implicit = rs.aov("score ~ group", data=numbered)
    text = rs.aov("score ~ group", data=students)
    assert_allclose(implicit.tables["anova"].F, text.tables["anova"].F, equal_nan=True)
    forced = rs.lm("score ~ study_hours + factor(group)", data=numbered)
    assert len(forced.tables["coefficients"]) == 4


def test_declared_category_order_and_unused_levels(students):
    students["group"] = pd.Categorical(
        students.group, categories=["tutoring", "practice", "control", "absent"]
    )
    result = rs.lm("score ~ group", data=students)
    assert result.metadata["coding"]["group"]["reference"] == "tutoring"
    assert len(result.tables["coefficients"]) == 3


def test_backticks_and_internal_token_names(students):
    df = students.rename(columns={"score": "exam score", "study_hours": "v1", "group": "v0"})
    result = rs.lm("`exam score` ~ v1 + v0", data=df)
    assert len(result.tables["coefficients"]) == 4


@pytest.mark.parametrize(
    "bad",
    [
        "y ~ __import__('os').system('false')",
        "y ~ np.log(x)",
        "y ~ x ** 2",
        "y ~ x - z",
        "y ~ x / z",
    ],
)
def test_formula_language_is_limited(bad, students):
    with pytest.raises(ValueError):
        rs.lm(bad, data=students)


def test_missing_case_accounting(students):
    students.loc[0, "score"] = np.nan
    result = rs.lm("score ~ study_hours", data=students)
    assert result.tables["sample"].iloc[0].excluded_rows == 1
    with pytest.raises(ValueError, match="missing"):
        rs.lm("score ~ study_hours", data=students, missing="raise")


def test_incomplete_repeated_subject_explicit_and_implicit(longitudinal):
    explicit = longitudinal.copy()
    explicit.loc[0, "score"] = np.nan
    implicit = longitudinal.drop(index=0)
    for data in [explicit, implicit]:
        result = rs.aov("score ~ time * group", data=data, subject="student_id", within="time")
        assert result.tables["subjects"].iloc[0].analyzed_subjects == 29
        assert result.tables["subjects"].iloc[0].excluded_subjects == 1
        with pytest.raises(ValueError, match="Incomplete"):
            rs.aov(
                "score ~ time * group",
                data=data,
                subject="student_id",
                within="time",
                missing="raise",
            )


def test_repeated_duplicate_and_changing_group(longitudinal):
    duplicated = pd.concat([longitudinal, longitudinal.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        rs.aov("score ~ time * group", data=duplicated, subject="student_id", within="time")
    longitudinal.loc[0, "group"] = "practice"
    with pytest.raises(ValueError, match="changes"):
        rs.aov("score ~ time * group", data=longitudinal, subject="student_id", within="time")


def test_missing_between_value_excludes_subject(longitudinal):
    longitudinal.loc[0, "group"] = None
    result = rs.aov("score ~ time * group", data=longitudinal, subject="student_id", within="time")
    assert result.tables["subjects"].iloc[0].excluded_subjects == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "welch", "posthocifsig": True}, "equal variances"),
        ({"method": "welch", "type": 3}, "does not use"),
        ({"type": 4}, "type must"),
        ({"posthocifsig": "unknown"}, "posthocifsig"),
        ({"posthocifsig": "dunnett"}, "control"),
        ({"control": "control"}, "control"),
        ({"covariates": ["group"]}, "separate"),
    ],
)
def test_anova_invalid_combinations(students, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rs.aov("score ~ group", data=students, **kwargs)


def test_tukey_does_not_ignore_covariates(students):
    with pytest.raises(ValueError, match="without covariates"):
        rs.aov("score ~ group", data=students, covariates=["study_hours"], posthocifsig=True)


def test_no_posthoc_for_nonsignificant_effect():
    df = pd.DataFrame({"y": [1, 2, 3, 1, 2, 3], "g": [1] * 3 + [2] * 3})
    result = rs.aov("y ~ g", data=df, posthocifsig=True)
    assert result.tables["posthoc"].empty
    assert any("No relevant omnibus" in note for note in result.diagnostics)


@pytest.mark.parametrize(
    "method", ["holm", "bonferroni", "sidak", "holm-sidak", "fdr-bh", "fdr-by"]
)
def test_factorial_comparison_families(factorial, method):
    factorial["y"] += 8 * factorial.A.map({"a": 0, "b": 1, "c": 2}) * (factorial.B == "yes")
    result = rs.aov("y ~ A * B", data=factorial, posthocifsig=method)
    table = result.tables["posthoc"]
    assert table.p_adjusted.between(0, 1).all()
    assert "A:B" in set(table.family)
    interaction = table.loc[table.family == "A:B"]
    assert interaction.condition.ne("marginal").all()
    # A pairwise in each B stratum (6), B pairwise in each A stratum (3).
    assert len(interaction) == 9


def test_repeated_comparisons_respect_pairing(longitudinal):
    result = rs.aov(
        "score ~ time * group",
        data=longitudinal,
        subject="student_id",
        within="time",
        posthocifsig="holm",
    )
    posthoc = result.tables["posthoc"]
    assert set(posthoc.loc[posthoc.target == "time", "test"]) == {"Paired subject-mean t test"}
    assert set(posthoc.loc[posthoc.target == "group", "test"]) == {"Welch subject-mean t test"}
    assert all(posthoc.loc[posthoc.target == "time", "df"] >= 14)


def test_invalid_repeated_formula_is_not_silently_expanded(longitudinal):
    with pytest.raises(ValueError, match="full factorial"):
        rs.aov("score ~ time + group", data=longitudinal, subject="student_id", within="time")


def test_logistic_event(students):
    named = students.assign(passed=students.passed.map({0: "no", 1: "yes"}))
    with pytest.raises(ValueError, match="event="):
        rs.logmodel("passed ~ study_hours", data=named)
    result = rs.logmodel("passed ~ study_hours", data=named, event="yes")
    other = rs.logmodel("passed ~ study_hours", data=named, event="no")
    assert_allclose(result.model.params, -other.model.params)
    assert result.metadata["event"] == "yes"


def test_separation_and_rank_deficiency():
    df = pd.DataFrame({"y": [0] * 10 + [1] * 10, "x": np.arange(20)})
    with pytest.raises(ValueError, match="separation|converge"):
        rs.logmodel("y ~ x", data=df)
    df["x2"] = df.x * 2
    with pytest.raises(ValueError, match="rank deficient"):
        rs.lm("y ~ x + x2", data=df)


def test_descriptives_missing_constant_and_categories():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, np.nan],
            "constant": [1] * 4,
            "empty": [np.nan] * 4,
            "group": ["a", "a", "b", None],
        }
    )
    result = rs.descriptives(data=df)
    numeric = result.tables["descriptives"].set_index("variable")
    assert numeric.loc["x", "mean"] == 2
    assert numeric.loc["x", "sd"] == 1
    assert numeric.loc["x", "missing"] == 1
    assert numeric.loc["empty", "n"] == 0
    assert np.isnan(numeric.loc["constant", "skewness"])
    frequencies = result.tables["frequencies"]
    assert_allclose(frequencies.percent.sum(), 100)


def test_frequency_denominators_and_fisher():
    df = pd.DataFrame({"a": ["a", "a", "b", "b", "b", None], "b": [0, 1, 0, 0, 1, 0]})
    row = rs.table("a ~ b", data=df, percent="row")
    assert_allclose(row.tables["percentages"].sum(axis=1), 100)
    col = rs.table("a ~ b", data=df, percent="column")
    assert_allclose(col.tables["percentages"].sum(axis=0), 100)
    result = rs.chisq("a ~ b", data=df, method="fisher")
    assert result.tables["test"].iloc[0].p == 1
    assert any("Sparse" in note for note in result.diagnostics)


@pytest.mark.parametrize(
    "observed,expected", [([-1, 2], None), ([1.5, 2], None), ([1, 2], [0.2, 0.2]), ([0, 0], None)]
)
def test_bad_goodness_of_fit(observed, expected):
    with pytest.raises(ValueError):
        rs.chisq(observed, expected=expected)


def test_perfect_fit_and_unsupported_welch_followup(students):
    df = pd.DataFrame({"y": [1, 3, 5, 7, 9], "x": [0, 1, 2, 3, 4]})
    with pytest.raises(ValueError, match="zero residual variance"):
        rs.lm("y ~ x", data=df)
    with pytest.raises(ValueError, match="games-howell"):
        rs.aov("score ~ group", data=students, method="welch", posthocifsig="holm")


def test_games_howell_requires_replication():
    df = pd.DataFrame({"y": [1, 2, 4, 8], "g": ["a", "b", "b", "b"]})
    with pytest.raises(ValueError, match="two observations"):
        rs.aov("y ~ g", data=df, posthocifsig="games-howell")
