"""Numerical equivalence against independently generated, checked-in R output."""

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT, reference
from numpy.testing import assert_allclose

import rstats as rs


def ordered_r_coefficients(actual, expected):
    names = [
        name.replace("Intercept", "(Intercept)").replace("[T.", "").replace("]", "")
        for name in actual.index
    ]
    return expected.loc[names]


@pytest.mark.parametrize(
    "kind,formula", [("simple", "score ~ study_hours"), ("multiple", "score ~ study_hours + group")]
)
def test_lm_against_r(students, kind, formula):
    result = rs.lm(formula, data=students)
    actual = result.tables["coefficients"]
    expected = ordered_r_coefficients(actual, reference(f"lm_{kind}_coefficients.csv"))
    assert_allclose(actual[["estimate", "std_error", "t", "p"]], expected, rtol=1e-10, atol=1e-11)
    model = reference(f"lm_{kind}_model.csv")
    assert_allclose(result.tables["model"][model.columns], model, rtol=1e-10, atol=1e-11)


def test_logistic_against_r(students):
    result = rs.logmodel("passed ~ study_hours + group", data=students)
    # R terminates IRLS slightly earlier than statsmodels' Newton optimizer.
    assert_allclose(
        result.tables["coefficients"][["estimate", "std_error", "z", "p"]],
        ordered_r_coefficients(
            result.tables["coefficients"], reference("logistic_coefficients.csv")
        ),
        rtol=2e-5,
        atol=1e-6,
    )
    expected = reference("logistic_model.csv")
    assert_allclose(result.tables["model"][expected.columns], expected, rtol=1e-8)


def test_oneway_against_r(students):
    actual = rs.aov("score ~ group", data=students).tables["anova"].set_index("effect")
    expected = reference("oneway.csv")
    assert_allclose(
        actual.loc["group", ["sum_sq", "df_numerator", "F", "p"]].astype(float),
        expected.loc["group", ["Sum Sq", "Df", "F value", "Pr(>F)"]],
        rtol=1e-10,
    )


@pytest.mark.parametrize("ss_type", [1, 2, 3])
def test_anova_types_against_r(factorial, ss_type):
    actual = rs.aov("y ~ A * B", data=factorial, type=ss_type).tables["anova"].set_index("effect")
    expected = reference(f"type{ss_type}.csv")
    if ss_type == 2:
        expected = expected.set_index("effect")
        columns = ["sum_sq", "df", "F", "p"]
    else:
        columns = ["Sum Sq" if ss_type == 1 else "Sum of Sq", "Df", "F value", "Pr(>F)"]
    assert_allclose(
        actual.loc[["A", "B", "A:B"], ["sum_sq", "df_numerator", "F", "p"]].to_numpy(float),
        expected.loc[["A", "B", "A:B"], columns].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_ancova_against_r(students):
    actual = (
        rs.aov("score ~ group", data=students, covariates=["study_hours"])
        .tables["anova"]
        .set_index("effect")
    )
    expected = reference("ancova.csv")
    assert_allclose(
        actual.loc[["group", "study_hours"], ["sum_sq", "F", "p"]].to_numpy(float),
        expected.loc[["group", "study_hours"], ["Sum of Sq", "F value", "Pr(>F)"]].to_numpy(float),
        rtol=1e-10,
    )


def test_welch_against_r(students):
    actual = rs.aov("score ~ group", data=students, method="welch").tables["anova"]
    expected = reference("welch.csv")
    assert_allclose(actual[expected.columns], expected, rtol=1e-10)


def test_tukey_against_r(students):
    actual = rs.aov("score ~ group", data=students, posthocifsig=True).tables["posthoc"]
    expected = reference("tukey.csv")
    for row in actual.itertuples():
        ref = expected.loc[f"{row.group_b}-{row.group_a}"]
        assert_allclose(
            [row.estimate, row.ci_low, row.ci_high, row.p_adjusted],
            [-ref["diff"], -ref["upr"], -ref["lwr"], ref["p adj"]],
            atol=1e-9,
            rtol=1e-7,
        )


def test_games_howell_against_r(students):
    actual = rs.aov(
        "score ~ group", data=students, method="welch", posthocifsig="games-howell"
    ).tables["posthoc"]
    expected = reference("games_howell.csv")
    assert_allclose(
        actual[["estimate", "p_adjusted", "ci_low", "ci_high"]],
        expected[["estimate", "p_adjusted", "ci_low", "ci_high"]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "method,r_method",
    [("holm", "holm"), ("bonferroni", "bonferroni"), ("fdr-bh", "BH"), ("fdr-by", "BY")],
)
def test_ancova_comparisons_against_r(students, method, r_method):
    actual = rs.aov(
        "score ~ group", data=students, covariates=["study_hours"], posthocifsig=method
    ).tables["posthoc"]
    expected = reference(f"ancova_posthoc_{r_method}.csv")
    cols = ["estimate", "std_error", "p_raw", "p_adjusted"]
    assert_allclose(actual[cols], expected[cols], rtol=1e-9, atol=1e-10)
    assert set(actual.ci_coverage) == {"pointwise"}


@pytest.mark.parametrize("design", ["mixed", "repeated_one", "repeated_two"])
def test_repeated_against_r(longitudinal, design):
    if design == "repeated_two":
        data = pd.read_csv(ROOT / "tests/reference/repeated_two.csv")
        result = rs.aov(
            "score ~ time * measure", data=data, subject="student_id", within=["time", "measure"]
        )
        expected = reference("repeated_two.csv.expected")
    else:
        result = rs.aov(
            "score ~ time * group" if design == "mixed" else "score ~ time",
            data=longitudinal,
            subject="student_id",
            within="time",
        )
        expected = reference(f"{design}.csv")
    actual = result.tables["anova"].copy()
    actual["effect"] = actual.effect.replace(
        {"Interaction": "time:group", "time * measure": "time:measure"}
    )
    actual = actual.set_index("effect")
    expected = expected.set_index("effect")
    assert_allclose(
        actual.loc[
            expected.index, ["F", "p_uncorrected", "df_numerator", "df_denominator"]
        ].to_numpy(float),
        expected[["F", "p", "df_numerator", "df_denominator"]].to_numpy(float),
        rtol=1e-9,
        atol=1e-10,
    )
    epsilon = reference("gg_epsilon.csv").set_index("design").epsilon
    if design == "mixed":
        assert_allclose(actual.loc["time", "eps"], epsilon["mixed"], rtol=1e-10)
    elif design == "repeated_one":
        assert_allclose(actual.loc["time", "eps"], epsilon["repeated_one"], rtol=1e-10)
    else:
        assert_allclose(
            actual.loc[["time", "measure", "time:measure"], "eps"],
            epsilon[["two_time", "two_measure", "two_interaction"]],
            rtol=1e-9,
        )


def test_chisq_against_r(students):
    actual = rs.chisq("group ~ passed", data=students).tables["test"]
    expected = reference("chisq.csv")
    assert_allclose(actual[expected.columns], expected, rtol=1e-10)
    expected = reference("gof.csv")
    assert_allclose(
        rs.chisq([20, 30, 50], expected=[0.25, 0.25, 0.5]).tables["test"][expected.columns],
        expected,
        rtol=1e-10,
    )


def test_scheffe_bounds_are_conservative(students):
    result = rs.aov("score ~ group", data=students, posthocifsig="scheffe")
    table = result.tables["posthoc"]
    assert np.all(table.p_adjusted >= table.p_raw)
    assert np.all(table.ci_low < table.estimate)
    assert np.all(table.ci_high > table.estimate)


def test_dunnett_control(students):
    result = rs.aov("score ~ group", data=students, posthocifsig="dunnett", control="control")
    table = result.tables["posthoc"]
    assert len(table) == 2
    assert set(table.group_b) == {"control"}
    assert table.p_adjusted.between(0, 1).all()
    means = students.groupby("group").score.mean()
    assert_allclose(table.estimate, [means[a] - means["control"] for a in table.group_a])


@pytest.mark.parametrize("method", ["scheffe", "dunnett"])
def test_specialized_against_extended_r(students, method):
    kwargs = {"control": "control"} if method == "dunnett" else {}
    actual = rs.aov("score ~ group", data=students, posthocifsig=method, **kwargs).tables["posthoc"]
    expected = pd.read_csv(ROOT / f"tests/reference/{method}.csv")
    assert list(actual.group_a) == list(expected.group_a)
    assert list(actual.group_b) == list(expected.group_b)
    assert_allclose(actual.estimate, expected.estimate, atol=1e-10)
    # Dunnett's SciPy engine uses randomized multivariate-t integration, and its
    # CI inversion has a documented relative tolerance. mvtnorm is independent.
    tolerance = 2e-3 if method == "dunnett" else 1e-9
    assert_allclose(
        actual[["p_adjusted", "ci_low", "ci_high"]],
        expected[["p_adjusted", "ci_low", "ci_high"]],
        atol=tolerance,
        rtol=tolerance,
    )
