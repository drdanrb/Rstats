import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from pandas.testing import assert_frame_equal

import rstats as rs


@pytest.mark.parametrize(
    "formula,kwargs",
    [
        ("score ~ study_hours", {}),
        ("score ~ group", {}),
        ("score", {}),
        ("group", {}),
        ("group ~ passed", {}),
        ("score ~ group", {"type": "violin", "palette": "colorblind"}),
        ("score", {"type": "density"}),
        ("passed ~ group", {"type": "count"}),
        ("passed ~ group", {"type": "proportion", "normalize": "row"}),
        ("score ~ study_hours", {"type": "scatter", "hue": "group"}),
    ],
)
def test_plot_types_and_integrity(students, formula, kwargs, tmp_path):
    original = students.copy(deep=True)
    original_style = dict(mpl.rcParams)
    ax = rs.vis(formula, data=students, **kwargs)
    ax.figure.savefig(tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 100
    assert_frame_equal(students, original)
    assert dict(mpl.rcParams) == original_style
    plt.close(ax.figure)


def test_repeated_lines(longitudinal):
    for kwargs in [{"estimator": "mean", "hue": "group"}, {"units": "student_id", "hue": "group"}]:
        ax = rs.vis("score ~ time", data=longitudinal, type="line", **kwargs)
        assert ax.lines
        plt.close(ax.figure)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"type": "line"}, "require"),
        ({"type": "proportion"}, "normalize"),
        ({"type": "unknown"}, "type must"),
    ],
)
def test_ambiguous_plots_require_choices(students, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rs.vis("score ~ group", data=students, **kwargs)
    plt.close("all")
