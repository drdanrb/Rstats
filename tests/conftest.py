from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def students():
    return pd.read_csv(ROOT / "examples/students.csv")


@pytest.fixture
def factorial():
    return pd.read_csv(ROOT / "tests/reference/factorial.csv")


@pytest.fixture
def longitudinal():
    return pd.read_csv(ROOT / "examples/longitudinal.csv")


def reference(name):
    return pd.read_csv(ROOT / "tests/reference" / name, index_col=0)
