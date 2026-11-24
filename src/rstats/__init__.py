"""Rstats: familiar commands, complete statistical reports."""

from .anova import aov
from .categorical import chisq, table
from .descriptive import descriptives
from .regression import lm, logmodel
from .result import AnalysisResult
from .visualization import vis

__version__ = "0.1.0"
__all__ = ["AnalysisResult", "aov", "chisq", "descriptives", "lm", "logmodel", "table", "vis"]
