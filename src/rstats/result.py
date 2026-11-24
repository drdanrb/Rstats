"""A single result interface for notebooks and scripts."""

from dataclasses import dataclass, field
from html import escape
from typing import Any

import pandas as pd


def _display_table(frame):
    out = frame.copy()
    for col in out:
        if pd.api.types.is_numeric_dtype(out[col]):
            if str(col).startswith("p_") or col == "p":
                out[col] = out[col].map(
                    lambda v: "—" if pd.isna(v) else "< .001" if v < 0.001 else f"{v:.3f}"
                )
            elif not pd.api.types.is_integer_dtype(out[col]):
                out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    return out


@dataclass
class AnalysisResult:
    """Reusable full-precision tables and a rich notebook representation.

    ``model`` is the underlying statsmodels result when available. Diagnostics
    are explanatory notes, not automatic changes to the requested analysis.
    """

    title: str
    tables: dict[str, pd.DataFrame]
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: Any = field(default=None, repr=False)

    def __str__(self):
        parts = [self.title]
        for name, frame in self.tables.items():
            parts.extend(["", name.replace("_", " ").title(), _display_table(frame).to_string()])
        if self.diagnostics:
            parts.extend(["", "Notes", *[f"• {note}" for note in self.diagnostics]])
        return "\n".join(parts)

    def _repr_html_(self):
        parts = [f"<section><h2>{escape(self.title)}</h2>"]
        for name, frame in self.tables.items():
            parts.append(f"<h3>{escape(name.replace('_', ' ').title())}</h3>")
            parts.append(_display_table(frame).to_html(escape=True, border=0))
        if self.diagnostics:
            parts.append("<h3>Notes</h3><ul>")
            parts.extend(f"<li>{escape(str(note))}</li>" for note in self.diagnostics)
            parts.append("</ul>")
        return "".join(parts) + "</section>"

    def show(self):
        """Display this result in a notebook, or print it in a script."""
        try:
            from IPython import get_ipython
            from IPython.display import display

            shell = get_ipython()
            if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
                display(self)
                return
        except ImportError:
            pass
        print(self)
