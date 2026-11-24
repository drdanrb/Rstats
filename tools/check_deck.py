"""Validate slide boundaries and execute every displayed teaching example."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
deck = Presentation(ROOT / "docs/Rstats-quickstart.pptx")
assert len(deck.slides) == 5
context = {}
previous = Path.cwd()
try:
    os.chdir(ROOT / "examples")
    for slide_number, slide in enumerate(deck.slides, 1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= deck.slide_width
            assert shape.top + shape.height <= deck.slide_height
            if shape.has_text_frame and any(
                run.font.name == "Consolas"
                for paragraph in shape.text_frame.paragraphs
                for run in paragraph.runs
            ):
                exec(compile(shape.text, f"slide-{slide_number}", "exec"), context)  # noqa: S102 -- repository-authored teaching examples
    print("All five slides are within bounds; every displayed Python example executes.")
finally:
    os.chdir(previous)
    plt.close("all")
