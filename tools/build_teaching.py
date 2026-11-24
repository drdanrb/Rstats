"""Regenerate synthetic datasets, the executable notebook, and five-slide deck."""

from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]


def datasets():
    rng = np.random.default_rng(20261124)
    n = 150
    group = np.repeat(["control", "practice", "tutoring"], n // 3)
    hours = np.round(rng.uniform(1, 9, n), 2)
    age = rng.integers(18, 30, n)
    effect = np.repeat([0, 6, 12], n // 3)
    score = np.round(40 + 3 * hours + effect + rng.normal(0, 8, n), 2)
    probability = 1 / (1 + np.exp(-(-3 + 0.48 * hours + effect / 12)))
    students = pd.DataFrame(
        {
            "student_id": np.arange(1, n + 1),
            "group": group,
            "study_hours": hours,
            "age": age,
            "score": score,
            "passed": rng.binomial(1, probability),
        }
    )
    students.to_csv(ROOT / "examples/students.csv", index=False)
    rows = []
    for sid in range(1, 31):
        subject_offset = rng.normal(0, 5)
        g = "control" if sid <= 15 else "practice"
        for time_index, time in enumerate(["baseline", "midterm", "final"]):
            rows.append(
                {
                    "student_id": sid,
                    "group": g,
                    "time": time,
                    "score": round(
                        50
                        + subject_offset
                        + 3 * time_index
                        + (4 * time_index if g == "practice" else 0)
                        + rng.normal(0, 3),
                        3,
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "examples/longitudinal.csv", index=False)


def notebook():
    cells = [
        nbf.v4.new_markdown_cell(
            "# Rstats: one line, a full report\n\nSynthetic teaching data. Install this checkout with `pip install -e .`. In Colab, install the repository after it is available on GitHub, then upload the two example CSVs."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\nfrom pathlib import Path\nimport pandas as pd\nimport rstats as rs\n\nexamples = Path('examples') if Path('examples/students.csv').exists() else Path('.')\ndf = pd.read_csv(examples / 'students.csv')\nlong_df = pd.read_csv(examples / 'longitudinal.csv')\nlong_df['time'] = pd.Categorical(long_df['time'], categories=['baseline', 'midterm', 'final'], ordered=True)\ndf.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Describe and tabulate\nMissingness and denominators are part of the report."
        ),
        nbf.v4.new_code_cell(
            "rs.descriptives(data=df, columns=['score', 'study_hours'], by='group')"
        ),
        nbf.v4.new_code_cell("rs.table('group ~ passed', data=df, percent='row')"),
        nbf.v4.new_markdown_cell(
            "## 2. Categorical association\nLook at expected cell counts as well as the p value."
        ),
        nbf.v4.new_code_cell("rs.chisq('group ~ passed', data=df)"),
        nbf.v4.new_markdown_cell(
            "## 3. ANOVA and ANCOVA\nANOVA predictors are groups. Covariates are continuous adjustments. `True` requests Tukey; other methods are named explicitly."
        ),
        nbf.v4.new_code_cell("rs.aov('score ~ group', data=df, posthocifsig=True)"),
        nbf.v4.new_code_cell(
            "rs.aov('score ~ group', data=df, covariates=['study_hours'], posthocifsig='holm')"
        ),
        nbf.v4.new_code_cell(
            "rs.aov('score ~ group', data=df, method='welch', posthocifsig='games-howell')"
        ),
        nbf.v4.new_code_cell(
            "rs.aov('score ~ time * group', data=long_df, subject='student_id', within='time', posthocifsig='holm')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Linear and logistic models\nText groups are categorical automatically. Numeric-coded groups can use `categorical=['group']`. Logistic output identifies the modeled event."
        ),
        nbf.v4.new_code_cell("rs.lm('score ~ study_hours + group', data=df)"),
        nbf.v4.new_code_cell("rs.logmodel('passed ~ study_hours + group', data=df)"),
        nbf.v4.new_markdown_cell(
            "## 5. Plot independently\nThese commands return Matplotlib axes; customize labels or save with `ax.figure.savefig(...)`."
        ),
        nbf.v4.new_code_cell("rs.vis('score ~ study_hours', data=df, hue='group')"),
        nbf.v4.new_code_cell("rs.vis('score ~ group', data=df, type='box')"),
        nbf.v4.new_code_cell(
            "rs.vis('score ~ time', data=long_df, type='line', estimator='mean', hue='group')"
        ),
        nbf.v4.new_markdown_cell(
            "## Reuse the actual tables\nFormatting never discards numerical precision. A significant association alone does not establish causation."
        ),
        nbf.v4.new_code_cell(
            "result = rs.lm('score ~ study_hours + group', data=df)\nresult.tables['coefficients']"
        ),
    ]
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    )
    nbf.write(nb, ROOT / "examples/quickstart.ipynb")


def deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slides = [
        (
            "Rstats",
            "Familiar commands. Complete statistical reports.",
            "import pandas as pd\nimport rstats as rs\ndf = pd.read_csv('students.csv')",
            "One call → readable tables, uncertainty, sample counts, and diagnostic notes.\nFormulas: outcome ~ predictor + another\nUse * to include an interaction.",
        ),
        (
            "01 / Understand your data",
            "Descriptives and percentages in one call",
            "rs.descriptives(data=df, columns=['score'], by='group')\n\nrs.table('group ~ passed', data=df, percent='row')",
            "Read n, missing values, mean/median, SD, and confidence intervals.\nCheck which denominator the percentages use.\nCSV loading stays with pandas.",
        ),
        (
            "02 / Test categorical association",
            "Chi-square with observed and expected counts",
            "rs.chisq('group ~ passed', data=df)\n\nrs.chisq([20, 30, 50], expected=[.25, .25, .5])",
            "Read χ², degrees of freedom, p, and Cramér’s V.\nInspect sparse expected cells before relying on asymptotic p values.\nFor a 2 × 2 table, method='fisher' explicitly selects Fisher’s exact test.",
        ),
        (
            "03 / Compare group means",
            "ANOVA, comparisons, and a separate plot",
            "rs.aov('score ~ group', data=df, posthocifsig=True)\n\nrs.aov('score ~ group', data=df, covariates=['study_hours'])\n\nrs.vis('score ~ group', data=df, type='box')",
            "True = Tukey after a significant effect; name another method explicitly.\nWelch → posthocifsig='games-howell'; adjusted/repeated → 'holm'.\nANOVA predictors are groups; covariates are continuous adjustments.",
        ),
        (
            "04 / Explain or predict an outcome",
            "lm for continuous outcomes · logmodel for binary outcomes",
            "rs.lm('score ~ study_hours + group', data=df)\n\nrs.logmodel('passed ~ study_hours + group', data=df)\n\nrs.vis('score ~ study_hours', data=df, type='scatter')",
            "Read the overall model test, coefficients, confidence intervals, and reference groups.\nNumeric-coded groups? Add categorical=['group'].\nBinary text outcome? Add event='passed'. Association does not establish causation.",
        ),
    ]
    for index, (heading, subtitle, code, notes) in enumerate(slides):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string("F4F6F8")

        def textbox(x, y, w, h, text, size, color, font="Aptos", slide=slide):
            shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tf = shape.text_frame
            tf.word_wrap = True
            for i, line in enumerate(text.splitlines()):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = line
                p.font.name, p.font.size, p.font.color.rgb = (
                    font,
                    Pt(size),
                    RGBColor.from_string(color),
                )
                p.space_after = Pt(10)
            return shape

        textbox(0.65, 0.45, 12, 0.7, heading, 34, "142D4E")
        textbox(0.7, 1.25, 12, 0.6, subtitle, 21, "47627E")
        textbox(0.7, 2.15, 12, 2.5, code, 20, "135D66", "Consolas")
        textbox(0.7, 5, 11.8, 1.9, notes, 18, "25364A")
        textbox(12.1, 6.95, 0.6, 0.3, f"{index + 1}/5", 11, "47627E")
        slide.notes_slide.notes_text_frame.text = "Synthetic teaching data. See docs/anova.md and docs/statistics.md for assumptions and supported designs."
    prs.save(ROOT / "docs/Rstats-quickstart.pptx")


def reports():
    import rstats as rs

    data = pd.read_csv(ROOT / "examples/students.csv")
    calls = {
        "anova": (
            "rs.aov('score ~ group', data=df, posthocifsig=True)",
            rs.aov("score ~ group", data=data, posthocifsig=True),
        ),
        "regression": (
            "rs.lm('score ~ study_hours + group', data=df)",
            rs.lm("score ~ study_hours + group", data=data),
        ),
    }
    for label, (command, result) in calls.items():
        content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Rstats — {label} example</title>
<style>body{{font:16px system-ui,sans-serif;color:#183047;max-width:1120px;margin:40px auto;padding:0 20px}}h1,h2,h3{{color:#164e63}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:9px 14px;text-align:right;border-bottom:1px solid #dae2e8}}th{{background:#eef4f6}}li{{margin:12px 0;line-height:1.5}}pre{{padding:20px;background:#eef4f6;border-radius:8px;overflow:auto}}</style>
<h1>Rstats: one command, a complete report</h1>
<p>Synthetic teaching data from students.csv. This report is static; tables remain full-precision dataframes in Python.</p>
<pre>{command}</pre>{result._repr_html_()}</html>"""
        (ROOT / f"examples/{label}-report.html").write_text(content)


if __name__ == "__main__":
    datasets()
    notebook()
    deck()
    reports()
