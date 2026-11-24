"""Standalone Seaborn plots with type-aware defaults and no global styling."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._data import complete, names, numeric, parse


def vis(
    formula,
    *,
    data,
    type=None,
    categorical=None,
    hue=None,
    palette=None,
    colors="default",
    xlabel=None,
    ylabel=None,
    title=None,
    estimator=None,
    normalize=None,
    units=None,
    ax=None,
    **kwargs,
):
    """Create a standalone plot and return its Matplotlib Axes.

    ``vis('score ~ age', data=df)`` defaults to scatter. ``vis('score', data=df)``
    defaults to histogram. ``line`` needs estimator='mean'/'median' or units=;
    ``proportion`` needs normalize='row'/'column'/'total'. Seaborn keyword
    options pass through after the shared arguments.
    """
    if isinstance(formula, str) and "~" not in formula:
        x, y = formula.strip().strip("`"), None
    else:
        spec = parse(formula)
        if len(spec.predictors) != 1 or spec.expression.strip() != next(iter(spec.tokens), ""):
            raise ValueError(
                "For a two-variable plot use 'outcome ~ predictor' without interactions."
            )
        y, x = spec.outcome, spec.predictors[0]
    columns = list(dict.fromkeys([c for c in [x, y, hue, units] if c is not None]))
    frame, _ = complete(data, columns)
    marked = names(categorical)
    if set(marked) - set(columns):
        raise ValueError("categorical must name variables used in the plot.")
    for col in marked:
        frame[col] = frame[col].astype("category")
    is_numeric = lambda col: (
        col is not None
        and pd.api.types.is_numeric_dtype(frame[col])
        and not pd.api.types.is_bool_dtype(frame[col])
    )
    if type is None:
        type = (
            ("hist" if is_numeric(x) else "count")
            if y is None
            else (
                "scatter"
                if is_numeric(x) and is_numeric(y)
                else "box"
                if is_numeric(x) != is_numeric(y)
                else "heatmap"
            )
        )
    allowed = {
        "scatter",
        "box",
        "violin",
        "hist",
        "density",
        "count",
        "proportion",
        "heatmap",
        "line",
    }
    if type not in allowed:
        raise ValueError(f"type must be one of {sorted(allowed)}.")
    if palette is not None and colors != "default":
        raise ValueError("Choose palette= or colors=, not both.")
    selected_palette = (
        palette if palette is not None else "colorblind" if colors == "default" else colors
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    style = (
        {"palette": selected_palette}
        if hue is not None
        else {"color": sns.color_palette(selected_palette)[0]}
    )
    common = {"data": frame, "ax": ax, **style}
    if hue is not None:
        common["hue"] = hue
    if type == "scatter":
        if y is None:
            raise ValueError("Scatter plots need 'y ~ x'.")
        numeric(frame[x], x)
        numeric(frame[y], y)
        sns.scatterplot(x=x, y=y, **common, **kwargs)
    elif type in {"box", "violin"}:
        if y is None or not (is_numeric(x) or is_numeric(y)):
            raise ValueError("Box/violin plots need a numeric outcome and grouping variable.")
        plot = sns.boxplot if type == "box" else sns.violinplot
        if is_numeric(y):
            frame[x] = frame[x].astype("category")
            plot(x=x, y=y, **common, **kwargs)
        else:
            frame[y] = frame[y].astype("category")
            plot(x=x, y=y, orient="h", **common, **kwargs)
    elif type in {"hist", "density"}:
        if y is not None:
            raise ValueError(
                "For a distribution use vis('score', data=df, type='hist' or 'density')."
            )
        numeric(frame[x], x)
        (sns.histplot if type == "hist" else sns.kdeplot)(x=x, **common, **kwargs)
    elif type == "count":
        if y is not None:
            if hue is not None:
                raise ValueError("A two-variable count plot already uses the outcome as hue.")
            common.pop("color", None)
            common.update(hue=y, palette=selected_palette)
        sns.countplot(x=x, **common, **kwargs)
    elif type == "heatmap":
        if y is None or hue is not None:
            raise ValueError("Contingency heatmaps need 'row ~ column' and no hue.")
        counts = pd.crosstab(frame[y], frame[x])
        options = {"annot": True, "fmt": "g", "cmap": "Blues", **kwargs}
        sns.heatmap(counts, ax=ax, **options)
    elif type == "proportion":
        if normalize not in {"row", "column", "total"} or y is None or hue is not None:
            raise ValueError(
                "Proportion plots need 'outcome ~ group', normalize='row'/'column'/'total', and no hue."
            )
        counts = pd.crosstab(
            frame[x],
            frame[y],
            normalize={"row": "index", "column": "columns", "total": "all"}[normalize],
        )
        long = (
            counts.rename_axis(index="_group", columns="_outcome")
            .stack()
            .rename("percent")
            .reset_index()
        )
        long["percent"] *= 100
        sns.barplot(
            data=long,
            x="_group",
            y="percent",
            hue="_outcome",
            palette=selected_palette,
            ax=ax,
            **kwargs,
        )
        ax.set_xlabel(x)
        ax.set_ylabel(f"Percent ({normalize} denominator)")
    elif type == "line":
        if y is None:
            raise ValueError("Line plots need 'y ~ x'.")
        numeric(frame[y], y)
        if estimator not in {"mean", "median", None}:
            raise ValueError("Line estimator must be 'mean', 'median', or None.")
        if estimator is None and units is None:
            raise ValueError("Line plots require estimator='mean'/'median' or units='subject_id'.")
        if estimator is not None and units is not None:
            raise ValueError("Use an aggregate estimator or individual units, not both.")
        options = {"estimator": estimator, **kwargs}
        if units is not None:
            options.update(units=units, errorbar=None)
        sns.lineplot(x=x, y=y, **common, **options)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax
