from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

from clscurves.binomial_ci import BinomialCI


def assess_prob_calibration(
    predictions_df: pd.DataFrame,
    label_column: str = "label",
    score_column: str = "prob",
) -> pd.DataFrame:
    """Compute probability calibration metrics.

    This is used to determine how well the predicted probabilities line up with
    the true "proportion positive" rate in the ground truth data.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions and labels.
    label_column : str, optional
        Column name for the true labels, by default "label".
    score_column : str, optional
        Column name for the predicted probabilities, by default "prob".
    """

    predictions_df["quantile"] = predictions_df[score_column].rank() / len(
        predictions_df
    )
    predictions_df["quantile_bucket"] = predictions_df["quantile"].round(2)
    predictions_df["score_bucket"] = predictions_df[score_column].round(2)

    score_prob_calibration = (
        predictions_df.groupby("score_bucket")
        .agg(
            avg_prediction=(score_column, "mean"),
            num_actual_pos=(label_column, "sum"),
            num_examples=(label_column, "count"),
        )
        .reset_index()
    )

    quantile_prob_calibration = (
        predictions_df.groupby("quantile_bucket")
        .agg(
            avg_prediction=(score_column, "mean"),
            num_actual_pos=(label_column, "sum"),
            num_examples=(label_column, "count"),
        )
        .reset_index()
    )

    prob_calibration = pd.concat([score_prob_calibration, quantile_prob_calibration])

    (
        prob_calibration["proportion"],
        prob_calibration["lower"],
        prob_calibration["upper"],
    ) = BinomialCI().get_ci(
        prob_calibration["num_actual_pos"],
        prob_calibration["num_examples"],
    )

    return prob_calibration.sort_values("avg_prediction")


def plot_probability_calibration(
    prob_calibration: pd.DataFrame,
    plot_confidence_band: bool = False,
    dpi: Optional[int] = None,
    color: str = "tab:blue",
    title: str = "Probability Calibration Curve",
    x_label: str = "Mean Predicted Value",
    y_label: str = "Proportion Labeled Positive",
    return_fig: bool = False,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Plot probability calibration curve.

    Parameters
    ----------
    prob_calibration : pd.DataFrame
        DataFrame containing probability calibration metrics (this is the
        output of `assess_prob_calibration`).
    plot_confidence_band : bool, optional
        Whether to plot 95% confidence band around the calibration curve, by
        default False. If False, error bars will be plotted on each point. If
        True, a shaded confidence band will be plotted instead.
    dpi : int, optional
        Dots per inch for the plot, by default None.
    color : str, optional
        Color for the plot, by default "tab:blue".
    title : str, optional
        Title for the plot, by default "Probability Calibration Curve".
    x_label : str, optional
        X-axis label for the plot, by default "Mean Predicted Value".
    y_label : str, optional
        Y-axis label for the plot, by default "Proportion Labeled Positive".

    Returns
    -------
    Optional[tuple[plt.Figure, plt.Axes]]
        If `return_fig` is True, returns a tuple containing the figure and axes
        objects.
    """

    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add line of perfect calibration
    ax.plot([0, 1], [0, 1], "k-")

    if plot_confidence_band:

        # Shade confidence band
        ax.fill_between(
            x=prob_calibration["avg_prediction"],
            y1=prob_calibration["lower"],
            y2=prob_calibration["upper"],
            alpha=0.3,
            color=color,
        )
        ax.plot(
            prob_calibration["avg_prediction"],
            prob_calibration["proportion"],
            color=color,
        )

    else:

        # Plot points
        ax.scatter(
            x=prob_calibration["avg_prediction"],
            y=prob_calibration["proportion"],
            s=30,
            color=color,
            marker=".",
            zorder=100,
        )

        # Plot error bars on points
        ax.errorbar(
            x=prob_calibration["avg_prediction"],
            y=prob_calibration["proportion"],
            yerr=[
                prob_calibration["proportion"] - prob_calibration["lower"],
                prob_calibration["upper"] - prob_calibration["proportion"],
            ],
            alpha=0.6,
            color=color,
            fmt=".",
            markersize=0,
            zorder=100,
        )

        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    if return_fig:
        return fig, ax
    return None


def plot_predictions(
    predictions_df: pd.DataFrame,
    label_column: str = "label",
    score_column: str = "pred",
    dpi: Optional[int] = None,
    color: str = "tab:blue",
    title: str = "Quantity Predictions",
    x_label: str = "Predicted Quantity",
    y_label: str = "Actual Quantity",
    x_rng: Optional[List[int]] = None,
    x_jitter: float = 0,
    y_jitter: float = 0,
    log_scale: bool = False,
    alpha: float = 0.5,
    size: float = 20,
    vmax: float = 10,
    scatter_density: bool = False,
    cmap: str = "viridis",
    return_fig: bool = False,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    """Plot predicted vs. actual values as a scatter plot.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions and labels.
    score_column : str, optional
        Column name for the predicted values, by default "pred".
    label_column : str, optional
        Column name for the true labels, by default "label".
    dpi : int, optional
        Dots per inch for the plot, by default None.
    color : str, optional
        Color for the plot, by default "tab:blue".
    title : str, optional
        Title for the plot, by default "Quantity Predictions".
    x_label : str, optional
        X-axis label for the plot, by default "Predicted Quantity".
    y_label : str, optional
        Y-axis label for the plot, by default "Actual Quantity".
    x_rng : Optional[List[int]], optional
        Range for x and y axes, by default None.
    x_jitter : float, optional
        Amount of Gaussian jitter to add to x values, by default 0.
    y_jitter : float, optional
        Amount of Gaussian jitter to add to y values, by default 0.
    log_scale : bool, optional
        Whether to use log scale for x and y axes, by default False.
    alpha : float, optional
        Transparency of points, by default 0.5.
    size : float, optional
        Size of points, by default 20.
    vmax : float, optional
        Maximum value for color map, by default 10.
    scatter_density : bool, optional
        Whether to use scatter density plot, by default False.
    cmap : str, optional
        Color map for scatter density plot, by default "viridis".
    return_fig : bool, optional
        Whether to return the figure and axes objects, by default False.

    Returns
    -------
    Optional[tuple[plt.Figure, plt.Axes]]
        If `return_fig` is True, returns a tuple containing the figure and axes
        objects.
    """

    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.grid(True, alpha=0.5)

    # Gaussian jitter
    x_jitter = np.random.normal(0, x_jitter, len(predictions_df))
    y_jitter = np.random.normal(0, y_jitter, len(predictions_df))

    if log_scale:
        x = predictions_df[score_column] * (1 + x_jitter)
        y = predictions_df[label_column] * (1 + y_jitter)
    else:
        x = predictions_df[score_column] + x_jitter
        y = predictions_df[label_column] + y_jitter

    if scatter_density:
        norm = mpl.colors.Normalize(0, vmax)
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        ax.scatter_density(x, y, cmap=cmap, norm=norm, dpi=dpi)  # type: ignore
    else:
        ax.scatter(x=x, y=y, s=size, color=color, marker=".", ec="none", alpha=alpha)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_aspect("equal")

    if x_rng is not None:
        ax.set_xlim(x_rng[0], x_rng[1])
        ax.set_ylim(x_rng[0], x_rng[1])

        # Add line of perfect calibration
        ax.plot([x_rng[0], x_rng[1]], [x_rng[0], x_rng[1]], "k-")

    # Include R2, RMSE, and QI as text in plot
    predictions_df["lit_0"] = 0
    r2 = sklearn.metrics.r2_score(
        predictions_df[label_column], predictions_df[score_column]
    )
    rmse = np.sqrt(
        sklearn.metrics.mean_squared_error(
            predictions_df[label_column], predictions_df[score_column]
        )
    )
    rmsle = np.sqrt(
        sklearn.metrics.mean_squared_log_error(
            predictions_df[[label_column, "lit_0"]].max(axis=1),
            predictions_df[[score_column, "lit_0"]].max(axis=1),
        )
    )
    qi = predictions_df[score_column].sum() / predictions_df[label_column].sum()

    ax.text(
        x=0.05,
        y=0.95,
        s=f"      $R^2$: {r2:.3f}\n RMSE: {rmse:.3f}\nRMSLE: {rmsle:.3f}\n      QI: {qi:.3f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=12,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")

    if return_fig:
        return fig, ax
    return None
