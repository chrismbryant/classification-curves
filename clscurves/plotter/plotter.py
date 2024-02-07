from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from clscurves.config import MetricsAliases
from clscurves.covariance import CovarianceEllipseGenerator
from clscurves.utils import MetricsResult


class MetricsPlotter(MetricsAliases):
    """A helper class to provide methods shared by each metrics plotter.

    These methods streamline the process of making a single classification
    curve metrics plot, making a bootstrapped plot, and adding a confidence
    ellipse to a specified operating point.
    """

    def __init__(
        self,
        metrics: MetricsResult,
        score_is_probability: bool,
    ) -> None:
        self.metrics = metrics
        self.score_is_probability = score_is_probability

    def _add_op_ellipse(
        self,
        curves: pd.DataFrame,
        op_value: float,
        x_col: str,
        y_col: str,
        ax: plt.Axes,
        thresh_key: str = "thresh",
    ) -> None:
        """A helper function to add a confidence ellipse to an metrics plot
        given a threshold operating value.

        Parameters
        ----------
        curves : pd.DataFrame
            metrics.curves DataFrame.
        op_value : float
            Threshold operating value.
        x_col : str
            metrics.curves key used in plot x axis.
        y_col : str
            metrics.curves key used in plot y axis.
        ax : plt.Axes
            Matplotlib axis object.
        thresh_key : str
            metrics.curves key used for coloring (default: "thresh").
        """

        def find_op_point(df: pd.DataFrame) -> pd.Series:
            """Find all entries at or above the operating point threshold."""
            match = df[df[thresh_key] >= op_value]
            if not match.empty:
                return match.iloc[0]
            return df.iloc[0]

        # Get operating point coordinates for each bootstrapped sample
        op_points = curves.groupby("_bootstrap_sample", dropna=False).apply(
            find_op_point
        )
        op_data = op_points[[x_col, y_col]].values.T

        # Compute covariance ellipse and add to ax
        ceg = CovarianceEllipseGenerator(op_data)
        ceg.create_ellipse_patch(ax=ax, color="black")
        ceg.add_ellipse_center(ax=ax)

        # Add individual operating points
        ax.scatter(x=op_data[0], y=op_data[1], s=2, c="black", alpha=0.7, marker=".")

    def _make_plot(
        self,
        curves: pd.DataFrame,
        x_col: str,
        y_col: str,
        cmap: str,
        dpi: Optional[int],
        color_by: str,
        cbar_rng: Optional[List[float]],
        cbar_label: Optional[str],
        grid: bool,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """A helper function to create a base Matplotlib scatter plot figure
        for metrics-related plotting.
        """

        # Get non-bootstrapped data
        curves = curves.loc[lambda x: x["_bootstrap_sample"].isnull()]

        # Create figure
        if not ax:
            fig = plt.figure(figsize=(10, 7), dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, aspect="equal")
            ax.grid(grid)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Make Color Bar
        color = curves[color_by]
        if cbar_rng is not None:
            [vmin, vmax] = cbar_rng
        else:
            sip = self.score_is_probability or color_by == "frac"
            vmin = 0.0 if sip else color.min()
            vmax = 1.0 if sip else color.max()
        norm = matplotlib.colors.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(vmin, vmax, 11))  # type: ignore
        default_cbar_label = (
            self.cbar_dict[color_by] if color_by in self.cbar_dict else "Value"
        )
        cbar_label = default_cbar_label if cbar_label is None else cbar_label
        cbar.set_label(cbar_label)

        # Make scatter plot
        print("Making scatter plot...")
        ax.scatter(
            curves[x_col],
            curves[y_col],
            s=100,
            c=color,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=".",
            edgecolors="none",
            zorder=int(1e4),
        )

        return fig, ax  # type: ignore

    def _make_bootstrap_plot(
        self,
        curves: pd.DataFrame,
        x_col: str,
        y_col: str,
        cmap: str,
        dpi: Optional[int],
        color_by: str,
        cbar_rng: Optional[List[float]],
        cbar_label: Optional[str],
        grid: bool,
        alpha: float,
        bootstrap_color: str,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """A helper function to add faint bootstrapped reference curves to an
        metrics plot to visualize the confidence we have in the main metrics
        curve.
        """

        # Create figure
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        ax.grid(grid)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot faint bootstrapped curves
        num_bootstrap_samples = curves["_bootstrap_sample"].nunique() - 1
        for i in range(num_bootstrap_samples):
            ax.plot(
                curves.loc[lambda x: x["_bootstrap_sample"] == i, x_col],
                curves.loc[lambda x: x["_bootstrap_sample"] == i, y_col],
                alpha=alpha,
                color=bootstrap_color,
                linewidth=1,
            )

        # Plot main colored curve (scatter plot) with color bar
        fig, ax = self._make_plot(
            curves=curves,
            x_col=x_col,
            y_col=y_col,
            cmap=cmap,
            dpi=dpi,
            color_by=color_by,
            cbar_rng=cbar_rng,
            cbar_label=cbar_label,
            grid=grid,
            fig=fig,
            ax=ax,
        )

        return fig, ax

    def _get_metrics(
        self,
        imputed: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """A helper function to get the metrics curves and scalars DataFrames
        for plotting.
        """
        curves = self.metrics.curves_imputed if imputed else self.metrics.curves
        scalars = self.metrics.scalars_imputed if imputed else self.metrics.scalars
        return curves, scalars
