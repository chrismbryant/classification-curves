from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from clscurves.config import MetricsAliases
from clscurves.covariance import CovarianceEllipseGenerator


class MetricsPlotter(MetricsAliases):
    """A helper class to provide methods shared by each metrics plotter.

    These methods streamline the process of making a single classification
    curve metrics plot, making a bootstrapped plot, and adding a confidence
    ellipse to a specified operating point.
    """

    def __init__(
        self,
        metrics_dict: Dict[str, Any],
        score_is_probability: bool,
    ) -> None:
        self.metrics_dict = metrics_dict
        self.score_is_probability = score_is_probability

    def _add_op_ellipse(
        self,
        op_value: float,
        x_key: str,
        y_key: str,
        ax: plt.axes,
        thresh_key: str = "thresh",
    ) -> None:
        """A helper function to add a confidence ellipse to an metrics plot
        given a threshold operating value.

        Parameters
        ----------
        op_value
            Threshold operating value.
        x_key
            metrics_dict key used in plot x axis.
        y_key
            metrics_dict key used in plot y axis.
        ax
            Matplotlib axis object.
        thresh_key
            metrics_dict key used for coloring (default: "thresh").
        """

        # Find all entries at or above the operating point threshold
        data = self.metrics_dict[thresh_key] >= op_value
        size = data.shape[0]

        # Find the index of the first entry at or above the operating threshold
        op_index = size - np.sum(np.cumsum(np.diff(data, axis=0), axis=0), axis=0)

        # Find number of points to plot
        num_points = self.metrics_dict[y_key].shape[1]

        # Get x-y coordinates for each bootstrapped operating point
        op_data = np.array(
            [
                self.metrics_dict[x_key][op_index, np.arange(num_points)],
                self.metrics_dict[y_key][op_index, np.arange(num_points)],
            ]
        )

        # Compute covariance ellipse and add to ax
        ceg = CovarianceEllipseGenerator(op_data)
        ceg.create_ellipse_patch(ax=ax, color="black")
        ceg.add_ellipse_center(ax=ax)

        # Add individual operating points
        ax.scatter(x=op_data[0], y=op_data[1], s=2, c="black", alpha=0.7, marker=".")

    def _make_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cmap: str,
        dpi: Optional[int],
        color_by: str,
        cbar_rng: Optional[List[float]],
        cbar_label: Optional[str],
        grid: bool,
        fig: Optional[plt.figure] = None,
        ax: Optional[plt.axes] = None,
    ) -> Tuple[plt.figure, plt.axes]:
        """A helper function to create a base Matplotlib scatter plot figure
        for metrics-related plotting.
        """

        # Create figure
        if not ax:
            fig = plt.figure(figsize=(10, 7), dpi=dpi)
            ax = fig.add_subplot(1, 1, 1, aspect="equal")
            ax.grid(grid)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Make Color Bar
        if cbar_rng is not None:
            [vmin, vmax] = cbar_rng
        else:
            sip = self.score_is_probability or color_by == "frac"
            vmin = 0.0 if sip else np.min(self.metrics_dict[color_by])
            vmax = 1.0 if sip else np.max(self.metrics_dict[color_by])
        norm = matplotlib.colors.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, ticks=np.linspace(vmin, vmax, 11))  # type: ignore
        default_cbar_label = (
            self.cbar_dict[color_by] if color_by in self.cbar_dict else "Value"
        )
        cbar_label = default_cbar_label if cbar_label is None else cbar_label
        cbar.set_label(cbar_label)

        # Make scatter plot
        print("Making scatter plot...")
        ax.scatter(
            x,
            y,
            s=100,
            c=self.metrics_dict[color_by][:, 0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=".",
            edgecolors="none",
            zorder=int(1e4),
        )

        return fig, ax

    def _make_bootstrap_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cmap: str,
        dpi: Optional[int],
        color_by: str,
        cbar_rng: Optional[List[float]],
        cbar_label: Optional[str],
        grid: bool,
        alpha: float,
        bootstrap_color: str,
    ) -> Tuple[plt.figure, plt.axes]:
        """A helper function to add faint bootstrapped reference curves to an
        metrics plot to visualize the confidence we have in the main metrics
        curve.
        """
        x_main = x[:, 0]
        y_main = y[:, 0]
        x_boot = x[:, 1:]
        y_boot = y[:, 1:]

        # Create figure
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        ax.grid(grid)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot faint bootstrapped curves
        for i in range(self.metrics_dict["num_bootstrap_samples"]):
            ax.plot(
                x_boot[:, i],
                y_boot[:, i],
                alpha=alpha,
                color=bootstrap_color,
                linewidth=1,
            )

        # Plot main colored curve (scatter plot) with color bar
        fig, ax = self._make_plot(
            x_main, y_main, cmap, dpi, color_by, cbar_rng, cbar_label, grid, fig, ax
        )

        return fig, ax
