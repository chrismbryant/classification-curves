from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from clscurves.plotter.plotter import MetricsPlotter


class CostPlotter(MetricsPlotter):
    def __init__(
        self,
        metrics_dict: Dict[str, Any],
        score_is_probability: bool,
    ) -> None:
        super().__init__(metrics_dict, score_is_probability)

    def compute_cost(
        self,
        fn_cost_multiplier=1,
        fp_cost_multiplier=1,
        use_weighted_fn=False,
        use_weighted_fp=False,
    ) -> None:

        fn = (
            self.metrics_dict["fn_w"] if use_weighted_fn else self.metrics_dict["fn"]
        )  # noqa
        fp = (
            self.metrics_dict["fp_w"] if use_weighted_fp else self.metrics_dict["fp"]
        )  # noqa

        self.metrics_dict["fn_cost"] = fn_cost_multiplier * fn
        self.metrics_dict["fp_cost"] = fp_cost_multiplier * fp
        self.metrics_dict["cost"] = (
            self.metrics_dict["fn_cost"] + self.metrics_dict["fp_cost"]
        )

    def plot_cost(  # noqa: C901
        self,
        title: str = "Misclassification Cost",
        cmap: str = "rainbow",
        log_scale: bool = False,
        x_axis: str = "thresh",
        x_label: Optional[str] = None,
        x_rng: Optional[List[float]] = None,
        y_label: str = "Cost",
        y_rng: Optional[List[float]] = None,
        color_by: str = "frac",
        cbar_rng: Optional[List[float]] = None,
        cbar_label: Optional[str] = None,
        grid: bool = True,
        dpi: Optional[int] = None,
        bootstrapped: bool = False,
        bootstrap_alpha: float = 0.15,
        bootstrap_color: str = "black",
        return_fig: bool = False,
    ) -> Optional[Tuple[plt.figure, plt.axes]]:
        """Plot the "Misclassification Cost" curve.

        Note: `compute_cost` must be run first to obtain cost values.

        Parameters
        ----------
        title
            Title of plot.
        cmap
            Colormap string specification.
        log_scale
            Boolean to specify whether the x-axis should be scaled by a log10
            transformation.
        x_axis
            Name of key in metrics_dict that specifies which values to use for
            the x coordinates of the cost curve.
        x_label
            Label to apply to x-axis. Defaults for common choices of x-axis
            will be supplied if no x-label override is supplied here.
        x_rng
            Specify an x-axis range of the form [min_value, max_value] to
            override the default range.
        y_label
            Label to apply to y-axis.
        y_rng
            Specify a y-axis range of the form [min_value, max_value] to
            override the default range.
        color_by
            Name of key in metrics_dict that specifies which values to use when
            coloring points along the cost curve.
        cbar_rng
            Specify a color bar range of the form [min_value, max_value] to
            override the default range.
        cbar_label
            Custom label to apply to the color bar. If None is supplied, the
            default ("Fraction Flagged") will be used.
        grid
            Whether to plot grid lines.
        dpi
            Resolution in "dots per inch" of resulting figure. If not
            specified, the Matplotlib default will be used. A good rule of
            thumb is 150 for good quality at normal screen resolutions and 300
            for high quality that maintains sharp features after zooming in.
        bootstrapped
            Specifies whether bootstrapped curves should be plotted behind the
            main colored performance scatter plot.
        bootstrap_alpha
            Opacity of bootstrap curves.
        bootstrap_color
            Color of bootstrap curves.
        return_fig
            If set to True, will return (fig, ax) as a tuple instead of
            plotting the figure.

        Returns
        -------
        Optional[Tuple[plt.figure, plt.axes]]
            The plot's figure and axis object.
        """
        assert "cost" in self.metrics_dict, "Run `compute_cost` first."

        # Create figure
        fig = plt.figure(figsize=(10, 6), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(grid)

        # Make Color Bar
        if cbar_rng is not None:
            [vmin, vmax] = cbar_rng
        else:
            vmin = 0.0 if color_by == "frac" else np.min(self.metrics_dict[color_by])
            vmax = 1.0 if color_by == "frac" else np.max(self.metrics_dict[color_by])
        norm = matplotlib.colors.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, ticks=np.linspace(vmin, vmax, 11))
        label = "Threshold Value" if cbar_label is None else cbar_label
        cbar.set_label("Fraction Flagged" if color_by == "frac" else label)

        # Make scatter plot
        cost = self.metrics_dict["cost"]
        x = self.metrics_dict[x_axis]
        colors = self.metrics_dict[color_by]

        # Make main colored scatter plot
        ax.scatter(
            np.log10(x[:, 0]) if log_scale else x[:, 0],
            cost[:, 0],
            s=100,
            c=colors[:, 0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=".",
            edgecolors="none",
            zorder=int(1e4),
        )

        # Plot faint bootstrapped curves
        if bootstrapped:
            x_boot = x[:, 1:] if x.shape[1] > 1 else x
            cost_boot = cost[:, 1:]
            for i in range(self.metrics_dict["num_bootstrap_samples"]):
                x_vals = x_boot if x_boot.shape[1] == 1 else x_boot[:, i]
                ax.plot(
                    np.log10(x_vals) if log_scale else x_vals,
                    cost_boot[:, i],
                    alpha=bootstrap_alpha,
                    color=bootstrap_color,
                    linewidth=1,
                )

        # Set x limits
        if not log_scale and x_axis in [
            "tpr",
            "fpr",
            "tpr_w",
            "fpr_w",
            "frac",
            "precision",
        ]:
            ax.set_xlim(0, 1)
        if x_rng:
            ax.set_xlim(*x_rng)
        if y_rng:
            ax.set_ylim(*y_rng)

        # Create label for x-axis
        if x_label:
            x_label = x_label
        elif x_axis in self.cbar_dict:
            x_label = self.cbar_dict[x_axis]
        else:
            x_label = x_axis
        if log_scale:
            x_label = "log$_{10}$(%s)" % x_label

        # Set labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if return_fig:
            return fig, ax

        else:
            # Display and close plot
            plt.show()
            plt.close()

        return None
