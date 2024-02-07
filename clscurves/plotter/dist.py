from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
from typing_extensions import Literal

from clscurves.plotter.plotter import MetricsPlotter
from clscurves.utils import MetricsResult

Label = Literal["all", 0, 1, None]


class DistPlotter(MetricsPlotter):
    def __init__(
        self,
        metrics: MetricsResult,
        score_is_probability: bool,
        reverse_thresh: bool,
    ) -> None:
        super().__init__(metrics, score_is_probability)
        self.reverse_thresh = reverse_thresh

    def plot_dist(  # noqa: C901
        self,
        weighted: bool = False,
        label: Label = "all",  # noqa
        kind: str = "CDF",
        kernel_size: float = 10,
        log_scale: bool = False,
        title: Optional[str] = None,
        cmap: str = "rainbow",
        color_by: str = "recall",
        cbar_rng: Optional[List[float]] = None,
        cbar_label: Optional[str] = None,
        grid: bool = True,
        x_rng: Optional[List[float]] = None,
        y_rng: Optional[List[float]] = None,
        dpi: Optional[int] = None,
        bootstrapped: bool = False,
        bootstrap_alpha: float = 0.15,
        bootstrap_color: str = "black",
        imputed: bool = False,
        return_fig: bool = False,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot the data distribution.

        This plots either the CDF (Cumulative Distribution Function) or PDF
        (Probability Density Function) curve.

        Parameters
        ----------
        weighted
            Specifies whether the weighted or unweighted fraction flagged
            should be used when computing the CDF or PDF. If unweighted, the
            fraction flagged is the number of cases flagged divided by the
            number of cases total. If weighted, it is the sum of the weights of
            all the cases flagged, divided by the sum of the weights of all
            the cases.
        label
            Class label to plot the CDF for; one of "all", 1, 0, or `None`.
        kind
            Either "cdf" or "pdf".
        kernel_size
            Used for PDF only: standard deviation of the Gaussian of kernel to
            use when smoothing the PDF curve.
        log_scale
            Boolean to specify whether the x-axis should be log-scaled.
        title
            Title of plot.
        cmap
            Colormap string specification.
        color_by
            Name of key in metrics.curves that specifies which values to use
            when coloring points along the PDF or CDF curve.
        cbar_rng
            Specify a color bar range of the form [min_value, max_value] to
            override the default range.
        cbar_label
            Custom label to apply to the color bar. If `None` is supplied, a
            default will be selected from the ``cbar_dict``.
        grid
            Whether to plot grid lines.
        x_rng
            Range of the horizontal axis.
        y_rng
            Range of the vertical axis.
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
        imputed
            Whether to plot imputed curves.
        return_fig
            If set to True, will return (fig, ax) as a tuple instead of
            plotting the figure.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            The plot's figure and axis object.
        """
        assert label in ["all", 0, 1, None], '`label` must be in ["all", 0, 1, None]'

        kind = kind.lower()
        assert kind in ["cdf", "pdf"], '`kind` must be "cdf" or "pdf"'

        # Get metrics
        curves, _ = self._get_metrics(imputed=imputed)

        # Compute CDF
        _w = "_w" if weighted else ""
        if label == "all":
            cdf = 1 - curves["frac" + _w]
        else:
            raise NotImplementedError("TODO: Implement label=None case.")

        # Account for reversed-behavior thresholds
        if self.reverse_thresh:
            cdf = 1 - cdf

        curves["_cdf"] = cdf
        x_col = "thresh"
        y_col = "_cdf"

        # TODO: Support PDF via KDE.

        # Make plot
        if not bootstrapped:
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
            )
        else:
            fig, ax = self._make_bootstrap_plot(
                curves=curves,
                x_col=x_col,
                y_col=y_col,
                cmap=cmap,
                dpi=dpi,
                color_by=color_by,
                cbar_rng=cbar_rng,
                cbar_label=cbar_label,
                grid=grid,
                alpha=bootstrap_alpha,
                bootstrap_color=bootstrap_color,
            )

        # Change x-axis range
        if x_rng:
            ax.set_xlim(*x_rng)

        # Log scale x-axis
        if log_scale:
            ax.set_xscale("log")
            if self.score_is_probability:
                x_rng = [0, 1] if x_rng is None else x_rng
                ax.set_xlim(*x_rng)

        # Change y-axis range
        if y_rng:
            ax.set_ylim(*y_rng)

        # Set aspect ratio
        x_size = x_rng[1] - x_rng[0] if x_rng else 1
        y_size = y_rng[1] - y_rng[0] if y_rng else 1
        ax.set_aspect(x_size / y_size)

        # Set labels
        weight_string = "Weighted " if weighted else ""
        label_string = f": Label = {label}" if label in [0, 1, None] else ""
        default_title = (
            f"{weight_string}CDF{label_string}"
            if kind == "cdf"
            else f"{weight_string}PDF{label_string}"
        )
        title = default_title if title is None else title
        ax.set_xlabel("Score")
        ax.set_ylabel("Cumulative Distribution" if kind == "cdf" else "Density")
        ax.set_title(title)

        if return_fig:
            return fig, ax

        else:
            # Display and close plot
            plt.show()
            plt.close()

        return None

    def plot_pdf(self, **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        return self.plot_dist(kind="pdf", **kwargs)

    def plot_cdf(self, **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        return self.plot_dist(kind="cdf", **kwargs)
