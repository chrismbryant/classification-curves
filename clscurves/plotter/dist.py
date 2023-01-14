from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from clscurves.plotter.plotter import MetricsPlotter


class DistPlotter(MetricsPlotter):
    def __init__(
        self,
        metrics_dict: Dict[str, Any],
        score_is_probability: bool,
        reverse_thresh: bool,
    ) -> None:
        super().__init__(metrics_dict, score_is_probability)
        self.reverse_thresh = reverse_thresh

    def plot_dist(  # noqa: C901
        self,
        weighted: bool = False,
        label: Optional[Union[str, int]] = "all",
        kind: str = "CDF",
        kernel_size: float = 10,
        log_scale: bool = False,
        title: Optional[str] = None,
        cmap: str = "rainbow",
        color_by: str = "tpr",
        cbar_rng: Optional[List[float]] = None,
        cbar_label: Optional[str] = None,
        grid: bool = True,
        x_rng: Optional[List[float]] = None,
        y_rng: Optional[List[float]] = None,
        dpi: Optional[int] = None,
        bootstrapped: bool = False,
        bootstrap_alpha: float = 0.15,
        bootstrap_color: str = "black",
        return_fig: bool = False,
    ) -> Optional[Tuple[plt.figure, plt.axes]]:
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
            Name of key in metrics_dict that specifies which values to use when
            coloring points along the PDF or CDF curve.
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
        return_fig
            If set to True, will return (fig, ax) as a tuple instead of
            plotting the figure.

        Returns
        -------
        Optional[Tuple[plt.figure, plt.axes]]
            The plot's figure and axis object.
        """
        assert label in ["all", 0, 1, None], '`label` must be in ["all", 0, 1, None]'

        kind = kind.lower()
        assert kind in ["cdf", "pdf"], '`kind` must be "cdf" or "pdf"'

        # Specify which values to plot in X and Y
        x = self.metrics_dict["thresh"] * np.ones(
            1 + self.metrics_dict["num_bootstrap_samples"]
        )

        # Compute CDF
        _w = "_w" if weighted else ""
        if label == "all":
            cdf = 1 - self.metrics_dict["frac" + _w]
        elif label == 1:
            denom = self.metrics_dict["pos" + _w]
            cdf = 1 - self.metrics_dict["tp" + _w] / denom
        elif label == 0:
            denom = self.metrics_dict["neg" + _w]
            cdf = 1 - self.metrics_dict["fp" + _w] / denom
        else:
            denom = self.metrics_dict["unk" + _w]
            cdf = 1 - self.metrics_dict["up" + _w] / denom

        # Account for reversed-behavior thresholds
        if self.reverse_thresh:
            cdf = 1 - cdf

        # Compute discrete difference to convert CDF to PDF
        dy = np.diff(cdf, axis=0)
        dx = np.diff(x, axis=0)
        zeros = np.zeros([1, dy.shape[1]])
        pdf = np.nan_to_num(
            np.concatenate([zeros, dy], axis=0) / np.concatenate([zeros, dx], axis=0)
        )

        # Smooth y if it's a PDF
        y = cdf if kind == "cdf" else gaussian_filter1d(pdf, kernel_size, axis=0)

        # Make plot
        if not bootstrapped:
            fig, ax = self._make_plot(
                x[:, 0], y[:, 0], cmap, dpi, color_by, cbar_rng, cbar_label, grid
            )
        else:
            fig, ax = self._make_bootstrap_plot(
                x,
                y,
                cmap,
                dpi,
                color_by,
                cbar_rng,
                cbar_label,
                grid,
                bootstrap_alpha,
                bootstrap_color,
            )

        # Change x-axis range
        if x_rng:
            ax.set_xlim(x_rng)

        # Log scale x-axis
        if log_scale:
            ax.set_xscale("log")
            if self.score_is_probability:
                ax.set_xlim([0, 1] if x_rng else x_rng)

        # Change y-axis range
        if y_rng:
            ax.set_ylim(y_rng)

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

    def plot_pdf(self, **kwargs) -> Optional[Tuple[plt.figure, plt.axes]]:
        return self.plot_dist(kind="pdf", **kwargs)

    def plot_cdf(self, **kwargs) -> Optional[Tuple[plt.figure, plt.axes]]:
        return self.plot_dist(kind="cdf", **kwargs)
