from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from clscurves.plotter.plotter import RPFPlotter


class DistPlotter(RPFPlotter):

    def __init__(self, rpf_dict: Dict[str, Any], score_is_probability: bool,
                 reverse_thresh: bool):
        super().__init__(rpf_dict, score_is_probability)
        self.reverse_thresh = reverse_thresh

    def plot_dist(
            self,
            weighted: bool = False,
            label: str = "all",
            kind: str = "CDF",
            kernel_size: float = 10,
            log_scale: bool = False,
            title: Optional[str] = None,
            cmap: str = "rainbow",
            color_by: str = "tpr",
            cbar_rng: Optional[List[float]] = None,
            cbar_label: Optional[str] = None,
            x_rng: Optional[List[float]] = None,
            y_rng: Optional[List[float]] = None,
            dpi: Optional[int] = None,
            bootstrapped: bool = False,
            bootstrap_alpha: float = 0.15,
            bootstrap_color: str = "black",
            op_value: Optional[float] = None,
            return_fig: bool = False) -> Optional[Tuple[plt.figure, plt.axes]]:
        """
        Plot the CDF (Cumulative Distribution Function) or PDF (Probability
        Density Function) curve.
        PARAMETERS
        ----------
          weighted -- specifies whether the weighted or unweighted fraction
            flagged should be used when computing the CDF or PDF. If unweighted,
            the fraction flagged is the number of cases flagged divided by
            the number of cases total. If weighted, it is the sum of the
            weights of all the cases flagged, divided by the sum of the
            weights of all the cases (default: False).

          label -- class label to plot the CDF for; one of "all", `1`, `0`,
            or `None` (default: `None`).

          kind -- either "cdf" or "pdf" (default: "cdf").

          kernel_size -- used for PDF only: standard deviation of the Gaussian of
            kernel to use when smoothing the PDF curve (default: 10).

          log_scale -- boolean to specify whether the x-axis should be log-scaled
            (default: False).

          title -- title of plot (default: None).
          cmap -- colormap string specification (default: "rainbow").
          color_by -- name of key in rpf_dict that specifies which values
            to use when coloring points along the PDF or CDF curve (default:
            "tpr").

          cbar_rng -- [Optional] specify a color bar range of the form
            [min_value, max_value] to override the default range (default:
            None).

          cbar_label -- [Optional] custom label to apply to the color bar. If
            None is supplied, a default will be selected from the cbar_dict
            (default: None).

          y_rng - [Optional] range of the vertical axis (default: None).

          dpi -- [Optional] resolution in "dots per inch" of resulting figure.
            If not specified, the Matplotlib default will be used. A good rule
            of thumb is 150 for good quality at normal screen resolutions
            and 300 for high quality that maintains sharp features
            after zooming in (default: None).

          bootstrapped -- [Optional] specifies whether bootstrapped curves
            should be plotted behind the main colored performance scatter plot
            (default: False).

          bootstrap_alpha -- [Optional] opacity of bootstrap curves (default:
            0.15).

          bootstrap_color -- [Optional] color of bootstrap curves (default: "black")

          op_value -- [Optional] threshold value to plot a confidence ellipse
            for when the plot is bootstrapped (default: None).

          return_fig -- [Optional] if set to True, will return (fig, ax) as
            a tuple instead of plotting the figure.
        """

        assert label in ["all", 0, 1, None], \
            "`label` must be in [\"all\", 0, 1, None]"

        kind = kind.lower()
        assert kind in ["cdf", "pdf"], \
            "`kind` must be \"cdf\" or \"pdf\""

        # Specify which values to plot in X and Y
        x = self.rpf_dict["thresh"] * np.ones(
            1 + self.rpf_dict["num_bootstrap_samples"])

        # Compute CDF
        _w = "_w" if weighted else ""
        if label == "all":
            cdf = 1 - self.rpf_dict["frac" + _w]
        elif label == 1:
            denom = self.rpf_dict["pos" + _w]
            cdf = 1 - self.rpf_dict["tp" + _w] / denom
        elif label == 0:
            denom = self.rpf_dict["neg" + _w]
            cdf = 1 - self.rpf_dict["fp" + _w] / denom
        elif label == None:
            denom = self.rpf_dict["unk" + _w]
            cdf = 1 - self.rpf_dict["up" + _w] / denom

        # Account for reversed-behavior thresholds
        if self.reverse_thresh:
            cdf = 1 - cdf

        # Compute discrete difference to convert CDF to PDF
        dy = np.diff(cdf, axis=0)
        dx = np.diff(x, axis=0)
        zeros = np.zeros([1, dy.shape[1]])
        pdf = np.nan_to_num(
            np.concatenate([zeros, dy], axis=0) / \
            np.concatenate([zeros, dx], axis=0)
        )

        # Smooth y if it's a PDF
        y = cdf if kind == "cdf" else gaussian_filter1d(pdf, kernel_size,
                                                        axis=0)

        # Make plot
        if not bootstrapped:
            fig, ax = self._make_plot(x[:, 0], y[:, 0], cmap, dpi, color_by,
                                      cbar_rng, cbar_label)
        else:
            fig, ax = self._make_bootstrap_plot(
                x, y, cmap, dpi, color_by, cbar_rng,
                cbar_label, bootstrap_alpha, bootstrap_color)

        # Plot 95% confidence ellipse
        if op_value is not None:
            self._add_op_ellipse(
                op_value=op_value,
                x_key=x_key,
                y_key=y_key,
                ax=ax,
                thresh_key=color_by)

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
        title = f"CDF{label_string}" if kind == "cdf" else f"PDF{label_string}"
        ax.set_xlabel("Score")
        ax.set_ylabel(
            "Cumulative Distribution" if kind == "cdf" else "Density")
        ax.set_title(title)

        if return_fig:
            return fig, ax

        else:
            # Display and close plot
            display(fig)
            plt.gcf().clear()
            plt.close()
            return

    def plot_pdf(self, **kwargs) -> Optional[Tuple[plt.figure, plt.axes]]:
        return self.plot_dist(kind="pdf", **kwargs)

    def plot_cdf(self, **kwargs) -> Optional[Tuple[plt.figure, plt.axes]]:
        return self.plot_dist(kind="cdf", **kwargs)