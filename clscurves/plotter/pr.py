from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from clscurves.plotter.plotter import MetricsPlotter
from clscurves.utils import MetricsResult


class PRPlotter(MetricsPlotter):
    def __init__(
        self,
        metrics: MetricsResult,
        score_is_probability: bool,
    ) -> None:
        super().__init__(metrics, score_is_probability)

    def plot_pr(
        self,
        weighted: bool = False,
        title: str = "Precision-Recall Curve",
        cmap: str = "rainbow",
        color_by: str = "thresh",
        cbar_rng: Optional[List[float]] = None,
        cbar_label: Optional[str] = None,
        dpi: Optional[int] = None,
        bootstrapped: bool = False,
        bootstrap_alpha: float = 0.15,
        bootstrap_color: str = "black",
        imputed: bool = False,
        f1_contour: bool = False,
        grid: bool = True,
        op_value: Optional[float] = None,
        return_fig: bool = False,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot the PR (Precision & Recall) curve.

        Parameters
        ----------
        weighted
            Specifies whether the weighted or unweighted recall (i.e. TPR)
            should be used. For example, recall (= tp/pos), if unweighted, is the
            number of positive cases captured above a threshold, divided by the
            total number of positive cases. If weighted, it is the sum of
            weights (or "amounts") associated with each positive case captured
            above a threshold, divided by the sum of weights associated with
            all positive cases. For the PR curve, this weighting applies only
            to the recall axis.
        title
            Title of plot.
        cmap
            Colormap string specification.
        color_by
            Name of key in metrics.curves that specifies which values to use when
            coloring points along the PR curve; this should be either "frac"
            for fraction of cases flagged or "thresh" for score discrimination
            threshold.
        cbar_rng
            Specify a color bar range of the form [min_value, max_value] to
            override the default range.
        cbar_label
            Custom label to apply to the color bar. If None is supplied, the
            default ("Threshold Value" or "Fraction Flagged", depending on the
            color_by value) will be used.
        dpi
            Resolution in "dots per inch" of resulting figure. If not
            specified, the Matplotlib default will be used. A good rule of
            thumb is 150 for good quality at normal screen resolutions and 300
            for high quality that maintains sharp features after zooming in.
        bootstrapped
            Sspecifies whether bootstrapped curves should be plotted behind the
            main colored performance scatter plot.
        bootstrap_alpha
            Opacity of bootstrap curves.
        bootstrap_color
            Color of bootstrap curves.
        imputed
            Whether to plot imputed curves.
        f1_contour
            Whether to include reference contours for curves of constant F1.
        grid
            Whether to plot grid lines.
        op_value
            Threshold value to plot a confidence ellipse for when the plot is
            bootstrapped.
        return_fig
            If set to True, will return (fig, ax) as a tuple instead of
            plotting the figure.
        """

        # Get metrics
        curves, scalars = self._get_metrics(imputed=imputed)

        # Specify which values to plot in X and Y
        x_col = "recall_w" if weighted else "recall"
        y_col = "precision"

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

        # Plot F1 contour curves
        if f1_contour:

            def _f1_contour(f: float):
                r = np.linspace(f / (2 - f), 1, 200)
                p = f * r / (2 * r - f)
                return r, p

            f1_values = np.arange(0.1, 1, 0.1)
            for f1 in f1_values:
                rec, pr = _f1_contour(f1)
                ax.plot(
                    rec,
                    pr,
                    zorder=-1,
                    c="black",
                    ls="--",
                    alpha=0.3,
                )
                ax.text(
                    x=rec[0] + 0.005 + 0.02 * f1**2,
                    y=0.98,
                    s=np.round(f1, 1),
                    size=8,
                    alpha=0.6,
                )

        # Extract class imbalance
        imb = scalars.loc[lambda x: x["_bootstrap_sample"].isnull()]["imbalance"].iloc[
            0
        ]

        # Plot line of randomness
        ax.plot([0, 1], [imb, imb], "k-")

        # Compute the ratio of class sizes
        class_ratio = 1 / (imb + 1e-9) - 1

        # Add class imbalance to plot
        ax.text(
            x=0.92,
            y=0.9,
            s="%sClass Imb. = %.1f : 1"
            % ("Mean " if bootstrapped else "", class_ratio),
            ha="right",
            va="center",
            bbox=dict(facecolor="gray", alpha=0.1, boxstyle="round"),
        )

        # Plot 95% confidence ellipse
        if op_value is not None:
            self._add_op_ellipse(
                curves=curves,
                op_value=op_value,
                x_col=x_col,
                y_col=y_col,
                ax=ax,
                thresh_key=color_by,
            )

        # Set labels
        weight_string = "Weighted " if weighted else ""
        ax.set_xlabel("%sRecall = recall = TP/(TP + FN)" % weight_string)
        ax.set_ylabel("Precision = TP/(TP + FP)")
        ax.set_title(title)

        if return_fig:
            return fig, ax

        else:
            # Display and close plot
            plt.show()
            plt.close()

        return None
