from typing import Optional, List, Tuple

from matplotlib import pyplot as plt

from clscurves.plotter.plotter import RPFPlotter


class RFPlotter(RPFPlotter):

    def __init__(self, rpf_dict, score_is_probability):
        super().__init__(rpf_dict, score_is_probability)

    def plot_rf(
            self,
            weighted: bool = False,
            scale: float = 1.0,
            title: str = "Recall-Fraction Curve",
            cmap: str = "rainbow",
            color_by: str = "thresh",
            cbar_rng: Optional[List[float]] = None,
            cbar_label: Optional[str] = None,
            dpi: Optional[int] = None,
            bootstrapped: bool = False,
            bootstrap_alpha: float = 0.15,
            bootstrap_color: str = "black",
            op_value: Optional[float] = None,
            return_fig: bool = False) -> Optional[Tuple[plt.figure, plt.axes]]:
        """
        Plot the RF (Recall & Fraction Flagged) curve.
        PARAMETERS
        ----------
          weighted -- specifies whether the weighted or unweighted TPR
            (i.e. recall) should be used. For example, TPR (= tp/pos), if
            unweighted, is the number of positive cases captured above a
            threshold, divided by the total number of positive cases. If
            weighted, it is the sum of weights (or "amounts") associated
            with each positive case captured above a threshold, divided by
            the sum of weights associated with all positive cases. For the
            RF curve, this weighting applies only to the TPR axis (default:
            False).

          scale -- value to scale the x-axis by. By choosing a lower number
            like 0.1, you can observe greater detail in sensitive regions
            of the RF curve (default: 1.0).

          title -- title of plot (default: "RF Curve").
          cmap -- colormap string specification (default: "rainbow").
          color_by -- name of key in rpf_dict that specifies which values
            to use when coloring points along the ROC curve; this should
            be either "frac" for fraction of cases flagged or "thresh" for
            score discrimination threshold (default: "thresh").

          cbar_rng -- [Optional] specify a color bar range of the form
            [min_value, max_value] to override the default range (default:
            None).

          cbar_label -- [Optional] custom label to apply to the color bar. If
            None is supplied, the default ("Threshold Value" or "Fraction
            Flagged", depending on the color_by value) will be used
            (default: None).

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

        # Specify which values to plot in X and Y
        x_key = "frac"
        y_key = "tpr_w" if weighted else "tpr"
        x = self.rpf_dict[x_key]
        y = self.rpf_dict[y_key]

        # Make plot
        if not bootstrapped:
            fig, ax = self._make_plot(x[:, 0], y[:, 0], cmap, dpi, color_by,
                                      cbar_rng, cbar_label)
        else:
            fig, ax = self._make_bootstrap_plot(
                x, y, cmap, dpi, color_by, cbar_rng,
                cbar_label, bootstrap_alpha, bootstrap_color)

        # Plot line of randomness
        ax.plot([0, 1], [0, 1], "k-")

        # Extract AUC
        auc = self.rpf_dict["rf_auc_w" if weighted else "rf_auc"]
        auc = auc[0] if type(
            auc) == np.ndarray and not bootstrapped else np.mean(auc)

        # Add RF AUC to plot
        ax.text(
            x=0.92,
            y=0.1,
            s="%sAURFC = %.3f" % ("Mean " if bootstrapped else "", auc),
            ha="right",
            va="center",
            bbox=dict(facecolor="gray",
                      alpha=0.1,
                      boxstyle="round"))

        # Plot 95% confidence ellipse
        if op_value is not None:
            self._add_op_ellipse(
                op_value=op_value,
                x_key=x_key,
                y_key=y_key,
                ax=ax,
                thresh_key=color_by)

        # Set labels
        weight_string = "Weighted " if weighted else ""
        ax.set_xlabel("Fraction Flagged = (TP + FP)/All")
        ax.set_ylabel("%sRecall = TPR = TP/(TP + FN)" % weight_string)
        ax.set_title(title)

        # Rescale x axis
        if scale != 1.0:
            ax.set_xlim([0, scale])
            ax.set_aspect(scale)
            ax.texts[0].set_visible(False)

        if return_fig:
            return fig, ax

        else:
            # Display and close plot
            display(fig)
            plt.gcf().clear()
            plt.close()
            return