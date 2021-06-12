class CostPlotter(RPFPlotter):

    def __init__(self, rpf_dict: Dict[str, Any], score_is_probability: bool):
        super().__init__(rpf_dict, score_is_probability)

    def compute_cost(
            self,
            fn_cost_multiplier=1,
            fp_cost_multiplier=1,
            use_weighted_fn=False,
            use_weighted_fp=False):

        fn = self.rpf_dict["fn_w"] if use_weighted_fn else self.rpf_dict["fn"]
        fp = self.rpf_dict["fp_w"] if use_weighted_fp else self.rpf_dict["fp"]

        self.rpf_dict["fn_cost"] = fn_cost_multiplier * fn
        self.rpf_dict["fp_cost"] = fp_cost_multiplier * fp
        self.rpf_dict["cost"] = self.rpf_dict["fn_cost"] + self.rpf_dict[
            "fp_cost"]

    def plot_cost(
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
            dpi: Optional[int] = None,
            bootstrapped: bool = False,
            bootstrap_alpha: float = 0.15,
            bootstrap_color: str = "black",
            return_fig: bool = False) -> Optional[Tuple[plt.figure, plt.axes]]:
        """
        Plot the "Misclassification Cost" curve. Note: `compute_cost` must
        be run first to obtain cost values.
        PARAMETERS
        ----------

          title -- title of plot (default: "Misclassification Cost").
          cmap -- colormap string specification (default: "rainbow").
          log_scale -- boolean to specify whether the x-axis should be
            scaled by a log10 transformation (default: False).

          x_axis -- name of key in rpf_dict that specifies which values
            to use for the x coordinates of the cost curve (default:
            "thresh").

          x_label -- [Optional] label to apply to x-axis. Defaults for
            common choices of x-axis will be supplied if no x-label
            override is supplied here (default: None).

          x_rng -- [Optional] specify an x-axis range of the form
            [min_value, max_value] to override the default range (default:
            None).

          y_label -- [Optional] label to apply to y-axis (default: "Cost").

          y_rng -- [Optional] specify a y-axis range of the form
            [min_value, max_value] to override the default range (default:
            None).

          color_by -- name of key in rpf_dict that specifies which values
            to use when coloring points along the cost curve (default:
            "frac").

          cbar_rng -- [Optional] specify a color bar range of the form
            [min_value, max_value] to override the default range (default:
            None).

          cbar_label -- [Optional] custom label to apply to the color bar. If
            None is supplied, the default ("Fraction Flagged") will be used
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

          return_fig -- [Optional] if set to True, will return (fig, ax) as
            a tuple instead of plotting the figure.

        """
        assert "cost" in self.rpf_dict, "Run `compute_cost` first."

        # Create figure
        fig = plt.figure(figsize=(10, 6), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)

        # Make Color Bar
        if cbar_rng is not None:
            [vmin, vmax] = cbar_rng
        else:
            vmin = 0.0 if color_by == "frac" else np.min(
                self.rpf_dict[color_by])
            vmax = 1.0 if color_by == "frac" else np.max(
                self.rpf_dict[color_by])
        norm = matplotlib.colors.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ticks=np.linspace(vmin, vmax, 11))
        label = "Threshold Value" if cbar_label == None else cbar_label
        cbar.set_label("Fraction Flagged" if color_by == "frac" else label)

        # Make scatter plot
        cost = self.rpf_dict["cost"]
        x = self.rpf_dict[x_axis]
        colors = self.rpf_dict[color_by]

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
            zorder=int(1E4))

        # Plot faint bootstrapped curves
        if bootstrapped:
            x_boot = x[:, 1:] if x.shape[1] > 1 else x
            cost_boot = cost[:, 1:]
            for i in range(self.num_bootstrap_samples):
                x_vals = x_boot if x_boot.shape[1] == 1 else x_boot[:, i]
                ax.plot(
                    np.log10(x_vals) if log_scale else x_vals,
                    cost_boot[:, i],
                    alpha=bootstrap_alpha,
                    color=bootstrap_color,
                    linewidth=1)

        # Set x limits
        if not log_scale and x_axis in [
            "tpr",
            "fpr",
            "tpr_w",
            "fpr_w",
            "frac",
            "precision"]:
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
            display(fig)
            plt.gcf().clear()
            plt.close()
            return