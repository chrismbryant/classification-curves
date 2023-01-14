from typing import Dict, Optional

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt


class CovarianceEllipseGenerator:
    """A class to generate a stylized covariance elipse.

    Given a collection of 2D points that are assumed to be distributed
    according to a bivariate normal distribution, compute and plot an
    elliptical confidence region representing the distribution of the points.

    Parameters
    ----------
    data
        (2, M)-dim numpy array.

    Examples
    --------
    >>> data = ...
    >>> ax = ...
    >>> ceg = CovarianceEllipseGenerator(data)
    >>> ceg.create_ellipse_patch(conf = 0.95, ax = ax)
    >>> ceg.add_ellipse_center(ax)
    """

    def __init__(self, data: np.ndarray):

        assert data.shape[0] == 2, f"Data must be of shape 2xM, not {data.shape}."

        self.data = data
        self.conf: float
        self.ellipse_data: Dict[str, float]
        self.ellipse_patch: patches.Ellipse

    def compute_cov_ellipse(self, conf: float = 0.95) -> Dict[str, float]:
        """Compute covariance ellipse geometry.

        Given a collection of 2D points, compute an elliptical confidence
        region representing the distribution of the points. Find the
        eigendecomposition of the covariance matrix of the data. The
        eigenvectors point in the directions of the ellipses axes. The
        eigenvalues specify the variance of the distribution in each of the
        principal directions. The 95% confidence interval in 2D spans 2.45
        standard deviations in each direction, so the width of a 95% confidence
        ellipse in a principal direction is found by taking 4.9 *
        sqrt(variance) in that direction.

        Parameters
        ----------
        conf
            Confidence level.

        Returns
        -------
        dict
            Dictionary of data to describe resulting confidence ellipse: {
                "x_center": horizontal value of ellipse center
                "y_center": vertical value of ellipse center
                "width": diameter of ellipse in first principal direction
                "height": diameter of ellipse in second principal direction
                "angle": counterclockwise rotation angle of ellipse from
                    horizontal (in degrees)
            }
        """
        self.conf = conf

        center = np.mean(self.data, axis=1)
        [x_center, y_center] = center.tolist()
        c = np.cov(self.data)
        (eigenval, eigenvec) = np.linalg.eig(c)
        angle = np.arctan(eigenvec[1, 0] / eigenvec[0, 0]) * 180 / np.pi
        num_std = np.sqrt(-2 * np.log(1 - conf))
        [width, height] = 2 * num_std * np.sqrt(eigenval)

        self.ellipse_data = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "angle": angle,
        }

        return self.ellipse_data

    def create_ellipse_patch(
        self,
        conf: float = 0.95,
        color: str = "black",
        alpha: float = 0.2,
        ax: Optional[plt.axes] = None,
    ) -> patches.Ellipse:
        """Create covariance ellipse Matplotlib patch.

        Create a Matplotlib ellipse patch for a specified confidence level.
        Add resulting patch to ax if supplied.

        Parameters
        ----------
        conf
            Confidence level.
        color
            Color of ellipse fill.
        alpha
            Opacity of ellipse fill.
        ax
            Matplotlib axis object.

        RETURNS
        -------
        patches.Ellipse
            Matplotlib ellipse patch.
        """
        if self.conf != conf:
            self.compute_cov_ellipse(conf)

        x_center = self.ellipse_data["x_center"]
        y_center = self.ellipse_data["y_center"]
        width = self.ellipse_data["width"]
        height = self.ellipse_data["height"]
        angle = self.ellipse_data["angle"]

        self.ellipse_patch = patches.Ellipse(
            (x_center, y_center),
            width,
            height,
            angle=angle,
            linewidth=2,
            fill=True,
            alpha=alpha,
            zorder=5000,
            color=color,
        )

        if ax:
            ax.add_patch(self.ellipse_patch)

        return self.ellipse_patch

    def add_ellipse_center(self, ax: plt.axes):
        """Add covariance ellipse patch to existing plot.

        Given an input Matplotlib axis object, add an opaque white dot at
        the center of the computed confidence ellipse.

        Parameters
        ----------
        ax
            Matplotlib axis object.
        """
        ax.scatter(
            self.ellipse_data["x_center"],
            self.ellipse_data["y_center"],
            color="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=10000,
            alpha=1,
            s=20,
        )
