from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.spatial import ConvexHull


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
    def __init__(self, data: np.array):

        assert data.shape[0] == 2, \
            f"Data must be of shape 2xM, not {data.shape}."

        self.data = data
        self.conf = None
        self.ellipse_data = None
        self.ellipse_patch = None

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
            "angle": angle
        }

        return self.ellipse_data

    def create_ellipse_patch(
            self,
            conf: float = 0.95,
            color: str = "black",
            alpha: float = 0.2,
            ax: Optional[plt.axes] = None) -> patches.Ellipse:
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
            color=color)

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
            s=20)


class EllipsePairHull:

    def __init__(
        self,
        ellipse_0: patches.Ellipse,
        ellipse_1: patches.Ellipse,
        left_to_right: bool = True,
    ) -> None:
        """A class to compute the convext hull for a pair of ellipses.
        
        Specifically, this computes "lower" and "upper" line segments that
        complete the convext hull of two ellipses.

        Parameters
        ----------
        ellipse_0 : patches.Ellipse
            First ellipse.
        ellipse_1 : patches.Ellipse
            Second ellipse.
        left_to_right : bool, optional
            This controls whether the line segments are specified from left to
            right or right to left.
        """
        self.ellipse_0 = ellipse_0
        self.ellipse_1 = ellipse_1
        self.left_to_right = left_to_right
    
    def get_hull_segments(self) -> List[np.ndarray]:
        """Compute the lower and upper hull segments.

        Returns
        -------
        List[np.ndarray]
            List of two line segments, each specified as a 2x2 array of points,
            where each row is an (x, y) coordinate.
        """
        # Get the ellipse points and indices corresponding to the hull
        points, hull = self.get_points_and_hull()
        num_points_0 = self.ellipse_0.get_verts().shape[0]

        # Find points where the hull switches between the ellipses
        switches = np.abs(np.diff((hull < num_points_0).astype(int)))
        switches = np.where(switches)[0]
        segments = []
        for switch in switches:
            segment = np.stack([points[hull[switch]], points[hull[switch + 1]]])
            segments.append(segment)
        if len(segments) == 1:
            segments.append([hull[-1], hull[0]])
        
        # Sort the segments
        segments = self.lower_then_upper(segments)
        segments = self.order_horizontally(segments)

        return segments
    
    def get_points_and_hull(self) -> Tuple[np.ndarray, List[int]]:
        """Get the ellipse points and hull indices.
        
        Returns
        -------
        np.ndarray
            Array of points defining the ellipses.
        List[int]
            List of indices describe which ellipse points correspond to the
            hull (in counterclockwise order).
        """
        # Get the points on the ellipses
        e0_points = self.ellipse_0.get_verts()
        e1_points = self.ellipse_1.get_verts()
        points = np.concatenate([e0_points, e1_points])

        # Counterclockwise ordering of points in the hull
        hull = ConvexHull(points).vertices

        return points, hull
        
    @staticmethod
    def lower_then_upper(segments: List[np.ndarray]) -> List[np.ndarray]:
        """Return the lower then upper segments.
        
        Parameters
        ----------
        segments : List[np.ndarray]
            List of line segments. Each element in this list is a 2x2 array,
            where each row is an (x, y) coordinate.
        
        Returns
        -------
        List[np.ndarray]
            List of line segments.
        """
        y_mean_0 = np.mean(segments[0][:, 1])
        y_mean_1 = np.mean(segments[1][:, 1])
        if y_mean_0 < y_mean_1:
            return segments
        else:
            return segments[::-1]
    
    def order_horizontally(
        self,
        segments: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Define each segment from left to right.
        
        Parameters
        ----------
        segments : List[np.ndarray]
            List of line segments. Each element in this list is a 2x2 array,
            where each row is an (x, y) coordinate.
        
        Returns
        -------
        List[np.ndarray]
            List of line segments.
        """
        new_segments = []
        for segment in segments:
            if segment[0, 0] > segment[1, 0]:
                new_segment = segment[::-1]
            else:
                new_segment = segment
            if self.left_to_right:
                new_segments.append(new_segment)
            else:
                new_segments.append(new_segment[::-1])
        return new_segments
