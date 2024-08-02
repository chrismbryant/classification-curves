from typing import Iterable, Literal, Tuple, Union

import numpy as np
import pandas as pd
import typing_extensions
from scipy import stats

# Define custom types to simplify type annotations
Count = Union[int, np.int64, np.ndarray, pd.Series]
CIResult = Union[
    Tuple[float, float, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


class BinomialCI:
    """A class to compute the confidence interval for a single proportion.

    This computes a confidence interval for a binomial distribution (sequence
    of independent Bernoulli trials).

    See:
      * https://www.statsmodels.org/devel/_modules/statsmodels/stats/proportion.html#proportion_confint
      * https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Example
    -------
    >>> ci = BinomialCI()

    With scalar inputs:
    >>> ci.get_ci(x=10, n=100, conf=0.95)
    (0.1, 0.049004689221485945, 0.1762225977400227)

    With array inputs:
    >>> ci.get_ci(np.array([1, 10]), np.array([10, 50]))
    (array([0.1, 0.2]),
    array([0.00252858, 0.10030224]),
    array([0.44501612, 0.33718311]))

    With DataFrame inputs:
    >>> df = pd.DataFrame({"x": [1, 10], "n": [10, 50]})
    >>> df["avg"], df["lower"], df["upper"] = ci.get_ci(df["x"], df["n"])
    >>> print(df)
        x   n  avg     lower     upper
    0   1  10  0.1  0.002529  0.445016
    1  10  50  0.2  0.100302  0.337183
    """

    def __init__(self) -> None:
        self.ci_methods = {
            "normal": self._get_ci_normal,
            "beta": self._get_ci_beta,
            "exact": self._get_ci_beta,
        }

    def get_ci(
        self,
        x: Count,
        n: Count,
        conf: float = 0.95,
        method: str = "beta",
    ) -> CIResult:
        """
        Compute a confidence interval for a binomial distribution (sequence of
        independent Bernoulli trials). This typically describes the confidence
        we have in the proportion of trials which were "successful", but in
        principle, this can be used for any proportion (a statistic which has
        an integer numerator and integer denominator).

        Parameters
        ----------
        x
            Number of successes (i.e. the numerator).
        n
            Number of trials (i.e. the denominator).
        conf
            Confidence level between 0 and 1.
        method
            Method to use for computing the confidence interval. Options are:
            "normal", "beta" (= "exact").
        """
        self._validate_x_and_n(x, n)
        x, n = self._coerce_x_and_n_dtypes(x, n)
        alpha = self._get_alpha(conf)
        if method in self.ci_methods:
            return self.ci_methods[method](x, n, alpha)
        else:
            raise NotImplementedError("Method '%s' is not available." % method)

    @staticmethod
    def _validate_x_and_n(x: Count, n: Count) -> None:
        assert type(x) in typing_extensions.get_args(Count), type(x)
        assert type(n) in typing_extensions.get_args(Count), type(n)
        _x = np.array([x]) if type(x) in [int, np.int64] else x
        _n = np.array([n]) if type(n) in [int, np.int64] else n
        _x = np.array(x.astype(int)) if type(x) == pd.Series else _x
        _n = np.array(n.astype(int)) if type(n) == pd.Series else _n
        assert (_x <= _n).all(), "`x` must be less than or equal to `n`"  # type: ignore
        assert (_x >= 0).all(), "`x` must be greater than or equal to 0"  # type: ignore
        assert (_n >= 0).all(), "`n` must be greater than or equal to 0"  # type: ignore

    @staticmethod
    def _coerce_x_and_n_dtypes(x: Count, n: Count) -> Tuple[Count, Count]:
        """Coerce x and n to standard int types if they belong to a series."""
        x = x.astype(int) if type(x) == pd.Series else x
        n = n.astype(int) if type(n) == pd.Series else n
        return x, n

    @staticmethod
    def _get_avg(x: Count, n: Count) -> Union[float, np.ndarray]:
        if type(x) == int and type(n) == int and n == 0:
            return np.nan
        else:
            return x / n  # type: ignore

    @staticmethod
    def _get_alpha(conf: float) -> float:
        assert conf > 0 and conf < 1, "`conf` must be between 0 and 1"
        return 1 - conf

    def _get_ci_normal(self, x: Count, n: Count, alpha: float) -> CIResult:
        avg = self._get_avg(x, n)
        z = stats.norm.isf(alpha / 2)
        std = np.sqrt(avg * (1 - avg) / n)
        dist = z * std
        lower = avg - dist
        upper = avg + dist
        return (avg, lower, upper)  # type: ignore

    def _get_ci_beta(self, x: Count, n: Count, alpha: float) -> CIResult:
        avg = self._get_avg(x, n)
        lower = stats.beta.ppf(alpha / 2, x, n - x + 1)
        upper = stats.beta.isf(alpha / 2, x + 1, n - x)
        if not np.shape(lower):
            lower = 0 if x == 0 else lower
            upper = 1 if x == n else upper
        else:
            lower[x == 0] = 0
            upper[x == n] = 1
        return (avg, lower, upper)  # type: ignore


class BetaCI:
    """A class to compute the confidence interval for a sequence of proportions.

    Each example in the sample is assumed to be a proportion in the range
    [0, 1], drawn from a beta distribution.

    See
      * https://statproofbook.github.io/P/beta-mome.html
      * https://en.wikipedia.org/wiki/Beta-binomial_distribution

    NOTE: the beta distribution approach does not solve the problem of zero-
    width confidence intervals when the variance of the sample data is 0. In
    the normal approximation case, 0-variance leads to a 0-width confidence
    interval, and in the beta distribution case, 0-variance leads to an
    undefined beta distribution (so we set the width to 0 in this case).

    NOTE(chris): I've performed the same "+1" correction to the alpha and beta
    parameters in the `isf` and `ppf` computations, respectively, as is done
    in the `BinomialCI` class. Without this correction, the confidence interval
    is not guaranteed to contain the sample mean (since it's only guaranteed
    to contain the distribution's median), especially when the provided
    confidence level is low. It's not clear to me whether the correction here
    is valid.

    Example
    -------
    >>> ci = BetaCI()

    >>> ci.get_ci([0, 0.1, 0.2], conf=0.95)
    (0.1, 0.001286388572857717, 0.3672915958685548)

    >>> ci.get_ci([0, 0.1, 0.2], method="normal")
    (0.1, -0.09599639845400539, 0.29599639845400544)

    >>> df = pd.DataFrame(
    ...     {
    ...         "a": [1, 1, 2, 2, 2, 2],
    ...         "b": [0.1, 0.2, 0.1, 0.2, 0.3, None],
    ...     }
    ... )
    >>> df.groupby("a")["b"].apply(lambda x: ci.get_ci(x)).reset_index()
       a                                                b
    0  1  (0.15, 0.03934782717676056, 0.3514633807376342)
    1  2  (0.2, 0.04331199871732203, 0.48089116422177497)
    """

    def __init__(self) -> None:
        """Initialize the BetaCI class."""
        self.ci_methods = {
            "mom": self._get_ci_mom,
            "normal": self._get_ci_normal,
        }

    def get_ci(
        self,
        x: Iterable[float],
        conf: float = 0.95,
        method: Literal["mom", "normal"] = "mom",
    ) -> Tuple[float, float, float]:
        """Compute a confidence interval for a beta distribution.

        Parameters
        ----------
        x : Iterable[float]
            A sequence of values between 0 and 1.
        conf : float
            Confidence level between 0 and 1.
        method : Literal["mom", "normal"]
            Method to use for computing the confidence interval. Options are:
            "mom" (method of moments), "normal" (normal approximation).
        """
        return self.ci_methods[method](x, conf)

    def _get_ci_mom(
        self,
        x: Iterable[float],
        conf: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Get the confidence interval using the method of moments.

        Estimate the parameters of a beta distribution using the method of
        moments, then compute the interval containing the central `conf`
        fraction of that distribution.
        """
        alpha = get_alpha(conf)
        sample_mean, sample_var = self._get_mean_and_var(x)
        if sample_var == 0 or len(x) == 1:  # type: ignore
            lower = upper = sample_mean
        else:
            alpha_param = sample_mean * (
                (sample_mean * (1 - sample_mean)) / sample_var - 1
            )
            beta_param = (1 - sample_mean) * (
                (sample_mean * (1 - sample_mean)) / sample_var - 1
            )
            lower = stats.beta.ppf(alpha / 2, alpha_param, beta_param + 1)
            upper = stats.beta.isf(alpha / 2, alpha_param + 1, beta_param)
        return sample_mean, lower, upper

    def _get_ci_normal(
        self,
        x: Iterable[float],
        conf: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Get the confidence interval using a normal approximation."""
        alpha = get_alpha(conf)
        sample_mean, sample_var = self._get_mean_and_var(x)
        std = np.sqrt(sample_var)
        z = stats.norm.isf(alpha / 2)
        dist = z * std
        lower = sample_mean - dist
        upper = sample_mean + dist
        return sample_mean, lower, upper

    @staticmethod
    def _get_mean_and_var(x: Iterable[float]) -> Tuple[float, float]:
        """Get the mean and unbiased sample variance of a sequence of values."""
        arr = np.array(x, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if (arr < 0).any() or (arr > 1).any():
            raise ValueError("Values must be between 0 and 1")
        mean = np.mean(arr)
        var = np.var(arr, ddof=1) if len(arr) > 1 else 0
        return mean, var


def get_alpha(conf: float) -> float:
    """Get the alpha value for a given confidence level."""
    assert conf > 0 and conf < 1, "`conf` must be between 0 and 1"
    return 1 - conf
