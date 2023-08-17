import itertools
import logging
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import psutil
from numpy.random import default_rng
from scipy.integrate import trapz
from tqdm import tqdm
from typing_extensions import Literal

from clscurves.config import MetricsAliases
from clscurves.plotter.cost import CostPlotter
from clscurves.plotter.dist import DistPlotter
from clscurves.plotter.pr import PRPlotter
from clscurves.plotter.prg import PRGPlotter
from clscurves.plotter.rf import RFPlotter
from clscurves.plotter.roc import ROCPlotter
from clscurves.utils import MetricsResult

LOG = logging.getLogger(__name__)

NullFillMethod = Literal["0", "1", "imb", "prob"]


class MetricsGenerator(
    ROCPlotter,
    PRPlotter,
    PRGPlotter,
    RFPlotter,
    CostPlotter,
    DistPlotter,
    MetricsAliases,
):
    """A class to generate classification curve metrics.

    A class for computing Precision/Recall/Fraction metrics across a binary
    classification algorithm's full range of discrimination thresholds, and
    plotting those metrics as ROC (Receiver Operating Characteristic), PR
    (Precision & Recall), or RF (Recall & Fraction) plots. The input data
    format for this class is a PySpark DataFrame with at least a column of
    labels and a column of scores (with an optional additional column of label
    weights).
    """

    def __init__(
        self,
        predictions_df: Optional[pd.DataFrame] = None,
        max_num_examples: int = 100000,
        label_column: str = "label",
        score_column: str = "probability",
        weight_column: Optional[str] = None,
        score_is_probability: bool = True,
        reverse_thresh: bool = False,
        num_bootstrap_samples: int = 0,
        imbalance_multiplier: float = 1,
        null_prob_column: Optional[str] = None,
        null_fill_method: Optional[NullFillMethod] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiating this class computes all the metrics.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Input DataFrame which must contain a column of labels (integer
            column of 1s and 0s) and a column of scores (either a dense vector
            column with two elements [prob_0, prob_1] or a real-valued column
            of scores).
        max_num_examples : int
            Max number of rows to sample to prevent numpy memory limits from
            being exceeded.
        label_column : str
            Name of the column containing the example labels, which must all be
            either 0 or 1.
        score_column : str
            Name of the column containing the example scores which rank model
            predictions. Even though binary classification models typically
            output probabilities, these scores need not be bounded by 0 and 1;
            they can be any real value. This column can be either a real value
            numeric type or a 2-element vector of probabilities, with the first
            element being the probability that the example is of class 0 and
            the second element that the example is of class 1.
        weight_column : Optional[str]
            Name of the column containing label weights associated with each
            example. These weights are useful when the cost of classifying an
            example incorrectly varies from example to example; see fraud, for
            instance: getting high dollar value cases wrong is more costly than
            getting low dollar value cases wrong, so a good measure of recall
            is, "How much money did we catch", not "How many cases did we
            catch". If no column name is specified, all weights will be set to
            1.
        score_is_probability : bool
            Specifies whether the values in the score column are bounded by 0
            and 1. This controls how the threshold range is determined. If
            true, the threshold range will sweep from 0 to 1. If false, it will
            sweep from the minimum to maximum score value.
        num_bootstrap_samples : int
            Number of bootstrap samples to generate from the original data when
            computing performance metrics.
        reverse_thresh : bool
            Boolean indicating whether the score threshold should be treated as
            a lower bound on "positive" predictions (as is standard) or instead
            as an upper bound. If `True`, the threshold behavior will be
            reversed from standard so that any prediction falling BELOW a score
            threshold will be marked as positive, with all those falling above
            the threshold marked as negative.
        imbalance_multiplier : float
            Positive value to artifically increase the positive class example
            count by a multiplicative weighting factor. Use this if you're
            generating metrics for a data distribution with a class imbalance
            that doesn't represent the true distribution in the wild. For
            example, if you trained on a 1:1 artifically balanced data set, but
            you have a 10:1 class imbalance in the wild (i.e. 10 negative
            examples for every 1 positive example), set the
            ``imbalance_multiplier`` value to 10.
        null_prob_column : Optional[str]
            Column containing calibrated label probabilities to use as the
            sampling distribution for imputing null label values. We provide
            this argument so that you can evaluate a possibly-uncalibrated
            model score (specified by the `score_column` argument) on a
            different provided calibrated label distribution. If this argument
            is `None`, then the ``score_column`` will be used as the estimated
            label distribution when necessary.
        null_fill_method : Optional[NullFillMethod]
            Methods to use when filling in null label values. Possible values:
                * "0" - fill with 0
                * "1" - fill with 1
                * "imb" - fill randomly according to the class imbalance of
                    labeled examples
                * "prob" - fill randomly according to the ``score_column``
                    probability distribution or the ``null_prob_column``
                    probability distribution, if provided.
            If a method is provided, once the default metrics DF is computed
            without imputing any null labels, then a new metrics DF will be
            computed for each method and stored in a ``curves_imputed``
            object. If not, only the default metrics DF will be computed.
        seed : Optional[int]
            Random seed for bootstrapping.

        Examples
        --------
        >>> mg = MetricsGenerator(
                predictions_df,
                label_column="label",
                score_column="score",
                weight_column="weight",
                score_is_probability=False,
                reverse_thresh=False,
                num_bootstrap_samples=20,
                seed=123,
            )

        >>> mg.plot_pr(bootstrapped=True)
        >>> mg.plot_roc()
        """

        self.predictions_df = predictions_df
        self.max_num_examples = max_num_examples
        self.label_column = label_column
        self.score_column = score_column
        self.weight_column = weight_column
        self.score_is_probability = score_is_probability
        self.reverse_thresh = reverse_thresh
        self.num_bootstrap_samples = num_bootstrap_samples
        self.imbalance_multiplier = imbalance_multiplier
        self.null_prob_column = null_prob_column
        self.null_fill_method = null_fill_method
        self.null_probabilities = None
        self.seed = seed

        # Metrics to be populated
        self.metrics: MetricsResult

        # Print imbalance multiplier warning
        if self.imbalance_multiplier != 1:
            print(f"Artificial imbalance multiplier: {self.imbalance_multiplier}")

        if null_fill_method not in [None, "0", "1", "imb", "prob"]:
            raise ValueError(
                f"Invalid null_fill_method: {null_fill_method}. Must be one of "
                f"None, '0', '1', 'imb', or 'prob'."
            )

        if predictions_df is not None:
            self.compute_all_metrics(predictions_df)

    def _get_rng(self, idx: int) -> np.random.Generator:
        """Get random generator for bootstrap sampling."""
        seed = None if self.seed is None else self.seed + idx + 1
        return default_rng(seed)

    def compute_all_metrics(
        self,
        predictions_df: pd.DataFrame,
        return_results: bool = False,
    ) -> Optional[MetricsResult]:
        """Compute all metrics."""
        print("Computing metrics...")

        # Check if there are any null labels
        labels_contain_null = predictions_df[self.label_column].isnull().any()
        if labels_contain_null:
            LOG.warning(" >>> WARNING: Labels contain null values.")

        # Keep only relevant columns
        cols = [self.label_column, self.score_column]
        if self.weight_column:
            cols.append(self.weight_column)
        df_sample = predictions_df[cols].copy()

        # Sample input data if too large
        if len(df_sample) > self.max_num_examples:
            rng = self._get_rng(-1)
            seed = int(rng.random() * 2**32)
            df_sample = df_sample.sample(
                n=self.max_num_examples,
                random_state=seed,
            )

        # List configurations to compute
        bootstrap_sample_options = [None, *range(self.num_bootstrap_samples)]
        null_fill_methods = list(set([None, self.null_fill_method]))
        all_options = itertools.product(
            bootstrap_sample_options,
            null_fill_methods,
        )

        # Compute metrics
        curves = pd.DataFrame()
        scalars = pd.DataFrame()
        num_workers = psutil.cpu_count()
        with Pool(num_workers) as pool:
            for metrics in tqdm(
                pool.starmap(
                    self.compute_metrics,
                    [
                        (df_sample, *options, self._get_rng(i))
                        for i, options in enumerate(all_options)
                    ],
                )
            ):
                curves = pd.concat([curves, metrics.curves])
                scalars = pd.concat([scalars, metrics.scalars])

        # Separate imputed metrics from default metrics
        curves_default = curves.loc[curves["_null_fill_method"].isnull()]
        scalars_default = scalars.loc[scalars["_null_fill_method"].isnull()]
        curves_imputed = curves.loc[curves["_null_fill_method"].notnull()]
        scalars_imputed = scalars.loc[scalars["_null_fill_method"].notnull()]

        metrics = MetricsResult(
            curves=curves_default,
            scalars=scalars_default,
            curves_imputed=curves_imputed,
            scalars_imputed=scalars_imputed,
        )

        print("Metrics computation complete.")

        # Optional return
        if return_results:
            return metrics

        self.metrics = metrics

        return None

    def compute_metrics(
        self,
        predictions_df: pd.DataFrame,
        bootstrap_sample: Optional[int] = None,
        null_fill_method: Optional[NullFillMethod] = None,
        rng: np.random.Generator = default_rng(),
    ) -> MetricsResult:
        """Compute metrics for a single bootstrap sample."""

        # Keep or drop nulls
        if null_fill_method is None:
            _df = predictions_df.dropna(subset=[self.label_column])
        else:
            _df = predictions_df

        # Make a bootstrap
        if bootstrap_sample is not None:
            _df = self._make_bootstrap(_df, rng)

        # Impute null labels
        labels = _df[self.label_column].values
        if null_fill_method is not None:
            labels = self._fill_null_labels(
                labels=labels,
                null_fill_method=null_fill_method,
                null_probs=self.null_probabilities,
                rng=rng,
            )
            assert not np.isnan(labels).any()

        # Compute metrics
        metrics = self._compute_metrics(
            scores=_df[self.score_column].values,
            labels=labels,
            weights=_df[self.weight_column].values if self.weight_column else None,
            reverse_thresh=self.reverse_thresh,
        )

        # Attach metadata to output
        metrics.curves["_bootstrap_sample"] = bootstrap_sample
        metrics.curves["_null_fill_method"] = null_fill_method
        metrics.scalars["_bootstrap_sample"] = bootstrap_sample
        metrics.scalars["_null_fill_method"] = null_fill_method

        return metrics

    def _compute_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
        reverse_thresh: bool = False,
    ) -> MetricsResult:
        """Compute metrics.

        Parameters
        ----------
        scores : np.ndarray
            Array of scores.
        labels : np.ndarray
            Binary array of labels. If labels are real-valued, then all values
            > 0 will be treated as positive examples, and all values <= 0 will
            be treated as negative examples.
        weights : Optional[np.ndarray]
            Array of weights associated with each example.
        reverse_thresh : bool
            Boolean indicating whether the score threshold should be treated as
            a lower bound on "positive" predictions (as is standard) or instead
            as an upper bound. If ``True``, the threshold behavior will be
            reversed from standard so that any prediction falling BELOW a score
            threshold will be marked as positive, with all those falling above
            the threshold marked as negative.

        Returns
        -------
        MetricsResult
            A class containing the computed metrics.
        """
        if np.isnan(labels).any():
            raise ValueError("Labels contain null values.")

        # Put arrays into DataFrame, treating scores as thresholds
        df = pd.DataFrame(
            {
                "thresh": scores,
                "label": (labels > 0).astype(int),
                "weight": weights if weights is not None else np.ones(len(scores)),
            },
        )
        df["weight_pos"] = df["label"] * df["weight"]

        # Collapse identical threshold values, and sort
        df = (
            df.groupby("thresh")
            .agg(
                label=("label", "sum"),
                weight=("weight", "sum"),
                weight_pos=("weight_pos", "sum"),
                num=("label", "count"),
            )
            .reset_index()
        )
        df = df.sort_values("thresh", ascending=not reverse_thresh)

        # Compute scalar values
        _num_examples = df["num"].sum()
        _num_examples_pos = df["label"].sum()
        num_examples_pos = _num_examples_pos * self.imbalance_multiplier
        num_examples_neg = _num_examples - _num_examples_pos
        num_examples = num_examples_pos + num_examples_neg
        _tot_weight = df["weight"].sum()
        _tot_weight_pos = df["weight_pos"].sum()
        tot_weight_pos = _tot_weight_pos * self.imbalance_multiplier
        tot_weight_neg = _tot_weight - _tot_weight_pos
        tot_weight = tot_weight_pos + tot_weight_neg
        imbalance = num_examples_pos / (num_examples_pos + num_examples_neg)

        # Add extra threshold value
        epsilon = 1e-6
        multiplier = 1 if reverse_thresh else -1
        extra_thresh = df.iloc[0]["thresh"] + multiplier * epsilon
        extra_row = pd.DataFrame([[extra_thresh, 0, 0, 0, 0]], columns=df.columns)
        df = pd.concat([extra_row, df]).reset_index(drop=True)

        # Compute confusion matrix
        df["pred_neg"] = df["num"].cumsum()
        df["pred_pos"] = _num_examples - df["pred_neg"]
        df["fn"] = df["label"].cumsum()
        df["tn"] = df["pred_neg"] - df["fn"]
        df["tp"] = _num_examples_pos - df["fn"]
        df["fp"] = _num_examples - df["pred_neg"] - df["tp"]
        df["pred_pos"] *= self.imbalance_multiplier
        df["fn"] *= self.imbalance_multiplier
        df["tp"] *= self.imbalance_multiplier

        df["recall"] = df["tp"] / (df["tp"] + df["fn"])
        df["precision"] = df["tp"] / (df["tp"] + df["fp"])
        df["frac"] = df["pred_pos"] / num_examples
        df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])
        df["fpr"] = df["fp"] / num_examples_neg
        df["fdr"] = df["fp"] / (df["fp"] + df["tp"])
        df["recall_gain"] = self._compute_gain(df["recall"], imbalance)
        df["precision_gain"] = self._compute_gain(df["precision"], imbalance)

        # Compute weighted confusion matrix
        df["pred_neg_w"] = df["weight"].cumsum()
        df["pred_pos_w"] = _tot_weight - df["pred_neg_w"]
        df["fn_w"] = df["weight_pos"].cumsum()
        df["tn_w"] = df["pred_neg_w"] - df["fn_w"]
        df["tp_w"] = _tot_weight_pos - df["fn_w"]
        df["fp_w"] = _tot_weight - df["pred_neg_w"] - df["tp_w"]
        df["pred_pos_w"] *= self.imbalance_multiplier
        df["fn_w"] *= self.imbalance_multiplier
        df["tp_w"] *= self.imbalance_multiplier

        df["recall_w"] = df["tp_w"] / (df["tp_w"] + df["fn_w"])
        df["precision_w"] = df["tp_w"] / (df["tp_w"] + df["fp_w"])
        df["frac_w"] = df["pred_pos_w"] / tot_weight
        df["fpr_w"] = df["fp_w"] / tot_weight_neg
        df["fdr_w"] = df["fp_w"] / (df["fp_w"] + df["tp_w"])

        # Fill nulls
        df.fillna({col: 0 for col in df.columns if col != "label"}, inplace=True)

        # Scalars
        scalars = {
            "num_examples": num_examples,
            "num_examples_pos": num_examples_pos,
            "num_examples_neg": num_examples_neg,
            "tot_weight": tot_weight,
            "tot_weight_pos": tot_weight_pos,
            "tot_weight_neg": tot_weight_neg,
            "imbalance": imbalance,
            "roc_auc": np.abs(trapz(df["recall"], df["fpr"])),
            "pr_auc": np.abs(trapz(df["precision"], df["recall"])),
            "rf_auc": np.abs(trapz(df["recall"], df["frac"])),
            "roc_auc_w": np.abs(trapz(df["recall_w"], df["fpr_w"])),
            "pr_auc_w": np.abs(trapz(df["precision_w"], df["recall_w"])),
            "rf_auc_w": np.abs(trapz(df["recall_w"], df["frac_w"])),
            "prg_auc": np.abs(trapz(df["precision_gain"], df["recall_gain"])),
        }
        scalars = pd.DataFrame([scalars])

        return MetricsResult(curves=df, scalars=scalars)

    def _make_bootstrap(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator = default_rng(),
    ) -> pd.DataFrame:
        """Make bootstrap sample.

        Sample with replacement from the input DataFrame to create a bootstrap
        sample of the same size as the input DataFrame.
        """
        random_state = (
            np.random.RandomState(int(rng.random() * 2**32)) if rng else None
        )
        return df.sample(
            n=len(df),
            replace=True,
            random_state=random_state,
        )

    def _fill_null_labels(
        self,
        labels: np.ndarray,
        null_fill_method: NullFillMethod,
        null_probs: Optional[np.ndarray] = None,
        rng: np.random.Generator = default_rng(),
    ) -> np.ndarray:
        """Fill null labels according to the specified method.

        Parameters
        ----------
        labels : np.ndarray
            Array of labels, some of which may be null.
        null_fill_method : NullFillMethod
            Method to use when filling null labels.
            * "0" -- fill unknown labels with 0.
            * "1" -- fill unknown labels with 1.
            * "imb" -- fill unknown labels with 0 or 1 probabilistically
                according to the class imbalance of the known labels.
            * "prob" -- fill unknown labels with 0 or 1 probabilistically
                according to the probability-calibrated model score.
        null_probs : Optional[np.ndarray]
            Array of probabilities to use when filling null labels. Only
            required if ``null_fill_method`` is "prob".
        """
        labels = labels.astype(float)
        imbalance = (labels > 0).sum() / len(labels)
        if null_fill_method == "0":
            return np.where(np.isnan(labels), 0, labels)
        if null_fill_method == "1":
            return np.where(np.isnan(labels), 1, labels)
        if null_fill_method == "imb":
            labels_from_imb = (rng.random(*labels.shape) < imbalance).astype(int)
            return np.where(np.isnan(labels), labels_from_imb, labels)
        if null_fill_method == "prob":
            if null_probs is None:
                raise ValueError(
                    "Must provide null_probs when using null_fill_method='prob'."
                )
            return np.where(
                np.isnan(labels),
                (rng.random(*labels.shape) < null_probs).astype(int),
                labels,
            )

    @staticmethod
    def _compute_gain(
        metric: np.ndarray,
        imbalance: float,
    ) -> np.ndarray:
        """Compute "gain".

        As defined in the "Precision-Recall-Gain" paper
        `here <https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd464\
        4bb2-Paper.pdf>`_.
        """
        return np.clip(
            (metric - imbalance) / ((1 - imbalance) * metric),
            a_min=0,
            a_max=1,
        )
