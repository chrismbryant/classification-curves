import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapz

from clscurves.config import MetricsAliases
from clscurves.plotter.cost import CostPlotter
from clscurves.plotter.dist import DistPlotter
from clscurves.plotter.pr import PRPlotter
from clscurves.plotter.prg import PRGPlotter
from clscurves.plotter.rf import RFPlotter
from clscurves.plotter.roc import ROCPlotter


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
        n_thresh: int = 500,
        max_num_examples: int = 100000,
        label_column: str = "label",
        score_column: str = "probability",
        weight_column: Optional[str] = None,
        score_is_probability: bool = True,
        reverse_thresh: bool = False,
        num_bootstrap_samples: int = 0,
        imbalance_multiplier: float = 1,
        null_prob_column: Optional[str] = None,
        null_fill_methods: Optional[List[str]] = None,
        seed: int = 1,
    ):
        """Instantiating this class computes all the metrics.

        PARAMETERS
        ----------
        predictions_df
            Input DataFrame which must contain a column of labels (integer
            column of 1s and 0s) and a column of scores (either a dense vector
            column with two elements [prob_0, prob_1] or a real-valued column
            of scores).
        n_thresh
            number of threshold values.
        max_num_examples
            Max number of rows to sample to prevent numpy memory limits from
            being exceeded.
        label_column
            Name of the column containing the example labels, which must all be
            either 0 or 1.
        score_column
            Name of the column containing the example scores which rank model
            predictions. Even though binary classification models typically
            output probabilities, these scores need not be bounded by 0 and 1;
            they can be any real value. This column can be either a real value
            numeric type or a 2-element vector of probabilities, with the first
            element being the probability that the example is of class 0 and
            the second element that the example is of class 1.
        weight_column
            Name of the column containing label weights associated with each
            example. These weights are useful when the cost of classifying an
            example incorrectly varies from example to example; see fraud, for
            instance: getting high dollar value cases wrong is more costly than
            getting low dollar value cases wrong, so a good measure of recall
            is, "How much money did we catch", not "How many cases did we
            catch". If no column name is specified, all weights will be set to
            1.
        score_is_probability
            Specifies whether the values in the score column are bounded by 0
            and 1. This controls how the threshold range is determined. If
            true, the threshold range will sweep from 0 to 1. If false, it will
            sweep from the minimum to maximum score value.
        num_bootstrap_samples
            Number of bootstrap samples to generate from the original data when
            computing performance metrics.
        reverse_thresh
            Boolean indicating whether the score threshold should be treated as
            a lower bound on "positive" predictions (as is standard) or instead
            as an upper bound. If `True`, the threshold behavior will be
            reversed from standard so that any prediction falling BELOW a score
            threshold will be marked as positive, with all those falling above
            the threshold marked as negative.
        imbalance_multiplier
            Positive value to artifically increase the negative class example
            count by a multiplicative weighting factor. Use this if you're
            generating metrics for a data distribution with a class imbalance
            that doesn't represent the true distribution in the wild. For
            example, if you trained on a 1:1 artifically balanced data set, but
            you have a 10:1 class imbalance in the wild (i.e. 10 negative
            examples for every 1 positive example), set the
            ``imbalance_multiplier`` value to 10.
        null_prob_column
            Column containing calibrated label probabilities to use as the
            sampling distribution for imputing null label values. We provide
            this argument so that you can evaluate a possibly-uncalibrated
            model score (specified by the `score_column` argument) on a
            different provided calibrated label distribution. If this argument
            is `None`, then the ``score_column`` will be used as the estimated
            label distribution when necessary.
        null_fill_methods
            List of methods to use when filling in null label values. Possible
            values:
                * "0" - fill with 0
                * "1" - fill with 1
                * "imb" - fill randomly according to the class imbalance of
                    labeled examples
                * "prob" - fill randomly according to the ``score_column``
                    probability distribution or the ``null_prob_column``
                    probability distribution, if provided.
            If a list of methods is provided, once the default metrics
            dictionary is computed without imputing any null labels, then a new
            metrics dict will be computed for each method and stored in an
            ``metrics_dict_imputed`` dictionary object. If not, only the
            default metrics dictionary will be computed.
        seed
            Random seed for bootstrapping.

        Examples
        --------
        >>> mg = MetricsGenerator(
            predictions_df,
            label_column = "label",
            score_column = "score",
            weight_column = "weight",
            score_is_probability = False,
            reverse_thresh = False,
            num_bootstrap_samples = 20)

        >>> mg.plot_pr(bootstrapped = True)
        >>> mg.plot_roc()
        """

        # Assign instance attributes
        self.N = n_thresh
        self.max_num_examples = max_num_examples
        self.label_column = label_column
        self.score_column = score_column
        self.weight_column = weight_column
        self.score_is_probability = score_is_probability
        self.reverse_thresh = reverse_thresh
        self.num_bootstrap_samples = num_bootstrap_samples
        self.imbalance_multiplier = imbalance_multiplier
        self.null_prob_column = null_prob_column
        self.null_fill_methods = null_fill_methods
        self.null_probabilities = None
        self.seed = seed
        self.metrics_dict: Dict[str, Any] = {}
        self.metrics_dict_imputed: Dict[str, Any] = {}

        # Set seed
        np.random.seed(self.seed)

        assert self.null_fill_methods is None or all(
            [m in ["0", "1", "imb", "prob"] for m in self.null_fill_methods]
        ), "Each null_fill_method must be in ['0', '1', 'imb', 'prob']."

        if predictions_df is not None:

            # Collect data into arrays
            self.predictions_df = predictions_df
            syw = self._collect_syw()

            # Assign thresholds
            self.thresholds = self._get_thresholds(syw["s"])

            # Bootstrap if necessary
            syw = self._make_bootstraps(syw)

            # Extract values
            self.scores = syw["s"]
            self.labels = syw["y"]
            self.weights = syw["w"]
            if "p" in syw:
                self.null_probabilities = syw["p"]

            # Compute standard classification curve metrics
            self.compute_metrics(self.scores, self.labels, self.weights)

            # Compute imputed-null classification curve metrics
            if self.null_fill_methods is not None:
                self.compute_metrics_with_unk()

    def _collect_syw(self) -> Dict[str, np.ndarray]:
        """
        Collect scores (s), labels (y), and weights (w) from a Pandas
        prediction DataFrame into 3 separate NumPy arrays. Convert all None
        labels to NaN after collecting.
        """
        syw = dict()

        syw["s"] = self.predictions_df[self.score_column].to_numpy().astype(np.float32)

        syw["y"] = self.predictions_df[self.label_column].to_numpy().astype(np.float32)

        # Set weight column to 1 if not specified
        if self.weight_column is not None:
            syw["w"] = (
                self.predictions_df[self.weight_column].to_numpy().astype(np.float32)
            )
        else:
            syw["w"] = np.ones(
                self.predictions_df[self.score_column].to_numpy().shape,
                dtype=np.float32,
            )

        # Only set p if the null prob column is specified
        if self.null_prob_column is not None:
            syw["p"] = (
                self.predictions_df[self.null_prob_column].to_numpy().astype(np.float32)
            )

        return syw

    def _make_bootstraps(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Construct matched bootstrap samples for all the arrays in an input
        dictionary if the number of bootstrap samples is set to a value greater
        than 0, then concatenate those bootstraps to the original arrays.
        Otherwise, reshape input arrays so that they are 2D (vertical vectors).
        """
        tot = len(arrays[list(arrays.keys())[0]])
        if self.num_bootstrap_samples > 0:
            print(f"Creating {self.num_bootstrap_samples} bootstrap samples...")
            ix = np.random.choice(np.arange(tot), (tot, self.num_bootstrap_samples))
            arrays = {
                key: np.concatenate([arr.reshape(tot, 1), arr[ix]], axis=1)
                for key, arr in arrays.items()
            }
        else:
            arrays = {key: arr.reshape(tot, 1) for key, arr in arrays.items()}

        return arrays

    def _get_thresholds(self, scores: np.ndarray) -> np.ndarray:
        """
        Given an array of scores, create an ordered list of threshold values
        which includes:
            1. The minimum input score,
            2. The maximum input score,
            3. A set of scores equally spaced between the min and max score
                value, and
            4. A set of scores equally spaced throughout the score quantile
                distribution.
        """
        num_examples = len(scores)
        score_bounds = (
            [0, 1] if self.score_is_probability else [min(scores), max(scores)]
        )  # noqa
        score_range = score_bounds[1] - score_bounds[0]
        thresh_equal = (
            score_bounds[0] + score_range * np.arange(self.N) / self.N
        )  # noqa
        indices = np.ndarray.astype(
            np.arange(self.N) * num_examples / self.N, int
        )  # noqa
        thresholds = np.sort(
            np.unique(
                np.concatenate(
                    [
                        np.array([score_bounds[0]]),
                        np.sort(scores)[indices],
                        thresh_equal,
                        np.array([score_bounds[1]]),
                    ]
                )
            )
        )
        thresholds = np.atleast_2d(thresholds).T

        return thresholds

    def _compute_pos_neg_dict(
        self, labels: np.ndarray, weights: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute number of positive-, negative-, and un-labeled examples,
        and the total weighted amount covered by each label type.
        """

        # Print imbalance multiplier warning
        if self.imbalance_multiplier != 1:
            print(f"Artificial imbalance multiplier: {self.imbalance_multiplier}")

        # Alias input parameters
        y = labels
        w = weights

        # Compute the label prior distributions
        pos_neg_dict = {
            "pos": np.sum(y == 1, axis=0),
            "neg": np.sum(y == 0, axis=0) * self.imbalance_multiplier,
            "unk": np.sum(np.isnan(y), axis=0),
            "pos_w": np.sum((y == 1) * w, axis=0),
            "neg_w": np.sum((y == 0) * w, axis=0) * self.imbalance_multiplier,
            "unk_w": np.sum(np.isnan(y) * w, axis=0),
        }

        # Compute class imbalance
        d = pos_neg_dict
        pos_neg_dict["imbalance"] = d["pos"] / (d["pos"] + d["neg"])

        return pos_neg_dict

    def _compute_tp_fp_dict(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        thresholds: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute unweighted and weighted TP and FP arrays, given input arrays of
        model scores, data labels, data weights, and threshold values. Also
        compute UP ("unknown positive") arrays if there are any records which
        have an unknown label (i.e. label = None).
        """
        print("Computing confusion matrices...")

        # Check if there are any null labels
        self.labels_contain_null = np.sum(np.isnan(labels)) > 0

        # Initialize empty lists TP and FP lists
        tp = []
        fp = []
        tp_w = []
        fp_w = []
        if self.labels_contain_null:
            print(" >>> WARNING: Label column contains some null values.")
            up = []
            up_w = []

        # Compute TP and FP at each threshold
        for t in thresholds:

            # Note: predictions shape = (num_examples, num_bootstrap_samples)
            predictions = (
                (scores <= t) if self.reverse_thresh else (scores >= t)
            ).astype(
                int
            )  # noqa
            tp_value = np.sum(
                np.logical_and(predictions == 1, labels == 1), axis=0
            ).T  # noqa
            fp_value = np.sum(
                np.logical_and(predictions == 1, labels == 0), axis=0
            ).T  # noqa
            tp_w_value = np.sum(
                weights * np.logical_and(predictions == 1, labels == 1), axis=0
            ).T
            fp_w_value = np.sum(
                weights * np.logical_and(predictions == 1, labels == 0), axis=0
            ).T

            tp.append(tp_value)
            fp.append(fp_value)
            tp_w.append(tp_w_value)
            fp_w.append(fp_w_value)

            # Account for unknown labels
            if self.labels_contain_null:

                up_value = np.sum(
                    np.logical_and(predictions == 1, np.isnan(labels)), axis=0
                ).T

                up_w_value = np.sum(
                    weights * np.logical_and(predictions == 1, np.isnan(labels)), axis=0
                ).T

                up.append(up_value)
                up_w.append(up_w_value)

        # Transform lists to arrays
        tp_fp_dict = {
            "tp": np.ndarray(tp),
            "fp": np.ndarray(fp) * self.imbalance_multiplier,
            "tp_w": np.ndarray(tp_w),
            "fp_w": np.ndarray(fp_w) * self.imbalance_multiplier,
        }

        # Account for unknown labels
        if self.labels_contain_null:
            tp_fp_dict["up"] = np.array(up)
            tp_fp_dict["up_w"] = np.array(up_w)

        return tp_fp_dict

    def _compute_tn_fn_dict(
        self, tp_fp_dict: Dict[str, np.ndarray], pos_neg_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute complementary TN and FN metrics for given TP and FP metrics.
        """

        # Combine input dicts into single dict
        d = {**tp_fp_dict, **pos_neg_dict}

        # Compute standard complementary metrics
        tn_fn_dict = {
            "fn": d["pos"] - d["tp"],
            "tn": d["neg"] - d["fp"],
            "fn_w": d["pos_w"] - d["tp_w"],
            "tn_w": d["neg_w"] - d["fp_w"],
        }

        # Compute complementary metrics for unknown labels
        if self.labels_contain_null:
            tn_fn_dict["un"] = d["unk"] - d["up"]
            tn_fn_dict["un_w"] = d["unk_w"] - d["up_w"]

        return tn_fn_dict

    def _compute_confusion_dict(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        thresholds: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all aspects of the unweighted and weighted confusion matrix,
        given input arrays of model scores, data labels, data weights, and
        threshold values.
        """
        pos_neg_dict = self._compute_pos_neg_dict(labels, weights)
        tp_fp_dict = self._compute_tp_fp_dict(
            scores, labels, weights, thresholds
        )  # noqa
        tn_fn_dict = self._compute_tn_fn_dict(tp_fp_dict, pos_neg_dict)
        confusion_dict = {**pos_neg_dict, **tp_fp_dict, **tn_fn_dict}
        return confusion_dict

    @staticmethod
    def _compute_metrics_dict(
        confusion_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute performance metrics from TP and FP values, making sure to
        account for divide-by-zero cases in the precision values.
        """
        print("Computing performance metric curves...")
        warnings.filterwarnings("ignore")  # Ignore divide-by-zero warnings
        d = confusion_dict
        imb = d["imbalance"]
        tot = d["pos"] + d["neg"] + d["unk"]
        tot_w = d["pos_w"] + d["neg_w"] + d["unk_w"]

        recall = d["tp"] / d["pos"]
        precision = np.nan_to_num(d["tp"] / (d["tp"] + d["fp"]))

        def compute_gain(
            metric: np.ndarray, imbal: Union[np.ndarray, float]
        ) -> np.ndarray:
            return np.clip((metric - imbal) / ((1 - imbal) * metric), a_min=0, a_max=1)

        recall_gain = compute_gain(recall, imb)
        precision_gain = compute_gain(precision, imb)

        metrics_dict = {
            "tpr": recall,
            "fpr": d["fp"] / d["neg"],
            "tpr_w": d["tp_w"] / d["pos_w"],
            "fpr_w": d["fp_w"] / d["neg_w"],
            "frac": (d["tp"] + d["fp"] + d.get("up", 0)) / tot,
            "frac_w": (d["tp_w"] + d["fp_w"] + d.get("up_w", 0)) / tot_w,
            "precision": precision,
            "f1": np.nan_to_num(2 * precision * recall / (precision + recall)),
            "recall_gain": recall_gain,
            "precision_gain": precision_gain,
        }
        return metrics_dict

    @staticmethod
    def _compute_area_metrics_dict(
        metrics_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute single-number area-under-the-curve metrics for all the computed
        performance curves.
        """

        # Compute area under ROC, PR, and RF curves
        print("Computing area metrics...")
        d = metrics_dict
        area_metrics_dict = {
            "roc_auc": np.abs(trapz(d["tpr"], d["fpr"], axis=0)),
            "pr_auc": np.abs(trapz(d["precision"], d["tpr"], axis=0)),
            "rf_auc": np.abs(trapz(d["tpr"], d["frac"], axis=0)),
            "roc_auc_w": np.abs(trapz(d["tpr_w"], d["fpr_w"], axis=0)),
            "pr_auc_w": np.abs(trapz(d["precision"], d["tpr_w"], axis=0)),
            "rf_auc_w": np.abs(trapz(d["tpr_w"], d["frac"], axis=0)),
            "prg_auc": np.abs(trapz(d["precision_gain"], d["recall_gain"], axis=0)),
        }
        return area_metrics_dict

    def compute_metrics_with_unk(self):
        """
        Compute new metrics dicts after filling in unknown labels with 0s or 1s
        via a variety of methods:
        * "0" -- fill unknown labels with 0
        * "1" -- fill unknown labels with 1
        * "imb" -- fill unknown labels with 0 or 1 probabilistically according
            to the class imbalance of the known labels.
        * "prob" -- fill unknown labels with 0 or 1 probabilistically
            according to the probability-calibrated model score.
        """
        scores = self.scores
        labels = self.labels
        weights = self.weights
        imbalance = self.metrics_dict["imbalance"]
        null_probs = (
            self.null_probabilities if self.null_prob_column else self.scores
        )  # noqa

        # Initialize empty dicts
        self.metrics_dict_imputed = {}
        labels_filled = {}

        # Create masked labels array
        masked_labels = np.ma.array(labels, mask=np.isnan(labels))

        # Probabilistically generate labels according to probability values or
        # class imb.
        labels_from_prob = (np.random.rand(*labels.shape) < null_probs).astype(
            int
        )  # noqa
        labels_from_imb = (np.random.rand(*labels.shape) < imbalance).astype(
            int
        )  # noqa

        # Fill null labels according to fill method
        labels_filled["0"] = masked_labels.filled(0)
        labels_filled["1"] = masked_labels.filled(1)
        labels_filled["imb"] = masked_labels.filled(labels_from_imb)
        labels_filled["prob"] = masked_labels.filled(labels_from_prob)

        # Helper function to compute metrics for each null fill method
        def compute_filled_metrics(fill_method):
            print(f"null ==> {fill_method}")
            self.metrics_dict_imputed[fill_method] = self.compute_metrics(
                scores, labels_filled[fill_method], weights, return_dict=True
            )
            print("")

        # Compute metrics for each null fill method
        for method in self.null_fill_methods:
            compute_filled_metrics(method)

    def compute_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        return_dict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Compute the classification curve metrics values.

        Parameters
        ----------
        scores
            Numpy array of score values.
        labels
            Numpy array of label values.
        weights
            Numpy array of weight values.
        return_dict
            If True, will return the resulting metrics_dict, otherwise will set
            result to ``self.metrics_dict``.

        Returns
        -------
        metrics
            Dictionary of classification curve metrics-related values: {
                "tp": numpy array of true positive values,
                "fp": numpy array of false positive values,
                "fn": numpy array of false negative values,
                "tn": numpy array of true negative values,
                "tp_w": numpy array of weighted true positive values,
                "fp_w": numpy array of weighted false positive values,
                "fn_w": numpy array of weighted false negative values,
                "tn_w": numpy array of weighted true negative values,
                "up": [optional] numpy array of unlabeled predicted positive values, # noqa
                "un": [optional] numpy array of unlabeled predicted negative values,
                "up_w": [optional] numpy array of unlabeled weighted pred. pos. values,
                "un_w": [optional] numpy array of unlabeled weighted pred. neg. values,
                "precision": numpy array of precision values,
                "f1": numpy array of f1 score values,
                "tpr": numpy array of recall values,
                "fpr": numpy array of false-positive rate values,
                "tpr_w": numpy array of weighted recall values,
                "fpr_w": numpy array of weighted false-positive rate values,
                "precision_gain": numpy array of "precision gain" values,
                "recall_gain": numpy array of "recall gain" values,
                "frac": numpy array of flag-rate values,
                "frac_w": numpy array of weighted flag-rate values,
                "imbalance": class imbalance (positive class size / total data size),
                "roc_auc": area under the ROC curve,
                "pr_auc": area under the PR curve,
                "rf_auc": area under the RF curve,
                "roc_auc_w": area under the weighted ROC curve,
                "pr_auc_w": area under the weighted PR curve,
                "rf_auc_w": area under the weighted RF curve,
                "thresh": numpy array of score decision threshold values,
                "pos": number of positive examples,
                "neg": number of negative examples,
                "unk": number of unlabeled examples with unknown label,
                "pos_w": weighted sum of positive examples,
                "neg_w": weighted sum of negative examples,
                "unk_w": weighted sum of examples with unknown label,
                "num_bootstrap_samples": number of bootstrap samples
            }
        """

        # Compute all metrics
        confusion_dict = self._compute_confusion_dict(
            scores, labels, weights, self.thresholds
        )
        metrics_dict = self._compute_metrics_dict(confusion_dict)
        area_metrics_dict = self._compute_area_metrics_dict(metrics_dict)

        # Populate dictionary of classification curve metrics
        metrics_dict = {
            **confusion_dict,
            **metrics_dict,
            **area_metrics_dict,
            "thresh": self.thresholds,
            "num_bootstrap_samples": self.num_bootstrap_samples,
        }

        # Extract single number values from any 1-element arrays
        for k, v in metrics_dict.items():
            if type(v) == np.ndarray:
                if v.size == 1:
                    metrics_dict[k] = v[0]

        print("Complete.")

        # Either return the computed metrics dict or set it to
        # self.metrics_dict
        if return_dict:
            return metrics_dict
        else:
            self.metrics_dict = metrics_dict

        return None
