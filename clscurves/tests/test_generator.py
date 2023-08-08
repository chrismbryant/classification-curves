import numpy as np
import pandas as pd

from .. import MetricsGenerator


def test_compute_confusion_matrix() -> None:
    scores = np.array([1, 2, 2, 2, 4, 5, 6, 8, 9, 10])
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0, 5, np.nan])
    weights = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    df = pd.DataFrame(
        {
            "scores": scores,
            "labels": labels,
            "weights": weights,
        }
    )

    mg = MetricsGenerator(
        df,
        label_column="labels",
        score_column="scores",
        weight_column="weights",
        num_bootstrap_samples=0,
        seed=123,
    )

    # TODO: Make a real test
    assert mg is not None
