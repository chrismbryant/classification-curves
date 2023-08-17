from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class MetricsResult:
    """A class to hold the results of a metrics computation.

    Parameters
    ----------
    curves : pd.DataFrame
        DataFrame containing metrics at each threshold.
    scalars : pd.DataFrame
        DataFrame containing scalar metrics which apply to the entire dataset.
    curves_imputed : Optional[pd.DataFrame]
        DataFrame containing metrics at each threshold after imputation.
    scalars_imputed : Optional[pd.DataFrame]
        DataFrame containing scalar metrics which apply to the entire dataset
        after imputation.
    """

    curves: pd.DataFrame
    scalars: pd.DataFrame
    curves_imputed: Optional[pd.DataFrame] = None
    scalars_imputed: Optional[pd.DataFrame] = None
