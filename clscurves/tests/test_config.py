import pytest

from clscurves.config import RPFDictKeys


@pytest.mark.parametrize("key", [
    "tp",
    "fp",
    "fn",
    "tn",
    "tp_w",
    "fp_w",
    "fn_w",
    "tn_w",
    "precision",
    "tpr",
    "fpr",
    "tpr_w",
    "fpr_w",
    "frac",
    "frac_w",
    "thresh"
])
def test_rpf_dict_keys(key: str):
    cbar_dict = RPFDictKeys.cbar_dict
    assert key in cbar_dict