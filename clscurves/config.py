class RPFDictKeys:
    """RPF key aliases.

    A class to provide human-readable labels for each key in an ``rpf_dict``
    that we might want to color an RPF plot by.
    """
    cbar_dict = {
        "tp": "True Positive (TP) Count",
        "fp": "False Positive (FP) Count",
        "fn": "False Negative (FN) Count",
        "tn": "True Negative (TN) Count",
        "tp_w": "Weighted True Positive (TP) Sum",
        "fp_w": "Weighted False Positive (FP) Sum",
        "fn_w": "Weighted False Negative (FN) Sum",
        "tn_w": "Weighted True Negative (TN) Sum",
        "precision": "Precision = TP/(TP + FP)",
        "tpr": "Recall = TP/(TP + FN)",
        "fpr": "FPR = FP/(FP + TN)",
        "tpr_w": "Weighted Recall = TP/(TP + FN)",
        "fpr_w": "Weighted FPR = FP/(FP + TN)",
        "frac": "Fraction Flagged",
        "frac_w": "Weighted Fraction Flagged",
        "thresh": "Score Threshold Value"
    }
