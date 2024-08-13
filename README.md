# classification-curves

A library for computing and plotting bootstrapped metrics (ROC curves,
Precision-Recall curves, etc.) to evaluate the performance of a classification
model.

## Example
```python
mg = MetricsGenerator(
    predictions_df,
    label_column="label",
    score_column="score",
    weight_column="weight",
    score_is_probability=False,
    reverse_thresh=False,
    num_bootstrap_samples=20,
)

mg.plot_pr(
    op_value=0.1,
    bootstrapped=True,
    bootstrap_alpha=0.05,
)
mg.plot_roc()
```

![Example PR curve](docs/img/pr_curve_bootstrapped.png)
![Example ROC curve](docs/img/roc_curve.png)
