# testAUC
Official implementation of the tools discussed in *The curious case of the test set AUROC*

It is a set of tools designed for better evaluation of **Binary Classification** tasks in ML and AI.
Specifically dealing with the expected performance of the model on **new data**.
## Quick start
    pip install testAUC

For and all-in-one view of the toolset, use the dashboad() function:
```python
from testAUC import faux_normal_predictions, dashboard

# Simulate a model, evaluated on a Validation set and a Test set:
y_true_val, y_score_val = faux_normal_predictions(neg_mu=0.3, pos_mu=0.8, seed=2023)
y_true_tst, y_score_tst = faux_normal_predictions(std=0.5, neg_mu=0.4, pos_mu=0.9, seed=2023)

# All in one Dashboard to evaluate the Validation vs. Test sets performance
dashboard(y_true_val, y_score_val, y_true_tst, y_score_tst)
```
![Demo](https://github.com/alonhzn/testAUC/blob/main/images/demo1.png?raw=true)

roc_drift, val_tst_colored_roc_curve, colored_roc_curve,dashboard
noise_robustness, bias_robustness, plot_noise_robustness, plot_bias_robustness
plot_wasserstein_distance_matrix, plot_predictions_hist

### Mini-documentation of functions:
* dashboard() -> An All-In-One dashboard to evaluate the test performance (good place to start!)
* roc_drift() -> Calculate the ROC drift from validation to test sets  
* noise_robustness() -> Calculate the robustness of the predictions to normal noise
* plot_noise_robustness() -> uses the noise_robustness to generate a plot
* bias_robustness() -> Calculate the robustness of the predictions to bias between the classes
* plot_bias_robustness() -> uses the bias_robustness to generate a plot
* plot_wasserstein_distance_matrix() -> See paper to understand the importance of the matrix
* colored_roc_curve() ->  Plot an ROC curve that is color coded by threshold
* val_tst_colored_roc_curve() -> Same colored ROC curve but for both val&tst sets (sharing color limits!)
* faux_normal_predictions() -> A small utility function to create fake model predictions
* plot_predictions_hist() -> Plot a histogram of predictions for the Positive and Negative classes



