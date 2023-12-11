# testAUC

## Quick start
    pip install testAUC

To evaluate the drift between the Validation set to the Test set use:
```python
from testAUC import roc_drift, faux_normal_predictions, dashboard

# Simulate a model, evaluated on a Validation set and a Test set:
y_true_val, y_score_val = faux_normal_predictions(neg_mu=0.3, pos_mu=0.8)
y_true_tst, y_score_tst = faux_normal_predictions(neg_mu=0.4, pos_mu=0.9)

# Calculate Drift between Validation set performance to Test set performance
tpr_drift, fpr_drift, drift, thresholds_val, mean_drift = roc_drift(y_true_val, y_score_val, y_true_tst, y_score_tst)

# All in one Dashboard to evaluate the Validation vs. Test sets performance
dashboard(y_true_val, y_score_val, y_true_tst, y_score_tst)
```
![Demo](https://github.com/alonhzn/testAUC/blob/main/images/demo1.png?raw=true)

 ## Motivation
 The ROC (Receiver operating characteristic) curve is generally used for evaluation of ML/AI model 
 performance in classification tasks. There are a number of pitfalls of using the ROC curve, 
 some of which are outlined in example below


Example 0:
Demonstrates how two very different predictions can result in the exact same ROC curve.

![Example0](https://github.com/alonhzn/testAUC/blob/main/images/example0.png?raw=true)

Example 1:
Demonstrates the shifting of the operation point, despite the favourable AUC. 
We show predictions of a (simulated) model on the Validation set and the Test set. 
The ROC AUC "happens" to be identical. 
However, if you choose a threshold on the Validation set, aiming for a Sensitivity/Specificity operation point, you would get completely different operation point on the Test set

![Example1](https://github.com/alonhzn/testAUC/blob/main/images/example1.png?raw=true)


Example 2:
Demonstrates a measure of robustness to noise.
We show two models, Initially it seems that Model 1 is far better, offering ROC AUC above 91%
However, evaluating the robustness to uniform noise shows Model 2 is more robust

![Example2](https://github.com/alonhzn/testAUC/blob/main/images/example2.png?raw=true)

Example 3:
Demonstrates a measure of DRIFT between the validation set and the test set
We show two models, both of them "happen" to have the exact same AUC on the validation and test sets.
However, evaluating the DRIFT of the Sensitivity/Specificity operation point reveals that Model 1 is better (although in this extreme case, both models exhibit significant drifts) 

Note that at some operation point, Model 2 actually exceeds a drift of -50% from the Sensitivity expected at Validation time, compared to the Sensitivity received at test time - without affecting the total AUC. 

![Example3](https://github.com/alonhzn/testAUC/blob/main/images/example3.png?raw=true)
