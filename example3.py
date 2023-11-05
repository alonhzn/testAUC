import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from src.testAUC.evaluate_auc import roc_drift

np.random.seed(2021)
N = 1000
std = 0.3
bins = 30
truth = np.concatenate([np.zeros(N), np.ones(N)])
labels = ['Negative class', 'Positive class']

# Simulate some possible model that outputs a score from 0 to 1
negatives = np.random.normal(0.2, std, N)
positives = np.random.normal(0.8, std, N)
val_prediction = np.concatenate((negatives, positives))
val_prediction -= np.min(val_prediction)
val_prediction /= np.max(val_prediction)
test_prediction1 = np.log(val_prediction + 1)  # Simulate some model predictions on test set
tpr_drift1, fpr_drift1, scores1, th1, result1 = roc_drift(truth, val_prediction, truth, test_prediction1)

test_prediction2 = val_prediction/1.1 + 0.2  # Simulate some OTHER model predictions on test set
tpr_drift2, fpr_drift2, scores2, th2, result2 = roc_drift(truth, val_prediction, truth, test_prediction2)

val_dist_ax = plt.subplot(4, 3, 1)
test1_dist_ax = plt.subplot(4, 3, 2)
test2_dist_ax = plt.subplot(4, 3, 3)
drift_ax = plt.subplot(4, 1, 2)
sensitivity_drift_ax = plt.subplot(4, 1, 3)
specificity_drift_ax = plt.subplot(4, 1, 4)

val_dist_ax.hist(val_prediction[:N], label='Negative Class', alpha=0.7, bins=bins)
val_dist_ax.hist(val_prediction[N:], label='Positive Class', alpha=0.7, bins=bins)
fpr, tpr, _ = roc_curve(truth, val_prediction)
val_dist_ax.set_title(f"Val set predictions (ROC AUC = {auc(fpr, tpr):.3f}) \nBoth models")
val_dist_ax.set_xlim([-0.1, 1])
val_dist_ax.legend()
val_dist_ax.set_xlabel('Predictions')
val_dist_ax.set_ylabel('Frequency')

test1_dist_ax.hist(test_prediction1[:N], label='Negative Class', alpha=0.7, bins=bins)
test1_dist_ax.hist(test_prediction1[N:], label='Positive Class', alpha=0.7, bins=bins)
fpr, tpr, _ = roc_curve(truth, test_prediction1)
test1_dist_ax.set_title(f"test set predictions (ROC AUC = {auc(fpr, tpr):.3f})\nModel 1")
test1_dist_ax.set_xlim([-0.1, 1])
test1_dist_ax.legend()
test1_dist_ax.set_xlabel('Predictions')
test1_dist_ax.set_ylabel('Frequency')

test2_dist_ax.hist(test_prediction2[:N], label='Negative Class', alpha=0.7, bins=bins)
test2_dist_ax.hist(test_prediction2[N:], label='Positive Class', alpha=0.7, bins=bins)
fpr, tpr, _ = roc_curve(truth, test_prediction2)
test2_dist_ax.set_title(f"test set predictions (ROC AUC = {auc(fpr, tpr):.3f})\nModel 2")
test2_dist_ax.set_xlim([-0.1, 1])
test2_dist_ax.legend()
test2_dist_ax.set_xlabel('Predictions')
test2_dist_ax.set_ylabel('Frequency')

drift_ax.plot(th1, scores1, label=f'Model 1 (Total Drift score = {result1:.3f}')
drift_ax.plot(th2, scores2, label=f'Model 2 (Total Drift score = {result2:.3f}')
drift_ax.legend()
# drift_ax.set_xlabel('Operation point threshold')
drift_ax.set_ylabel('Drift score (Lower is better)')

sensitivity_drift_ax.plot(th1, tpr_drift1, label=f'Model 1')
sensitivity_drift_ax.plot(th2, tpr_drift2, label=f'Model 2')
sensitivity_drift_ax.legend()
# sensitivity_drift_ax.set_xlabel('Operation point threshold')
sensitivity_drift_ax.set_ylabel('Sensitivity Drift')

# spec_drift = spec_val - spec_tst = (1 - fpr_val) - (1 - fpr_tst) = fpr_tst - fpr_val
specificity_drift_ax.plot(th1, -fpr_drift1, label=f'Model 1')
specificity_drift_ax.plot(th2, -fpr_drift2, label=f'Model 2')
specificity_drift_ax.legend()
specificity_drift_ax.set_xlabel('Operation point threshold')
specificity_drift_ax.set_ylabel('Specificity Drift')


plt.show()