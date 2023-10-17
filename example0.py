#####################################################################################################################
# In this example code demonstrate that ROC can be very misleading.
# We show this by taking a (simulated) ML model evaluations on two very different datasets that have the exact same ROC
#####################################################################################################################
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

np.random.seed(2021)
N = 1000
std = 0.3

truth = np.concatenate([np.zeros(N), np.ones(N)])
labels = ['Negative class', 'Positive class']

# Simulate some possible model that outputs a score from 0 to 1
negatives = np.random.normal(0.3, std, N)
positives = np.random.normal(0.7, std, N)
val_prediction = np.concatenate((negatives, positives))
val_prediction -= np.min(val_prediction)
val_prediction /= np.max(val_prediction)

test_prediction = np.log(val_prediction+0.1)*1.5  # Assume some different prediction on the test set
# test_prediction = val_prediction / 1.1  # Assume some different prediction on the test set
# test_prediction = val_prediction - 0.03  # Assume some different prediction on the test set

fig, [[ax_val_dist, ax_tst_dist],[ax_val_roc, ax_tst_roc]] = plt.subplots(2,2)


ax_val_dist.hist(val_prediction[:N], label='Negative Class', alpha=0.7, bins=30)
ax_val_dist.hist(val_prediction[N:], label='Positive Class', alpha=0.7, bins=30)
ax_val_dist.set_ylabel('Frequency')
ax_val_dist.set_xlabel('Score')
ax_val_dist.set_title('Model Predictions on dataset A', size=10)
ax_val_dist.legend()

val_set_fpr, val_set_tpr, validation_thresholds = roc_curve(truth, val_prediction)
rocauc = auc(val_set_fpr, val_set_tpr)
ax_val_roc.plot(val_set_fpr, val_set_tpr, label=f'AUC={rocauc:.6f}')
ax_val_roc.set_ylabel('Sensitivity')
ax_val_roc.set_xlabel('1-Specificity')
ax_val_roc.set_title('ROC on dataset A', size=10)
ax_val_roc.legend()

ax_tst_dist.hist(test_prediction[:N], label='Negative Class', alpha=0.7, bins=30)
ax_tst_dist.hist(test_prediction[N:], label='Positive Class', alpha=0.7, bins=30)
ax_tst_dist.set_ylabel('Frequency')
ax_tst_dist.set_xlabel('Score')
ax_tst_dist.set_title('Model Predictions on dataset B', size=10)
ax_tst_dist.legend()

tst_set_fpr, tst_set_tpr, test_thresholds = roc_curve(truth, test_prediction)
rocauc = auc(tst_set_fpr, tst_set_tpr)
ax_tst_roc.plot(tst_set_fpr, tst_set_tpr, label=f'AUC={rocauc:.6f}')
ax_tst_roc.set_ylabel('Sensitivity')
ax_tst_roc.set_xlabel('1-Specificity')
ax_tst_roc.set_title('ROC on dataset B', size=10)
ax_tst_roc.legend()

plt.show()
