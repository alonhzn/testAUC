import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

np.random.seed(2021)

N = 1000
std = 0.3

truth = np.concatenate([np.zeros(N), np.ones(N)])
labels = ['Negative class', 'Positive class']

# Simulate some possible model that outputs a score from 0 to 1
negatives = np.random.normal(0.2, std, N)
positives = np.random.normal(0.8, std, N)
val_prediction = np.concatenate((negatives, positives))
val_prediction -= np.min(val_prediction)
val_prediction /= np.max(val_prediction)

test_prediction = np.log(val_prediction + 1)  # Assume some different prediction on the test set
# test_prediction = val_prediction / 1.1  # Assume some different prediction on the test set
# test_prediction = val_prediction - 0.03  # Assume some different prediction on the test set

val_dist_ax = plt.subplot(3, 2, 1)
test_dist_ax = plt.subplot(3, 2, 2)
val_roc_ax = plt.subplot(3, 2, 3)
test_roc_ax = plt.subplot(3, 2, 4)


val_dist_ax.boxplot([val_prediction[truth == 0], val_prediction[truth == 1]], labels=labels, showfliers=False)
test_dist_ax.boxplot([test_prediction[truth == 0], test_prediction[truth == 1]], labels=labels, showfliers=False)
test_dist_ax.set_ylim([0, 1])
val_dist_ax.set_ylim([0, 1])
val_dist_ax.set_title('AI prediction on Validation set')
test_dist_ax.set_title('AI prediction on Test set')
val_dist_ax.set_ylabel('AI score')
test_dist_ax.set_ylabel('AI score')
val_dist_ax.grid('on')
test_dist_ax.grid('on')

val_set_fpr, val_set_tpr, validation_thresholds = roc_curve(truth, val_prediction)
rocauc = auc(val_set_fpr, val_set_tpr)
val_roc_ax.plot(val_set_fpr, val_set_tpr)
d_from_90 = np.abs(val_set_tpr-0.9)
sens_90_ind = np.where(d_from_90 == np.amin(d_from_90))[0][0]
validation_sens_90_th = validation_thresholds[sens_90_ind]
fpr_at_90sens=100*val_set_fpr[sens_90_ind]
val_roc_ax.set_title(f'Validation ROC AUC = {round(rocauc, 3)}')
val_roc_ax.annotate(f'90% sensitivity\n'
                    f'{round(100-fpr_at_90sens, 1)}% Specificity\n'
                    f'Threshold={round(validation_sens_90_th,2)}',
                    xytext=(0.5, 0.5),
                    xy=(val_set_fpr[sens_90_ind],0.9),
                    arrowprops=dict(facecolor='black', shrink=0.01, width=1))
val_roc_ax.set_ylabel('Sensitivity')
val_roc_ax.set_xlabel('1-Specificity')

test_set_fpr, test_set_tpr, test_thresholds = roc_curve(truth, test_prediction)
rocauc = auc(test_set_fpr, test_set_tpr)
test_roc_ax.plot(test_set_fpr, test_set_tpr)
test_roc_ax.set_title(f'"False" Test ROC AUC = {round(rocauc, 3)}')
d_from_90 = np.abs(test_set_tpr-0.9)
sens_90_ind = np.where(d_from_90 == np.amin(d_from_90))[0][0]
test_sens_90_th = test_thresholds[sens_90_ind]
fpr_at_90sens=100*test_set_fpr[sens_90_ind]
test_roc_ax.annotate(f'90% sensitivity\n'
                    f'{round(100-fpr_at_90sens, 1)}% Specificity\n'
                    f'Threshold={round(test_sens_90_th,2)}',
                    xytext=(0.5, 0.5),
                    xy=(test_set_fpr[sens_90_ind],0.9),
                    arrowprops=dict(facecolor='black', shrink=0.01, width=1))
test_roc_ax.set_ylabel('Sensitivity')
test_roc_ax.set_xlabel('1-Specificity')

def calc_tpr_fpr(pred, ground_truth, th):
    tp = np.sum(pred[ground_truth == 1] >= th)
    tn = np.sum(pred[ground_truth == 0] < th)
    fp = np.sum(pred[ground_truth == 0] >= th)
    fn = np.sum(pred[ground_truth == 1] < th)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return fpr, tpr


def roc_drift(ax, ground_truth, test_set_prediction, val_set_thresholds, val_set_fpr, val_set_tpr):
    i=0
    for th, vfpr, vtpr in zip(val_set_thresholds,val_set_fpr, val_set_tpr):
        if i % 40 == 0 or i>len(val_set_thresholds)-3:  # just for visualization, prune points to make the plot legible
            fpr, tpr = calc_tpr_fpr(test_set_prediction, ground_truth, th)
            lin = ax.plot([vfpr, fpr],[vtpr,tpr])
            ax.plot(vfpr, vtpr, marker='.', color=lin[0].get_color())
            ax.plot(fpr, tpr, marker='^', color=lin[0].get_color())
        i += 1

test_true_roc_ax=plt.subplot(3, 1, 3)
roc_drift(test_true_roc_ax, truth, test_prediction, validation_thresholds, val_set_fpr, val_set_tpr)
test_true_roc_ax.grid('on')
test_true_roc_ax.set_title('True ROC drift from Validation to Test')
fpr, tpr = calc_tpr_fpr(test_prediction, truth, validation_sens_90_th)
msg = f'''90% Validation sensitivity
{round(100-fpr_at_90sens, 1)}% Validation Specificity
Drift to:
{round(tpr*100, 1)}% Test sensitivity
{round(100-100*fpr, 1)}% Test Specificity'''
test_true_roc_ax.annotate(msg,
                    xytext=(0.5, 0.5),
                    ha='center', va='center',
                    xy=(val_set_fpr[sens_90_ind],0.9),
                    arrowprops=dict(facecolor='black', shrink=0.01, width=1))
test_true_roc_ax.annotate(msg,
                    xytext=(0.5, 0.5),
                    ha='center', va='center',
                    xy=(fpr,tpr),
                    arrowprops=dict(facecolor='black', shrink=0.01, width=1))
test_true_roc_ax.plot([val_set_fpr[sens_90_ind], fpr],[0.9,tpr],color='red')
test_true_roc_ax.plot(val_set_fpr[sens_90_ind], 0.9, marker='.', color='red')
test_true_roc_ax.plot(fpr, tpr, marker='^', color='red')
test_true_roc_ax.set_ylabel('Sensitivity')
test_true_roc_ax.set_xlabel('1-Specificity')

plt.show()