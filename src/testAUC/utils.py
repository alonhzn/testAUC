import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from testAUC.evaluate_auc import roc_drift, val_tst_colored_roc_curve


def faux_normal_predictions(N = 1000, std = 0.3, neg_mu=0.2, pos_mu=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    truth = np.concatenate([np.ones(N), np.zeros(N)])
    negatives = np.random.normal(neg_mu, std, N)
    positives = np.random.normal(pos_mu, std, N)
    prediction = np.concatenate((positives, negatives))
    # prediction -= np.min(prediction)
    # prediction /= np.max(prediction)
    return truth, prediction

def plot_predictions_hist(y_true, y_score, pos_label=None, bins=30, ax=None, set_name=''):
    if ax is None:
        pltshow=True
        _, ax = plt.subplots(1,1)
    else:
        pltshow=False
    if pos_label is None:
        pos_label = max(y_true)

    pos_val_predictions = y_score[y_true == pos_label]
    neg_val_predictions = y_score[y_true != pos_label]
    ax.hist(pos_val_predictions, label='Negative Class', alpha=0.7, bins=bins)
    ax.hist(neg_val_predictions, label='Positive Class', alpha=0.7, bins=bins)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.set_title(f"{set_name} set predictions (ROCAUC = {auc(fpr, tpr):.3f})")
    ax.legend()
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Frequency')
    if pltshow:
        plt.show()

def dashboard(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=None):
    tpr_drift, fpr_drift, drift, thresholds_val, mean_drift = roc_drift(y_true_val, y_score_val,
                                                                        y_true_tst, y_score_tst)
    fig = plt.figure(figsize=(16, 9))

    val_dist_ax = fig.add_subplot(2, 3, 1)
    tst_dist_ax = fig.add_subplot(2, 3, 2)
    roc_val_tst = fig.add_subplot(2, 3, 3)
    drift_ax = fig.add_subplot(2, 2, 3)
    sensitivity_specificity_drift_ax = fig.add_subplot(2, 2, 4)

    val_tst_colored_roc_curve(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=pos_label, axis=roc_val_tst)

    plot_predictions_hist(y_true_val, y_score_val, bins=30, ax=val_dist_ax, set_name='Validation')
    plot_predictions_hist(y_true_tst, y_score_tst, bins=30, ax=tst_dist_ax, set_name='Test')

    drift_ax.plot(thresholds_val, drift, label=f'Drift Mean = {mean_drift:.3f}')
    drift_ax.legend()
    drift_ax.set_ylabel('Drift (Lower is better)')
    drift_ax.set_title("Total Drift")
    drift_ax.set_xlabel('Operation point threshold')
    drift_ax.grid(color='lightgrey')
    val_dist_ax.grid(color='lightgrey')
    tst_dist_ax.grid(color='lightgrey')

    sensitivity_specificity_drift_ax.plot(thresholds_val, tpr_drift, label=f'Sensitivity')
    sensitivity_specificity_drift_ax.plot(thresholds_val, -fpr_drift, label=f'Specificity')
    sensitivity_specificity_drift_ax.legend()
    sensitivity_specificity_drift_ax.set_ylabel('Drift')
    sensitivity_specificity_drift_ax.set_xlabel('Operation point threshold')
    sensitivity_specificity_drift_ax.set_title("Sensitivity/Specificity Drift")
    sensitivity_specificity_drift_ax.grid(color='lightgrey')



    fig.subplots_adjust(hspace=0.4, left=0.06, bottom=0.06, right=0.96, top=0.96)

    plt.show()