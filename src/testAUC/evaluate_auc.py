from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from testAUC.utils import calc_tpr_fpr, infer_pos_label, NoiseRobustness, BiasRobustness, _get_segments, _get_lc


def roc_drift(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=None):
    """Compute Sensitivity & Specificity Drift from Validation set to Test set.

    Parameters
    ----------
    y_true_val : array-like of shape (n_samples,)
        True binary labels on Validation set. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score_val : array-like of shape (n_samples,)
        Target scores on Validation set, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    y_true_tst : array-like of shape (n_samples,)
        True binary labels on Test set. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score_tst : array-like of shape (n_samples,)
        Target scores on Test set, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    Returns
    -------
    tpr_drift : ndarray of shape (>2,)
        the true-positive-rate (Sensitivity) drift between the validation set
        and the test set, per given threshold.

    fpr_drift : ndarray of shape (>2,)
        the false-positive-rate (1-Specificity) drift between the validation set
        and the test set, per given threshold.

    drift : ndarray of shape (>2,)
        the L2 distance in the sensitivity-specificity drift space between the validation set
        and the test set, per given threshold.

    thresholds_val : ndarray of shape (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr on the Validation set. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `np.inf`.

    mean_drift : numpy float
        The mean of the sensitivity-specificity drift

    """
    fpr_val, tpr_val, thresholds_val = roc_curve(y_true_val, y_score_val, pos_label=pos_label)
    tpr_drift = []
    fpr_drift = []
    for th, vfpr, vtpr in zip(thresholds_val,fpr_val, tpr_val):
        tfpr, ttpr = calc_tpr_fpr(y_score_tst, y_true_tst, th)
        tpr_drift.append(ttpr-vtpr)
        fpr_drift.append(tfpr-vfpr)
    drift = np.linalg.norm(np.array([tpr_drift,fpr_drift]), axis=0)
    tpr_drift = np.array(tpr_drift)
    fpr_drift = np.array(fpr_drift)
    mean_drift = np.mean(drift)
    return tpr_drift, fpr_drift, drift, thresholds_val, mean_drift

def val_tst_colored_roc_curve(y_true_val, y_score_val,y_true_tst, y_score_tst, pos_label=None, axis=None, colormap='tab20', linewidth=8):

    if axis is None:
        fig, axis = plt.subplots(1, 1)
        pltshow = True
    else:
        pltshow = False

    color_lim = (min(y_score_val.min(), y_score_tst.min()), max(y_score_val.max(), y_score_tst.max()))

    colored_roc_curve(y_true_val, y_score_val, pos_label=pos_label, label='Val Set', linestyle="-", axis=axis,
                      color_lim=color_lim, colormap=colormap, linewidth=linewidth, colorbar=False)
    colored_roc_curve(y_true_tst, y_score_tst, pos_label=pos_label, label='Test Set', linestyle="--", axis=axis,
                      color_lim=color_lim, colormap=colormap, linewidth=linewidth, colorbar=True)

    if pltshow:
        plt.show()

def colored_roc_curve(y_true, y_score, pos_label=None, label=None, linestyle="-", axis=None, color_lim=None, colormap='jet', linewidth=8, colorbar=False):

    if axis is None:
        fig, axis = plt.subplots(1, 1)
        pltshow = True
    else:
        pltshow = False

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    thresholds = thresholds[thresholds != np.inf]
    if color_lim is None:
        color_lim = (thresholds.min(), thresholds.max())

    segments = _get_segments(fpr, tpr)

    lc = _get_lc(segments, thresholds, colormap, linewidth, color_lim)
    axis.add_collection(lc)
    if colorbar:
        plt.colorbar(lc, ax=axis)
    axis.plot(fpr, tpr, color='k', linewidth=linewidth/8, linestyle=linestyle, label=f"{label} - AUC={auc(fpr, tpr):.4f}" )

    if label is not None:
        axis.legend(loc='lower right')
    axis.set_xlim(-0.01, 1)  # a bit wider to make space for the line colored shading
    axis.set_ylim(0, 1.01)
    axis.set_title(f"Threshold colored ROC")
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.grid(color='lightgrey')
    if pltshow:
        plt.show()

def dashboard(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=None):
    tpr_drift, fpr_drift, drift, thresholds_val, mean_drift = roc_drift(y_true_val, y_score_val,
                                                                        y_true_tst, y_score_tst)
    fig = plt.figure(figsize=(16, 9))

    val_dist_ax = fig.add_subplot(2, 4, 2)
    tst_dist_ax = fig.add_subplot(2, 4, 3)
    wasserstein_ax = fig.add_subplot(2, 4, 1)

    roc_val_tst = fig.add_subplot(2, 4, 4)
    drift_ax = fig.add_subplot(2, 4, 7)
    sensitivity_specificity_drift_ax = fig.add_subplot(2, 4, 8)
    noise_robustness_ax = fig.add_subplot(2, 4, 6)
    bias_robustness_ax = fig.add_subplot(2, 4, 5)

    plot_wasserstein_distance_matrix(y_true_val, y_score_val, y_true_tst, y_score_tst, axis=wasserstein_ax, pos_label=pos_label)
    val_tst_colored_roc_curve(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=pos_label, axis=roc_val_tst)
    σmax = max(y_score_val.max(), y_score_tst.max())
    plot_noise_robustness(y_true_val, y_score_val, axis=noise_robustness_ax, ttl='Val', pos_label=pos_label, σmax=σmax)
    plot_noise_robustness(y_true_tst, y_score_tst, axis=noise_robustness_ax, ttl='Tst', pos_label=pos_label, σmax=σmax)
    plot_bias_robustness(y_true_val, y_score_val, axis=bias_robustness_ax, ttl='Val', pos_label=pos_label, σmax=σmax)
    plot_bias_robustness(y_true_tst, y_score_tst, axis=bias_robustness_ax, ttl='Tst', pos_label=pos_label, σmax=σmax)

    xlim = (min(y_score_val.min(), y_score_tst.min()), max(y_score_val.max(), y_score_tst.max()))  # for visual perspective
    plot_predictions_hist(y_true_val, y_score_val, bins=30, ax=val_dist_ax, set_name='Validation', xlim=xlim)
    plot_predictions_hist(y_true_tst, y_score_tst, bins=30, ax=tst_dist_ax, set_name='Test', xlim=xlim)



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

    fig.subplots_adjust(hspace=0.3,wspace=0.25, left=0.06, bottom=0.06, right=0.96, top=0.96)
    plt.show()

def noise_robustness(y, ŷ, pos_label=None ,n_points=20, σmax=None):
    pos_label = infer_pos_label(pos_label, y)
    if σmax is None:
        σmax = max(ŷ)
    σs = np.linspace(0, σmax, num=n_points, endpoint=True)
    auc_gains, perturbation_auc = NoiseRobustness(pos_label=pos_label, σs=σs)(y, ŷ)
    return σs, auc_gains, perturbation_auc

def bias_robustness(y, ŷ, pos_label=None ,n_points=20, σmax=None):
    pos_label = infer_pos_label(pos_label, y)
    if σmax is None:
        σmax = max(ŷ)
    σs = np.linspace(0, σmax, num=n_points, endpoint=True)
    auc_gains, perturbation_auc = BiasRobustness(pos_label=pos_label, σs=σs)(y, ŷ)
    return σs, auc_gains, perturbation_auc

def plot_noise_robustness(y, ŷ,  pos_label=None, axis=None, ttl='', σmax=None):
    if axis is None:
        pltshow=True
        _, axis = plt.subplots(1,1)
    else:
        pltshow=False
    σs, auc_gains, perturbation_auc = noise_robustness(y, ŷ, pos_label=pos_label, σmax=σmax)

    axis.plot(σs, auc_gains*100, label=f'{ttl} score {perturbation_auc:.3f}')
    axis.set_title(f'Noise Robustness')
    axis.set_xlabel('Perturbation scale')
    axis.set_ylabel('Relative ROCAUC [%]')
    axis.legend()
    if pltshow:
        plt.show()

def plot_bias_robustness(y, ŷ,  pos_label=None, axis=None, ttl='', σmax=None):
    if axis is None:
        pltshow=True
        _, axis = plt.subplots(1,1)
    else:
        pltshow=False
    σs, auc_gains, perturbation_auc = bias_robustness(y, ŷ, pos_label=pos_label, σmax=σmax)

    axis.plot(σs, auc_gains*100, label=f'{ttl} score {perturbation_auc:.3f}')
    axis.set_title(f'Bias Robustness')
    axis.set_xlabel('Perturbation scale')
    axis.set_ylabel('Relative ROCAUC [%]')
    axis.legend()
    if pltshow:
        plt.show()

def plot_wasserstein_distance_matrix(y_true_val, y_score_val, y_true_tst, y_score_tst, axis, pos_label=None):
    if axis is None:
        pltshow=True
        _, axis = plt.subplots(1,1)
    else:
        pltshow=False

    pos_label = infer_pos_label(pos_label, y_true_val)

    y_score_valid_0 = y_score_val[y_true_val != pos_label]
    y_score_valid_1 = y_score_val[y_true_val == pos_label]
    y_score_test_0 = y_score_tst[y_true_tst != pos_label]
    y_score_test_1 = y_score_tst[y_true_tst == pos_label]

    v0t0 = wasserstein_distance(y_score_valid_0, y_score_test_0)
    v1t1 = wasserstein_distance(y_score_valid_1, y_score_test_1)
    v0v1 = wasserstein_distance(y_score_valid_0, y_score_valid_1)
    t0t1 = wasserstein_distance(y_score_test_0, y_score_test_1)
    poscol = '#FFA556'
    negcol = '#629FCA'
    costs = [[f"{v0v1: .3f} \n(higher is better)", f"{v1t1:.3f} \n(lower is better)"], [f"{v0t0: .3f} \n(lower is better)", f"{t0t1: .3f} \n(higher is better)"]]
    tbl=axis.table(cellText=costs,
               rowLabels=['Val -', 'Tst +'],
               rowColours=[negcol, poscol],
               colLabels=['Val +', 'Tst -'],
               colColours=[poscol, negcol],
                   cellLoc='center',
               bbox=[0, 0, 1, 0.8],
                   )
    tbl.scale(1, 2)
    tbl.set_fontsize(12)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axis.text(0, 1, f'Wasserstein distance matrix\nWscore={(v0v1+t0t1)/(v0t0+v1t1):.3f} (higher is better)',
              transform=axis.transAxes, fontsize=14,
              verticalalignment='top', bbox=props)
    axis.axis('off')
    if pltshow:
        plt.show()

def plot_predictions_hist(y_true, y_score, pos_label=None, bins=30, ax=None, set_name='', xlim=None):
    if ax is None:
        pltshow=True
        _, ax = plt.subplots(1,1)
    else:
        pltshow=False
    pos_label = infer_pos_label(pos_label, y_true)

    pos_val_predictions = y_score[y_true == pos_label]
    neg_val_predictions = y_score[y_true != pos_label]
    ax.hist(pos_val_predictions, label='Negative Class', alpha=0.7, bins=bins)
    ax.hist(neg_val_predictions, label='Positive Class', alpha=0.7, bins=bins)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.set_title(f"{set_name} set predictions (ROCAUC = {auc(fpr, tpr):.3f})")
    ax.legend()
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Frequency')
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if pltshow:
        plt.show()
