from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

def calc_tpr_fpr(pred, ground_truth, th):
    binary_pred = pred >= th
    tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return fpr, tpr


def robustness_to_uniform_noise(y_true, y_score, pos_label=None):
    if y_score.max()>1 or y_score.min()<0:
        raise Exception('robustness_to_uniform_noise supports scores between 0 and 1 and the moment')

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    original_auc = auc(fpr, tpr)

    scales = np.logspace(start=-3, stop=0, num=20)

    auc_gains = []
    for scale in scales:
        noise = np.array([1 + np.random.uniform(low=-scale*2, high=scale*2) for _ in y_score])
        fpr, tpr, thresholds = roc_curve(y_true, y_score*noise, pos_label=pos_label)
        auc_gains.append(auc(fpr, tpr)/original_auc)

    auc_gains = np.array(auc_gains)
    robustness = np.trapz(auc_gains)/len(auc_gains)
    return scales, auc_gains, robustness


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
