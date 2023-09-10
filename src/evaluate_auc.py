from sklearn.metrics import roc_curve, auc
import numpy as np

def calc_tpr_fpr(pred, ground_truth, th):
    tp = np.sum(pred[ground_truth == 1] >= th)
    tn = np.sum(pred[ground_truth == 0] < th)
    fp = np.sum(pred[ground_truth == 0] >= th)
    fn = np.sum(pred[ground_truth == 1] < th)
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


def roc_drift_score(y_true_val, y_score_val, y_true_tst, y_score_tst, pos_label=None):
    """
    :param y_true_val: ground truth on validation set
    :param y_score_val: predictions on validation set
    :param y_true_tst: ground truth on TEST set
    :param y_score_tst: predictions on TEST set
    :param pos_label: positive label (default is 1)
    :return: drift score - Lower is better.
    """
    fpr_val, tpr_val, thresholds_val = roc_curve(y_true_val, y_score_val, pos_label=pos_label)

    scores=[]
    for th, vfpr, vtpr in zip(thresholds_val,fpr_val, tpr_val):
        tfpr, ttpr = calc_tpr_fpr(y_score_tst, y_true_tst, th)
        dist = np.linalg.norm(np.array([vfpr, vtpr])-np.array([tfpr, ttpr]))
        scores.append(dist)
    scores = np.array(scores)

    return scores, thresholds_val ,np.mean(scores)
