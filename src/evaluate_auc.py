from sklearn.metrics import roc_curve, auc
import numpy as np

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

