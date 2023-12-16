from abc import abstractmethod, ABC
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import roc_curve, auc, confusion_matrix

def faux_normal_predictions(N = 1000, std = 0.3, neg_mu=0.2, pos_mu=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    truth = np.concatenate([np.ones(N), np.zeros(N)])
    negatives = np.random.normal(neg_mu, std, N)
    positives = np.random.normal(pos_mu, std, N)
    prediction = np.concatenate((positives, negatives))
    return truth, prediction

def infer_pos_label(pos_label, y_true):
    if pos_label is None:
        pos_label = max(y_true)
    return pos_label

class Perturbation(ABC):
    def __init__(self, pos_label, σs):
        self.σs = σs
        self.pos_label = pos_label

    def get_auc(self, y, ŷ):
        fpr, tpr, _ = roc_curve(y, ŷ, pos_label=self.pos_label)
        return auc(fpr, tpr)


    def get_auc_gain(self, original_auc, y, ŷ):
        auc = self.get_auc(y, ŷ)
        return auc / original_auc

    def _check_inputs(self, ŷ):
        if ŷ.min() < 0 or 1 < ŷ.max():
            raise Exception('Supports scores between 0 and 1 and the moment')

    @abstractmethod
    def _get_y_scoreδ(self, y_true, y_score, σ):
        raise NotImplementedError

    def _get_gain(self, original_auc, y, ŷ, σ):
        ŷ = self._get_y_scoreδ(y, ŷ, σ)
        return self.get_auc_gain(original_auc, y, ŷ)

    def _get_pertrubed_socre_auc(self, y, ŷ, σ):
        ŷ = self._get_y_scoreδ(y, ŷ, σ)
        return self.get_auc(y, ŷ)

    def __call__(self, y, ŷ):

        original_auc = self.get_auc(y, ŷ)

        auc_gains = [self._get_gain(original_auc, y, ŷ, σ) for σ in self.σs]
        # auc_gains = [self._get_pertrubed_socre_auc(y, ŷ, σ) for σ in self.σs]
        auc_gains = np.array(auc_gains)

        perturbation_auc = np.trapz(auc_gains) / len(auc_gains)
        return auc_gains, perturbation_auc

class NoiseRobustness(Perturbation):
    def _get_y_scoreδ(self, y_true, y_score, σ):
        return np.random.normal(loc=y_score, scale=σ)

class BiasRobustness(Perturbation):
    def _get_y_scoreδ(self, y, ŷ, σ):
        is_pos = y == self.pos_label
        ŷ = np.copy(ŷ)
        ŷ[is_pos] -= σ
        return ŷ

def calc_tpr_fpr(pred, ground_truth, th):
    binary_pred = pred >= th
    tn, fp, fn, tp = confusion_matrix(ground_truth, binary_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return fpr, tpr

def _get_segments(fpr, tpr):
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def _get_lc(segments, thresholds, colormap, linewidth, color_lim):
    lc = LineCollection(segments, cmap=colormap, norm=plt.Normalize(*color_lim))
    lc.set_array(thresholds)
    lc.set_linewidth(linewidth)
    return lc
