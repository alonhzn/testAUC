import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from src.evaluate_auc import robustness_to_uniform_noise

np.random.seed(2023)
N = 10000
std = 0.3
bins = 30
fig, axs = plt.subplots(2, 2)
labels = ['Negative class', 'Positive class']


def simulate_model_results(N, neg_mean, pos_mean, std):
    negatives = np.random.normal(neg_mean, std, N)
    positives = np.random.normal(pos_mean, std, N)
    negatives = negatives[np.bitwise_and(negatives<1,negatives>0)]
    positives = positives[np.bitwise_and(positives<1,positives>0)]
    predictions = np.concatenate((negatives, positives))
    return np.concatenate([np.zeros(len(negatives)), np.ones(len(positives))]), predictions

truth1, test1_predictions = simulate_model_results(N, neg_mean=0.49, pos_mean=0.51, std=0.01)
truth2, test2_predictions = simulate_model_results(N, neg_mean=0.3, pos_mean=0.7, std=0.3)


axs[0, 0].hist(test1_predictions[:N], label='Negative Class', alpha=0.7, bins=bins)
axs[0, 0].hist(test1_predictions[N:], label='Positive Class', alpha=0.7, bins=bins)
fpr, tpr, thresholds1 = roc_curve(truth1, test1_predictions)
axs[0, 0].set_title(f"Model 1 ROC AUC = {auc(fpr, tpr):.5f}")
axs[0, 0].set_xlim([0, 1])
axs[0, 0].legend()
axs[0, 0].set_xlabel('Predictions')
axs[0, 0].set_ylabel('Frequency')

scales, auc_gains, robustness = robustness_to_uniform_noise(truth1, test1_predictions)
axs[1, 0].set_xscale('log')
axs[1, 0].set_ylim([0,1])
axs[1, 0].set_xlim([scales.min(), 1])
axs[1, 0].plot(scales, auc_gains)
axs[1, 0].set_title(f'Model 1 Robustness = {robustness:.3f}')
axs[1, 0].set_xlabel('Noise scale')
axs[1, 0].set_ylabel('Noised-predictions AUC to Original AUC ratio')


axs[0, 1].hist(test2_predictions[:N], label='Negative Class', alpha=0.7, bins=bins)
axs[0, 1].hist(test2_predictions[N:], label='Positive Class', alpha=0.7, bins=bins)
axs[0, 1].set_xlim([0, 1])
fpr, tpr, thresholds2 = roc_curve(truth2, test2_predictions)
axs[0, 1].set_title(f"Model 2 ROC AUC = {auc(fpr, tpr):.5f}")
axs[0, 1].legend()
axs[0, 1].set_xlabel('Predictions')
axs[0, 1].set_ylabel('Frequency')

scales, auc_gains, robustness = robustness_to_uniform_noise(truth2, test2_predictions)
axs[1, 1].set_xscale('log')
axs[1, 1].set_ylim([0, 1])
axs[1, 1].set_xlim([scales.min(), 1])
axs[1, 1].plot(scales, auc_gains)
axs[1, 1].set_title(f'Model 2 Robustness = {robustness:.3f}')
axs[1, 1].set_xlabel('Noise scale')
axs[1, 1].set_ylabel('Noised-predictions AUC to Original AUC ratio')

plt.show()



