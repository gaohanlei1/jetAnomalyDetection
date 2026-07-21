"""
Visualization utilities for analyzing training and anomaly detection performance.

This module includes:
- Anomaly score distribution histograms
- ROC curve evaluation based on reconstruction loss
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd


def plot_anomaly_score(test_scores, anomaly_scores, background_label, signal_label, save_path=f"plots/test-plots/anomaly_score_hybrid3.png"):
    """
    Plot a histogram comparing the anomaly scores (MSE loss) for signal and background samples.

    Args:
        test_scores (List[float]): MSE losses for background (QCD) data.
        anomaly_scores (List[float]): MSE losses for signal events (e.g. WJets).
        background_label (str): Label to annotate the background histogram.
        signal_label (str): Label to annotate the signal histogram.
        save_path (str): File path to save the plot.

    Returns:
        None. Saves plot to disk.
    """
    plt.figure()
    bins = 100
    range_ = (
        min(np.min(anomaly_scores), np.min(test_scores)),
        max(np.max(anomaly_scores), np.max(test_scores))
    )

    plt.hist(anomaly_scores, bins=bins, range=range_, color='red', alpha=0.5,
             label=f'Anomalous - Signal: {signal_label}', density=True)
    plt.hist(test_scores, bins=bins, range=range_, color='blue', alpha=0.5,
             label=f'Non-anomalous - Background: {background_label}', density=True)

    plt.xlabel('Loss (MSE)')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(background_loss, signal_loss, signal_label, background_label, savepath, examples, loss_fn, properties=[]):
    """
    Compute and plot the ROC curve using signal and background reconstruction losses.

    Args:
        background_loss (list): List of reconstruction losses for background events.
        signal_loss (list): List of reconstruction losses for signal events.
        signal_label (str): Label for signal class.
        background_label (str): Label for background class.
        savepath (str): File path to save the plot.
        examples (bool): Unused.
        loss_fn (callable): Loss function used.
        properties (list): Reserved.

    Returns:
        None
    """
    test_loss = np.array(background_loss)
    signal_loss = np.array(signal_loss)

    # Labels: 0 = background, 1 = signal
    y_true = np.concatenate([np.zeros_like(test_loss), np.ones_like(signal_loss)])
    y_scores = np.concatenate([test_loss, signal_loss])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    # Plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel(f'False Positive Rate')
    plt.ylabel(f'True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - BG: {background_label}, Signal: {signal_label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
