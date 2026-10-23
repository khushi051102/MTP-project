"""Evaluation metrics for RUL regression."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score (lower is better).
    d = predicted - true. Early predictions (d<0) are penalised gently
    (scale 13), late ones (d>0) harshly (scale 10): predicting failure AFTER
    it happens is the dangerous mistake in maintenance."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))
