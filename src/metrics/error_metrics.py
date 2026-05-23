"""Базовые метрики качества прогноза"""
import numpy as np
from typing import Sequence

def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Считает среднюю абсолютную процентную ошибку прогноза"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Считает симметричную среднюю абсолютную процентную ошибку"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    if not mask.any():
        return np.nan
    return 200 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Считает квадратный корень из средней квадратичной ошибки"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Считает среднюю абсолютную ошибку прогноза"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
