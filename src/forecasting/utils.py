
"""Вспомогательные функции аппроксимации и экстраполяции рядов"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Any, Tuple

def fit_approximation(x: np.ndarray, y: np.ndarray, approx_method: str = 'auto') -> Tuple[LinearRegression, str]:

    """Подбирает линейную или степенную аппроксимацию для заданного фрагмента ряда"""
    if approx_method == 'linear' or (y <= 0).any():
        model = LinearRegression().fit(x, y)
        used = 'linear' if approx_method != 'power' else 'linear_nonpositive'
        return model, used
    else:
        log_x = np.log(x)
        log_y = np.log(y)
        model = LinearRegression().fit(log_x, log_y)
        used = 'power'
        return model, used

def predict_one(model: LinearRegression, used: str, last_index: int) -> float:
    """Вычисляет одно следующее значение по обученной аппроксимирующей функции"""
    if str(used).startswith('linear'):
        return model.predict([[last_index + 1]])[0]
    else:
        log_pred = model.intercept_ + model.coef_[0] * np.log(last_index + 1)
        return np.exp(log_pred)

def direct_forecast(series: pd.Series, window_approx: int, horizon: int, approx_method: str = 'auto') -> Tuple[list[float], str]:
    """
    Строит одну модель на последнем окне и прогнозирует сразу horizon шагов
    Если данных меньше window_approx, использует все доступные
    """
    actual_window = min(window_approx, len(series))
    values = series.values[-actual_window:].copy()
    if len(values) == 0:
        raise ValueError("Empty series in direct_forecast")
    x = np.arange(1, len(values)+1).reshape(-1, 1)
    model, used = fit_approximation(x, values, approx_method)
    if str(used).startswith('linear'):
        x_pred = np.arange(len(values)+1, len(values)+1+horizon).reshape(-1, 1)
        forecast = model.predict(x_pred)
    else:
        x_pred = np.arange(len(values)+1, len(values)+1+horizon)
        log_pred = model.intercept_ + model.coef_[0] * np.log(x_pred)
        forecast = np.exp(log_pred)
    return forecast.tolist(), used

def recurrent_forecast(series: pd.Series, window_approx: int, horizon: int, approx_method: str = 'auto') -> Tuple[list[float], list[str]]:
    """
    Рекуррентный прогноз: на каждом шаге модель строится заново на последнем окне,
    прогнозируется одно значение, окно сдвигается
    Если данных меньше window_approx, использует все доступные
    """
    actual_window = min(window_approx, len(series))
    values = series.values[-actual_window:].copy()
    if len(values) == 0:
        raise ValueError("Empty series in recurrent_forecast")
    forecast = []
    used_list = []
    current_window = values.copy()
    for step in range(horizon):
        x = np.arange(1, len(current_window)+1).reshape(-1, 1)
        model, used = fit_approximation(x, current_window, approx_method)
        next_pred = predict_one(model, used, len(current_window))
        forecast.append(next_pred)
        used_list.append(used)
        current_window = np.append(current_window[1:], next_pred)
    return forecast, used_list
