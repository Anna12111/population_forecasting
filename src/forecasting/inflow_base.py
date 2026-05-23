"""Общая реализация балансовой модели притока и оттока"""
from .base import ForecastingMethod
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import numpy as np

class InflowMethod(ForecastingMethod):
    """Родительский класс для методов, основанных на балансе притока и оттока населения"""
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Инициализирует объект и сохраняет параметры, которые используются при дальнейшем расчёте"""
        super().__init__(params)
        params = params or {}
        self.window_approx = params.get('window_approx', 10)
        self.approx_method = params.get('approx_method', 'auto')
        self.use_moving_window = params.get('use_moving_window', False)

    def fit(self, data: pd.DataFrame) -> None:
        """Подготавливает внутренние ряды метода на основе обучающего фрагмента данных"""
        self.fitted_series = data
        self.inflow = data['inflow']
        self.outflow = data['outflow']
        self.total = data['population']
        self.total_actual = data['total_actual']
        self.last_year = data.index[-1]

        self.base_total = self.total.iloc[0]

        self.inflow_int = self.inflow.cumsum()
        self.outflow_int = self.outflow.cumsum()

        self.r_inflow = self.inflow_int.pct_change().dropna()
        self.r_outflow = self.outflow_int.pct_change().dropna()

        return self

    def _forecast_from_r(self, r_inflow_pred: pd.Series, r_outflow_pred: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Восстанавливает интегралы, годовые значения и население из прогнозных r (доли)

        Формулы:
            I_hat_t = I_hat_{t-1} * (1 + r_hat_I_t)
            O_hat_t = O_hat_{t-1} * (1 + r_hat_O_t)
            P_hat_t = P_0 + I_hat_t - O_hat_t
        """
        last_inflow_int = self.inflow_int.iloc[-1]
        last_outflow_int = self.outflow_int.iloc[-1]

        inflow_int_pred = []
        outflow_int_pred = []
        cur_in = last_inflow_int
        cur_out = last_outflow_int

        for i in range(len(r_inflow_pred)):
            cur_in = cur_in * (1 + r_inflow_pred[i])
            cur_out = cur_out * (1 + r_outflow_pred[i])
            inflow_int_pred.append(cur_in)
            outflow_int_pred.append(cur_out)

        inflow_annual = [inflow_int_pred[0] - last_inflow_int] + [
            inflow_int_pred[i] - inflow_int_pred[i - 1]
            for i in range(1, len(inflow_int_pred))
        ]
        outflow_annual = [outflow_int_pred[0] - last_outflow_int] + [
            outflow_int_pred[i] - outflow_int_pred[i - 1]
            for i in range(1, len(outflow_int_pred))
        ]

        total_pred = [
            self.base_total + inflow_int_pred[i] - outflow_int_pred[i]
            for i in range(len(inflow_int_pred))
        ]

        return inflow_annual, outflow_annual, total_pred, inflow_int_pred, outflow_int_pred

    def _forecast_from_integrals(self, inflow_int_pred: pd.Series, outflow_int_pred: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Прямой прогноз интегралов. Возвращает годовые значения и население

        Формула населения:
            P_hat_t = P_0 + I_hat_t - O_hat_t
        """
        last_inflow_int = self.inflow_int.iloc[-1]
        last_outflow_int = self.outflow_int.iloc[-1]

        inflow_annual = [inflow_int_pred[0] - last_inflow_int] + [
            inflow_int_pred[i] - inflow_int_pred[i - 1]
            for i in range(1, len(inflow_int_pred))
        ]
        outflow_annual = [outflow_int_pred[0] - last_outflow_int] + [
            outflow_int_pred[i] - outflow_int_pred[i - 1]
            for i in range(1, len(outflow_int_pred))
        ]

        total_pred = [
            self.base_total + inflow_int_pred[i] - outflow_int_pred[i]
            for i in range(len(inflow_int_pred))
        ]

        return inflow_annual, outflow_annual, total_pred

    def _store_forecasts(self, inflow_annual: pd.Series, outflow_annual: pd.Series, total_pred: pd.Series,
                         r_inflow_pred: Optional[pd.Series] = None, r_outflow_pred: Optional[pd.Series] = None,
                         inflow_int_pred: Optional[pd.Series] = None, outflow_int_pred: Optional[pd.Series] = None,
                         alpha_inflow_pred: Optional[pd.Series] = None, alpha_outflow_pred: Optional[pd.Series] = None,
                         used_func_inflow: Optional[Any] = None, used_func_outflow: Optional[Any] = None) -> pd.Series:
        """
        Сохраняет все прогнозные ряды в атрибуты
        """
        start_year = self.last_year + 1
        index = pd.RangeIndex(start=start_year, stop=start_year + len(total_pred))

        self.inflow_forecast = pd.Series(inflow_annual, index=index)
        self.outflow_forecast = pd.Series(outflow_annual, index=index)

        self.r_inflow_forecast = pd.Series(r_inflow_pred, index=index) if r_inflow_pred is not None else None
        self.r_outflow_forecast = pd.Series(r_outflow_pred, index=index) if r_outflow_pred is not None else None

        self.inflow_int_forecast = pd.Series(inflow_int_pred, index=index) if inflow_int_pred is not None else None
        self.outflow_int_forecast = pd.Series(outflow_int_pred, index=index) if outflow_int_pred is not None else None

        self.alpha_inflow_forecast = pd.Series(alpha_inflow_pred, index=index) if alpha_inflow_pred is not None else None
        self.alpha_outflow_forecast = pd.Series(alpha_outflow_pred, index=index) if alpha_outflow_pred is not None else None

        self.used_func_inflow_series = pd.Series(used_func_inflow, index=index) if used_func_inflow is not None else None
        self.used_func_outflow_series = pd.Series(used_func_outflow, index=index) if used_func_outflow is not None else None

        return pd.Series(total_pred, index=index)
