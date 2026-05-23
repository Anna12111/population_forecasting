"""Метод прогнозирования скоростей изменения процентных приростов"""
from .inflow_base import InflowMethod
from . import utils
import pandas as pd
from typing import Any, Dict, Optional
import numpy as np

class GrowthRateSpeed(InflowMethod):
    """Метод, который прогнозирует скорость изменения процентных приростов интегральных потоков"""
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Инициализирует объект и сохраняет параметры, которые используются при дальнейшем расчёте"""
        super().__init__(params)
        self.window_extrap = params.get('window_extrap', 5)

    def fit(self, data: pd.DataFrame) -> None:
        """Подготавливает внутренние ряды метода на основе обучающего фрагмента данных"""
        super().fit(data)
        w = self.window_extrap
        self.alpha_inflow = pd.Series(index=self.r_inflow.index, dtype=float)
        self.alpha_outflow = pd.Series(index=self.r_outflow.index, dtype=float)
        for i in range(w, len(self.r_inflow)):
            self.alpha_inflow.iloc[i] = (self.r_inflow.iloc[i-w+1] - self.r_inflow.iloc[i]) / w
        for i in range(w, len(self.r_outflow)):
            self.alpha_outflow.iloc[i] = (self.r_outflow.iloc[i-w+1] - self.r_outflow.iloc[i]) / w
        self.alpha_inflow = self.alpha_inflow.dropna()
        self.alpha_outflow = self.alpha_outflow.dropna()
        return self

    def predict(self, horizon: int) -> pd.Series:
        """Строит прогноз на заданное число шагов вперёд"""
        if self.use_moving_window:
            return self._predict_moving(horizon)
        else:
            return self._predict_direct(horizon)

    def _predict_direct(self, horizon: int) -> pd.Series:
        """Строит прогноз сразу на весь горизонт текущего окна"""
        alpha_inflow_pred, _ = utils.direct_forecast(self.alpha_inflow, self.window_approx, horizon, self.approx_method)
        alpha_outflow_pred, _ = utils.direct_forecast(self.alpha_outflow, self.window_approx, horizon, self.approx_method)

        last_r_in = self.r_inflow.iloc[-1]
        last_r_out = self.r_outflow.iloc[-1]

        alpha_in_for_r = alpha_inflow_pred[-1]
        alpha_out_for_r = alpha_outflow_pred[-1]
        r_inflow_pred = [last_r_in - alpha_in_for_r * (i+1) for i in range(horizon)]
        r_outflow_pred = [last_r_out - alpha_out_for_r * (i+1) for i in range(horizon)]

        used_inflow = ['alpha'] * horizon
        used_outflow = ['alpha'] * horizon

        if any(r < 0 for r in r_inflow_pred):
            r_inflow_pred, used = utils.direct_forecast(self.r_inflow, self.window_approx, horizon, self.approx_method)
            used_inflow = [f'power_r ({used})'] * horizon
        if any(r < 0 for r in r_outflow_pred):
            r_outflow_pred, used = utils.direct_forecast(self.r_outflow, self.window_approx, horizon, self.approx_method)
            used_outflow = [f'power_r ({used})'] * horizon

        inflow_annual, outflow_annual, total_pred, inflow_int_pred, outflow_int_pred = self._forecast_from_r(
            r_inflow_pred, r_outflow_pred
        )
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     r_inflow_pred=r_inflow_pred, r_outflow_pred=r_outflow_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     alpha_inflow_pred=alpha_inflow_pred, alpha_outflow_pred=alpha_outflow_pred,
                                     used_func_inflow=used_inflow, used_func_outflow=used_outflow)

    def _predict_moving(self, horizon: int) -> pd.Series:
        """Строит прогноз пошагово с обновлением окна после каждого рассчитанного значения"""
        current_alpha_in = self.alpha_inflow.copy()
        current_alpha_out = self.alpha_outflow.copy()
        current_r_in = self.r_inflow.copy()
        current_r_out = self.r_outflow.copy()

        alpha_in_pred = []
        alpha_out_pred = []
        r_in_pred = []
        r_out_pred = []
        used_in = []
        used_out = []

        for step in range(horizon):
            next_alpha_in, u_in = utils.recurrent_forecast(current_alpha_in, self.window_approx, 1, self.approx_method)
            next_alpha_out, u_out = utils.recurrent_forecast(current_alpha_out, self.window_approx, 1, self.approx_method)
            alpha_in_pred.append(next_alpha_in[0])
            alpha_out_pred.append(next_alpha_out[0])

            new_year = self.last_year + step + 1
            current_alpha_in = pd.concat([current_alpha_in, pd.Series([next_alpha_in[0]], index=[new_year])])
            current_alpha_out = pd.concat([current_alpha_out, pd.Series([next_alpha_out[0]], index=[new_year])])

            last_r_in = current_r_in.iloc[-1]
            last_r_out = current_r_out.iloc[-1]
            next_r_in = last_r_in - next_alpha_in[0]
            next_r_out = last_r_out - next_alpha_out[0]

            if next_r_in < 0:
                next_r_in, u_in_r = utils.recurrent_forecast(current_r_in, self.window_approx, 1, self.approx_method)
                next_r_in = next_r_in[0]
                used_in.append(f'power_r ({u_in_r[0]})')
            else:
                used_in.append('alpha')
            if next_r_out < 0:
                next_r_out, u_out_r = utils.recurrent_forecast(current_r_out, self.window_approx, 1, self.approx_method)
                next_r_out = next_r_out[0]
                used_out.append(f'power_r ({u_out_r[0]})')
            else:
                used_out.append('alpha')

            r_in_pred.append(next_r_in)
            r_out_pred.append(next_r_out)

            current_r_in = pd.concat([current_r_in, pd.Series([next_r_in], index=[new_year])])
            current_r_out = pd.concat([current_r_out, pd.Series([next_r_out], index=[new_year])])

        inflow_annual, outflow_annual, total_pred, inflow_int_pred, outflow_int_pred = self._forecast_from_r(
            r_in_pred, r_out_pred
        )
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     r_inflow_pred=r_in_pred, r_outflow_pred=r_out_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     alpha_inflow_pred=alpha_in_pred, alpha_outflow_pred=alpha_out_pred,
                                     used_func_inflow=used_in, used_func_outflow=used_out)
