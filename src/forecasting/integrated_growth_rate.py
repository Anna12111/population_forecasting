"""Метод прогнозирования процентных приростов интегральных рядов"""
from .inflow_base import InflowMethod
from . import utils
import pandas as pd

class IntegratedGrowthRate(InflowMethod):
    """Метод аппроксимации процентных приростов интегральных потоков"""
    def predict(self, horizon: int) -> pd.Series:
        """Строит прогноз на заданное число шагов вперёд"""
        if self.use_moving_window:
            return self._predict_moving(horizon)
        else:
            return self._predict_direct(horizon)

    def _predict_direct(self, horizon: int) -> pd.Series:
        """Строит прогноз сразу на весь горизонт текущего окна"""
        r_inflow_pred, used_in = utils.direct_forecast(self.r_inflow, self.window_approx, horizon, self.approx_method)
        r_outflow_pred, used_out = utils.direct_forecast(self.r_outflow, self.window_approx, horizon, self.approx_method)

        inflow_annual, outflow_annual, total_pred, inflow_int_pred, outflow_int_pred = self._forecast_from_r(
            r_inflow_pred, r_outflow_pred
        )
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     r_inflow_pred=r_inflow_pred, r_outflow_pred=r_outflow_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     used_func_inflow=[f'r_{used_in}'] * horizon,
                                     used_func_outflow=[f'r_{used_out}'] * horizon)

    def _predict_moving(self, horizon: int) -> pd.Series:
        """Строит прогноз пошагово с обновлением окна после каждого рассчитанного значения"""
        current_r_inflow = self.r_inflow.copy()
        current_r_outflow = self.r_outflow.copy()

        r_inflow_pred = []
        r_outflow_pred = []
        used_in = []
        used_out = []
        for step in range(horizon):
            next_r_in, u_in = utils.recurrent_forecast(current_r_inflow, self.window_approx, 1, self.approx_method)
            next_r_out, u_out = utils.recurrent_forecast(current_r_outflow, self.window_approx, 1, self.approx_method)
            r_inflow_pred.append(next_r_in[0])
            r_outflow_pred.append(next_r_out[0])
            used_in.append(f'r_{u_in[0]}')
            used_out.append(f'r_{u_out[0]}')
            new_year = self.last_year + step + 1
            current_r_inflow = pd.concat([current_r_inflow, pd.Series([next_r_in[0]], index=[new_year])])
            current_r_outflow = pd.concat([current_r_outflow, pd.Series([next_r_out[0]], index=[new_year])])

        inflow_annual, outflow_annual, total_pred, inflow_int_pred, outflow_int_pred = self._forecast_from_r(
            r_inflow_pred, r_outflow_pred
        )
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     r_inflow_pred=r_inflow_pred, r_outflow_pred=r_outflow_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     used_func_inflow=used_in,
                                     used_func_outflow=used_out)
