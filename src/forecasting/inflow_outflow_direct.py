"""Метод прямого прогнозирования интегральных рядов притока и оттока"""
from .inflow_base import InflowMethod
from . import utils
import pandas as pd

class InflowOutflowDirect(InflowMethod):
    """Метод прямой аппроксимации интегрального притока и интегрального оттока"""
    def predict(self, horizon: int) -> pd.Series:
        """Строит прогноз на заданное число шагов вперёд"""
        if self.use_moving_window:
            return self._predict_moving(horizon)
        else:
            return self._predict_direct(horizon)

    def _predict_direct(self, horizon: int) -> pd.Series:

        """Строит прогноз сразу на весь горизонт текущего окна"""
        inflow_int_pred, used_in = utils.direct_forecast(self.inflow_int, self.window_approx, horizon, self.approx_method)
        outflow_int_pred, used_out = utils.direct_forecast(self.outflow_int, self.window_approx, horizon, self.approx_method)

        inflow_annual, outflow_annual, total_pred = self._forecast_from_integrals(inflow_int_pred, outflow_int_pred)
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     used_func_inflow=[f'integral_{used_in}'] * horizon,
                                     used_func_outflow=[f'integral_{used_out}'] * horizon)

    def _predict_moving(self, horizon: int) -> pd.Series:
        """Строит прогноз пошагово с обновлением окна после каждого рассчитанного значения"""
        current_in_int = self.inflow_int.copy()
        current_out_int = self.outflow_int.copy()

        inflow_int_pred = []
        outflow_int_pred = []
        used_in = []
        used_out = []
        for step in range(horizon):
            next_in_int, u_in = utils.recurrent_forecast(current_in_int, self.window_approx, 1, self.approx_method)
            next_out_int, u_out = utils.recurrent_forecast(current_out_int, self.window_approx, 1, self.approx_method)
            inflow_int_pred.append(next_in_int[0])
            outflow_int_pred.append(next_out_int[0])
            used_in.append(f'integral_{u_in[0]}')
            used_out.append(f'integral_{u_out[0]}')
            new_year = self.last_year + step + 1
            current_in_int = pd.concat([current_in_int, pd.Series([next_in_int[0]], index=[new_year])])
            current_out_int = pd.concat([current_out_int, pd.Series([next_out_int[0]], index=[new_year])])

        inflow_annual, outflow_annual, total_pred = self._forecast_from_integrals(inflow_int_pred, outflow_int_pred)
        return self._store_forecasts(inflow_annual, outflow_annual, total_pred,
                                     inflow_int_pred=inflow_int_pred, outflow_int_pred=outflow_int_pred,
                                     used_func_inflow=used_in,
                                     used_func_outflow=used_out)
