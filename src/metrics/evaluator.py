"""Модуль оценки и запуска прогнозирования по скользящим окнам"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Type
from .error_metrics import mape, rmse, mae, smape

class Evaluator:
    """Небольшой класс-обёртка для расчёта набора метрик на общем индексе факта и прогноза"""
    def __init__(self, actual: pd.Series, forecast: pd.Series) -> None:
        """Инициализирует объект и сохраняет параметры, которые используются при дальнейшем расчёте"""
        self.actual = actual
        self.forecast = forecast

    def compute_all(self) -> Dict[str, float]:
        """Рассчитывает основной набор метрик на пересечении индексов факта и прогноза"""
        common_index = self.actual.index.intersection(self.forecast.index)
        y_true = self.actual.loc[common_index] if len(common_index) else self.actual.iloc[:0]
        y_pred = self.forecast.loc[common_index] if len(common_index) else self.forecast.iloc[:0]
        return {
            'mape': mape(y_true, y_pred),
            'smape': smape(y_true, y_pred),
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred)
        }

def rolling_forecast(data: pd.DataFrame, method_class: Type[Any], method_params: Dict[str, Any],
                     start_forecast_year: int, end_forecast_year: int,
                     window_approx: int, window_extrap: int) -> List[Dict[str, Any]]:
    """
    Выполняет прогнозирование по последовательным окнам

    На каждом шаге метод получает исторический ряд от первого доступного года
    до конца текущего окна аппроксимации
    Последние window_approx наблюдений используются для подбора локальной аппроксимации,
    а более ранняя часть ряда сохраняет общую базу интегральных показателей
    """
    data = data.copy()
    data.index = data.index.astype(int)
    min_year = int(data.index.min())

    if start_forecast_year - window_approx < min_year:
        raise ValueError(
            f"Недостаточно данных: для первого прогноза нужно {window_approx} лет до "
            f"{start_forecast_year-1}, минимальный год в данных {min_year}."
        )

    current_data = data.copy()
    current_data['source'] = 'historical'

    results = []
    current_year = int(start_forecast_year)

    while current_year <= end_forecast_year:
        approx_end = current_year - 1
        approx_start = current_year - window_approx
        current_horizon = min(window_extrap, end_forecast_year - current_year + 1)
        forecast_end = current_year + current_horizon - 1

        history_years = list(range(min_year, approx_end + 1))
        missing = set(history_years) - set(current_data.index)
        if missing:
            break

        train_data = current_data.loc[history_years]

        model = method_class(method_params)
        model.fit(train_data)

        forecast_total = model.predict(current_horizon)

        forecast_dict = {}
        for attr, key in [
            ('inflow_forecast', 'inflow_forecast'),
            ('outflow_forecast', 'outflow_forecast'),
            ('r_inflow_forecast', 'r_inflow_forecast'),
            ('r_outflow_forecast', 'r_outflow_forecast'),
            ('inflow_int_forecast', 'inflow_int_forecast'),
            ('outflow_int_forecast', 'outflow_int_forecast'),
            ('alpha_inflow_forecast', 'alpha_inflow_forecast'),
            ('alpha_outflow_forecast', 'alpha_outflow_forecast'),
            ('used_func_inflow_series', 'used_func_inflow'),
            ('used_func_outflow_series', 'used_func_outflow'),
        ]:
            value = getattr(model, attr, None)
            if value is not None:
                forecast_dict[key] = value

        forecast_years = list(range(current_year, forecast_end + 1))
        actual_data = data[data.index.isin(forecast_years)]

        result = {
            'approx_start': approx_start,
            'approx_end': approx_end,
            'forecast_start': current_year,
            'forecast_end': forecast_end,
            'train_data': train_data,
            'actual': actual_data,
            'forecast': forecast_total,
            'forecast_dict': forecast_dict,
            'inflow_forecast': forecast_dict.get('inflow_forecast'),
            'outflow_forecast': forecast_dict.get('outflow_forecast'),
            'years_with_actual': actual_data.index.tolist()
        }
        results.append(result)

        new_rows = []
        for year in forecast_years:
            if year not in current_data.index:
                pop_forecast = forecast_total.loc[year] if year in forecast_total.index else None
                inflow_forecast = forecast_dict.get('inflow_forecast')
                outflow_forecast = forecast_dict.get('outflow_forecast')
                inflow_val = inflow_forecast.loc[year] if inflow_forecast is not None and year in inflow_forecast.index else None
                outflow_val = outflow_forecast.loc[year] if outflow_forecast is not None and year in outflow_forecast.index else None
                new_rows.append({
                    'year': year,
                    'population': pop_forecast,
                    'inflow': inflow_val,
                    'outflow': outflow_val,
                    'total_actual': None,
                    'source': 'forecast'
                })
        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index('year')
            current_data = pd.concat([current_data, new_df])

        current_year += window_extrap

    return results
