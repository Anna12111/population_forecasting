"""Модуль верхнего уровня для запуска прогнозного эксперимента и сборки результатов"""
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from src.data_processing.storage import get_inflow_outflow
from src.data_processing.un_forecast import load_un_forecast, get_un_forecast_description, get_effective_un_forecast_variant
from src.forecasting.inflow_outflow_direct import InflowOutflowDirect
from src.forecasting.integrated_growth_rate import IntegratedGrowthRate
from src.forecasting.growth_rate_speed import GrowthRateSpeed
from src.metrics.evaluator import rolling_forecast
from src.metrics.error_metrics import smape, rmse, mae
from .config import ExperimentConfig

def _safe_percentage_error(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """Абсолютная процентная ошибка на общем индексе; нули в знаменателе игнорируются"""
    if actual is None or forecast is None:
        return pd.Series(dtype=float)
    actual = pd.Series(actual).dropna().astype(float)
    forecast = pd.Series(forecast).dropna().astype(float)
    common = actual.index.intersection(forecast.index)
    if common.empty:
        return pd.Series(dtype=float)
    denom = actual.loc[common].replace(0, np.nan).abs()
    err = (actual.loc[common] - forecast.loc[common]).abs() / denom * 100
    return err.replace([np.inf, -np.inf], np.nan).dropna()

def _safe_accuracy(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """
    Точность по формуле пользователя:
        (1 - abs(N(t) - N'(t)) / N(t)) * 100%
    где actual=N(t), forecast=N'(t)
    """
    err = _safe_percentage_error(actual, forecast)
    if err.empty:
        return pd.Series(dtype=float)
    return 100 - err

def _metric_bundle(actual: pd.Series, forecast: pd.Series) -> Optional[Dict[str, Any]]:
    """Единый набор метрик для сравнения двух рядов на пересекающихся годах"""
    if actual is None or forecast is None:
        return None
    actual = pd.Series(actual).dropna().astype(float)
    forecast = pd.Series(forecast).dropna().astype(float)
    common = actual.index.intersection(forecast.index)
    if common.empty:
        return None
    y_true = actual.loc[common]
    y_pred = forecast.loc[common]
    accuracy = _safe_accuracy(y_true, y_pred)
    return {
        'mape': _safe_percentage_error(y_true, y_pred).mean(),
        'accuracy': accuracy.mean() if not accuracy.empty else None,
        'smape': smape(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'years': common.tolist()
    }

def _make_reference_series(actual: pd.Series, un_forecast: pd.Series, forecast_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
    """
    Опорный ряд для метрик: факт имеет приоритет, а там, где факта нет,
    используется выбранный сценарий ООН. Прогноз ООН не участвует в обучении
    """
    years = pd.Index(forecast_index).astype(int)
    ref = pd.Series(index=years, dtype=float)
    source = pd.Series(index=years, dtype=object)

    if un_forecast is not None:
        un = pd.Series(un_forecast).dropna()
        common_un = years.intersection(un.index.astype(int))
        if not common_un.empty:
            ref.loc[common_un] = un.loc[common_un].astype(float)
            source.loc[common_un] = 'ООН'

    if actual is not None:
        act = pd.Series(actual).dropna()
        common_actual = years.intersection(act.index.astype(int))
        if not common_actual.empty:
            ref.loc[common_actual] = act.loc[common_actual].astype(float)
            source.loc[common_actual] = 'Факт'

    ref = ref.dropna()
    source = source.loc[ref.index]
    return ref, source

def run_rolling_forecast(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Запускает рекуррентное прогнозирование согласно конфигурации
    Возвращает словарь с объединёнными результатами и списком окон

    Прогнозные сценарии ООН WPP подключаются только после построения
    собственного прогноза и используются исключительно как внешний внешний ориентир для сравнения
    В train_data и в рекуррентный rolling_forecast они не передаются
    """
    data = get_inflow_outflow(config.country, migration_policy=config.migration_policy)
    if data is None:
        raise ValueError(f"No inflow/outflow data for country {config.country}")
    data.set_index('year', inplace=True)
    data.index = data.index.astype(int)

    min_year = data.index.min()
    if config.t_start - config.window_approx < min_year:
        raise ValueError(
            f"Недостаточно данных: для первого прогноза нужно {config.window_approx} лет до "
            f"{config.t_start-1}, минимальный год в данных {min_year}."
        )

    method_class = {
        "inflow_outflow_direct": InflowOutflowDirect,
        "integrated_growth_rate": IntegratedGrowthRate,
        "growth_rate_speed": GrowthRateSpeed
    }[config.method]

    params = {
        "window_approx": config.window_approx,
        "use_moving_window": config.use_moving_window,
        "approx_method": config.approx_method,
        "migration_policy": config.migration_policy
    }
    if config.method == "growth_rate_speed":
        params["window_extrap"] = config.window_extrap

    end_forecast_year = config.t_start + config.horizon - 1

    results = rolling_forecast(
        data, method_class, params,
        config.t_start, end_forecast_year,
        config.window_approx, config.window_extrap
    )

    forecast_list = [res['forecast'] for res in results]
    combined_forecast = pd.concat(forecast_list).sort_index()

    inflow_forecast_list = [res['forecast_dict']['inflow_forecast'] for res in results
                            if 'inflow_forecast' in res['forecast_dict']]
    combined_inflow_forecast = pd.concat(inflow_forecast_list).sort_index() if inflow_forecast_list else None

    outflow_forecast_list = [res['forecast_dict']['outflow_forecast'] for res in results
                             if 'outflow_forecast' in res['forecast_dict']]
    combined_outflow_forecast = pd.concat(outflow_forecast_list).sort_index() if outflow_forecast_list else None

    actual_years = combined_forecast.index.intersection(data.index)
    combined_actual = data.loc[actual_years, 'total_actual']
    combined_inflow_actual = data.loc[actual_years, 'inflow'] if 'inflow' in data.columns else None
    combined_outflow_actual = data.loc[actual_years, 'outflow'] if 'outflow' in data.columns else None

    un_variant_requested = getattr(config, 'un_forecast_variant', None)
    un_variant = get_effective_un_forecast_variant(un_variant_requested) if un_variant_requested else None
    un_forecast_data = None
    combined_un_forecast = None
    combined_un_inflow_forecast = None
    combined_un_outflow_forecast = None
    un_metrics = None
    un_error_series = pd.Series(dtype=float)
    un_mape_series = pd.Series(dtype=float)
    un_accuracy_series = pd.Series(dtype=float)
    un_load_error = None

    if un_variant:
        try:
            un_forecast_data = load_un_forecast(
                country=config.country,
                variant=un_variant,
                migration_policy=config.migration_policy,
                start_year=config.t_start,
                end_year=end_forecast_year,
            )
            if un_forecast_data is not None and not un_forecast_data.empty:
                un_idx = un_forecast_data.set_index('year')
                combined_un_forecast = un_idx['un_total_forecast']
                combined_un_inflow_forecast = un_idx['un_inflow']
                combined_un_outflow_forecast = un_idx['un_outflow']

                pop_un_metrics = _metric_bundle(combined_un_forecast, combined_forecast)
                inflow_un_metrics = _metric_bundle(combined_un_inflow_forecast, combined_inflow_forecast)
                outflow_un_metrics = _metric_bundle(combined_un_outflow_forecast, combined_outflow_forecast)
                un_error_series = _safe_percentage_error(combined_un_forecast, combined_forecast)
                un_mape_series = un_error_series.expanding().mean()
                un_accuracy_series = _safe_accuracy(combined_un_forecast, combined_forecast)

                un_metrics = {
                    'variant': un_variant,
                    'description': get_un_forecast_description(un_variant),
                    'requested_variant': un_variant_requested if un_variant_requested != un_variant else None,
                    'population': pop_un_metrics,
                    'inflow': inflow_un_metrics,
                    'outflow': outflow_un_metrics,
                    'error_series': un_error_series.tolist(),
                    'mape_series': un_mape_series.tolist(),
                    'accuracy_series': un_accuracy_series.tolist(),
                    'years_with_un_forecast': un_error_series.index.tolist()
                }
            else:
                un_metrics = {
                    'variant': un_variant,
                    'description': get_un_forecast_description(un_variant),
                    'requested_variant': un_variant_requested if un_variant_requested != un_variant else None,
                    'info': 'Для выбранной страны/периода нет строк в прогнозе ООН.'
                }
        except Exception as exc:
            un_load_error = str(exc)
            un_metrics = {
                'variant': un_variant,
                'description': get_un_forecast_description(un_variant),
                'requested_variant': un_variant_requested if un_variant_requested != un_variant else None,
                'error': un_load_error
            }

    error_series = _safe_percentage_error(combined_actual, combined_forecast)
    mape_series = error_series.expanding().mean()
    accuracy_series = _safe_accuracy(combined_actual, combined_forecast)
    overall_mape = error_series.mean() if not error_series.empty else None

    inflow_metrics = _metric_bundle(combined_inflow_actual, combined_inflow_forecast)
    outflow_metrics = _metric_bundle(combined_outflow_actual, combined_outflow_forecast)
    pop_metrics = _metric_bundle(combined_actual, combined_forecast)
    if pop_metrics is not None:
        metrics = {
            'mape': overall_mape,
            'accuracy': pop_metrics['accuracy'],
            'smape': pop_metrics['smape'],
            'rmse': pop_metrics['rmse'],
            'mae': pop_metrics['mae'],
            'inflow': inflow_metrics,
            'outflow': outflow_metrics,
            'error_series': error_series.tolist(),
            'mape_series': mape_series.tolist(),
            'accuracy_series': accuracy_series.tolist(),
            'years_with_actual': error_series.index.tolist()
        }
    else:
        metrics = {"info": "Нет фактических данных для оценки"}

    reference_pop, reference_pop_source = _make_reference_series(
        combined_actual, combined_un_forecast, combined_forecast.index
    )
    reference_inflow, reference_inflow_source = _make_reference_series(
        combined_inflow_actual, combined_un_inflow_forecast, combined_forecast.index
    )
    reference_outflow, reference_outflow_source = _make_reference_series(
        combined_outflow_actual, combined_un_outflow_forecast, combined_forecast.index
    )

    reference_error_series = _safe_percentage_error(reference_pop, combined_forecast)
    reference_mape_series = reference_error_series.expanding().mean()
    reference_accuracy_series = _safe_accuracy(reference_pop, combined_forecast)
    reference_metrics = {
        'description': 'Факт используется там, где он доступен; для будущих лет используется выбранный сценарий ООН.',
        'population': _metric_bundle(reference_pop, combined_forecast),
        'inflow': _metric_bundle(reference_inflow, combined_inflow_forecast),
        'outflow': _metric_bundle(reference_outflow, combined_outflow_forecast),
        'years': reference_error_series.index.tolist(),
        'source_by_year': reference_pop_source.loc[reference_error_series.index].to_dict() if not reference_error_series.empty else {}
    }

    train_data = data[data.index < config.t_start]

    result = {
        'config': config,
        'results': results,
        'combined_forecast': combined_forecast,
        'combined_inflow_forecast': combined_inflow_forecast,
        'combined_outflow_forecast': combined_outflow_forecast,
        'combined_actual': combined_actual,
        'combined_inflow_actual': combined_inflow_actual,
        'combined_outflow_actual': combined_outflow_actual,
        'combined_un_forecast': combined_un_forecast,
        'combined_un_inflow_forecast': combined_un_inflow_forecast,
        'combined_un_outflow_forecast': combined_un_outflow_forecast,
        'combined_reference_forecast': reference_pop,
        'combined_reference_source': reference_pop_source,
        'combined_reference_inflow': reference_inflow,
        'combined_reference_outflow': reference_outflow,
        'un_forecast_data': un_forecast_data,
        'un_forecast_variant': un_variant,
        'un_load_error': un_load_error,
        'train': train_data,
        'metrics': metrics,
        'un_metrics': un_metrics,
        'reference_metrics': reference_metrics,
        'error_series': error_series,
        'mape_series': mape_series,
        'accuracy_series': accuracy_series,
        'un_error_series': un_error_series,
        'un_mape_series': un_mape_series,
        'un_accuracy_series': un_accuracy_series,
        'reference_error_series': reference_error_series,
        'reference_mape_series': reference_mape_series,
        'reference_accuracy_series': reference_accuracy_series
    }
    return result
