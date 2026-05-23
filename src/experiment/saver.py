"""Модуль сохранения результатов экспериментов в CSV, JSON и Excel"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../outputs')

def _safe_float(v: Any) -> Any:
    """Преобразует значения numpy и pandas в обычные типы Python для корректной сериализации"""
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if not pd.isna(v) else None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v

def _to_jsonable(v: Any) -> Any:
    """Рекурсивно подготавливает словари и списки к записи в JSON"""
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return _safe_float(v)

def _series_to_dict_serializable(d: Any) -> Any:
    """Приводит структуру с метриками и рядами к JSON-совместимому виду"""
    return _to_jsonable(d or {})

def build_combined_export_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Собирает итоговую таблицу для CSV/Excel-выгрузки приложения"""
    series_items = [
        result.get('combined_forecast'), result.get('combined_actual'), result.get('combined_un_forecast'), result.get('combined_reference_forecast'),
        result.get('combined_inflow_forecast'), result.get('combined_inflow_actual'), result.get('combined_un_inflow_forecast'), result.get('combined_reference_inflow'),
        result.get('combined_outflow_forecast'), result.get('combined_outflow_actual'), result.get('combined_un_outflow_forecast'), result.get('combined_reference_outflow'),
    ]
    train = result.get('train')
    years = set()
    if train is not None and not train.empty:
        years |= set(train.index)
    for s in series_items:
        if s is not None:
            years |= set(pd.Series(s).dropna().index)
    table = pd.DataFrame(index=sorted(years))
    table.index.name = 'Год'

    if train is not None and not train.empty:
        table['Население (ист.)'] = train['total_actual'] if 'total_actual' in train.columns else train.get('population')
        if 'inflow' in train.columns:
            table['Приток (ист.)'] = train['inflow']
        if 'outflow' in train.columns:
            table['Отток (ист.)'] = train['outflow']

    if result.get('combined_forecast') is not None:
        table['Население (наш прогноз)'] = result['combined_forecast']
    if result.get('combined_actual') is not None:
        table.loc[result['combined_actual'].index, 'Население (факт)'] = result['combined_actual']
    if result.get('combined_un_forecast') is not None:
        table.loc[result['combined_un_forecast'].index, 'Население (ООН прогноз)'] = result['combined_un_forecast']
    if result.get('combined_reference_forecast') is not None:
        table.loc[result['combined_reference_forecast'].index, 'Население (опорный ряд)'] = result['combined_reference_forecast']
    if result.get('combined_reference_source') is not None:
        src = result['combined_reference_source']
        table.loc[src.index, 'Источник опорного ряда'] = src

    if result.get('combined_inflow_forecast') is not None:
        table['Приток (наш прогноз)'] = result['combined_inflow_forecast']
    if result.get('combined_inflow_actual') is not None:
        table.loc[result['combined_inflow_actual'].index, 'Приток (факт)'] = result['combined_inflow_actual']
    if result.get('combined_un_inflow_forecast') is not None:
        table.loc[result['combined_un_inflow_forecast'].index, 'Приток (ООН прогноз)'] = result['combined_un_inflow_forecast']
    if result.get('combined_reference_inflow') is not None:
        table.loc[result['combined_reference_inflow'].index, 'Приток (опорный ряд)'] = result['combined_reference_inflow']

    if result.get('combined_outflow_forecast') is not None:
        table['Отток (наш прогноз)'] = result['combined_outflow_forecast']
    if result.get('combined_outflow_actual') is not None:
        table.loc[result['combined_outflow_actual'].index, 'Отток (факт)'] = result['combined_outflow_actual']
    if result.get('combined_un_outflow_forecast') is not None:
        table.loc[result['combined_un_outflow_forecast'].index, 'Отток (ООН прогноз)'] = result['combined_un_outflow_forecast']
    if result.get('combined_reference_outflow') is not None:
        table.loc[result['combined_reference_outflow'].index, 'Отток (опорный ряд)'] = result['combined_reference_outflow']

    for base_col, ref_col, out_col in [
        ('Население (наш прогноз)', 'Население (факт)', 'Ошибка населения к факту (%)'),
        ('Население (наш прогноз)', 'Население (ООН прогноз)', 'Ошибка населения к ООН (%)'),
        ('Население (наш прогноз)', 'Население (опорный ряд)', 'Ошибка населения к опорному ряду (%)'),
        ('Приток (наш прогноз)', 'Приток (факт)', 'Ошибка притока к факту (%)'),
        ('Приток (наш прогноз)', 'Приток (ООН прогноз)', 'Ошибка притока к ООН (%)'),
        ('Приток (наш прогноз)', 'Приток (опорный ряд)', 'Ошибка притока к опорному ряду (%)'),
        ('Отток (наш прогноз)', 'Отток (факт)', 'Ошибка оттока к факту (%)'),
        ('Отток (наш прогноз)', 'Отток (ООН прогноз)', 'Ошибка оттока к ООН (%)'),
        ('Отток (наш прогноз)', 'Отток (опорный ряд)', 'Ошибка оттока к опорному ряду (%)'),
    ]:
        if base_col in table.columns and ref_col in table.columns:
            denom = table[ref_col].replace(0, np.nan).abs()
            table[out_col] = np.where(
                table[base_col].notna() & table[ref_col].notna(),
                (table[base_col] - table[ref_col]).abs() / denom * 100,
                np.nan,
            )
            accuracy_col = out_col.replace('Ошибка', 'Точность')
            table[accuracy_col] = 100 - table[out_col]

    return table

def save_experiment(result: Dict[str, Any], name: Optional[str] = None) -> str:
    """
    Сохраняет результаты одного эксперимента
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, name)

    with open(path + "_config.json", "w", encoding="utf-8") as f:
        json.dump(result['config'].dict(), f, indent=2, ensure_ascii=False)

    with open(path + "_metrics.json", "w", encoding="utf-8") as f:
        metrics_payload = {
            'actual_metrics': result.get('metrics'),
            'un_metrics': result.get('un_metrics'),
            'reference_metrics': result.get('reference_metrics')
        }
        json.dump(_series_to_dict_serializable(metrics_payload), f, indent=2, ensure_ascii=False)

    result['combined_forecast'].to_csv(path + "_forecast.csv", header=True)

    build_combined_export_table(result).to_csv(path + "_full_table.csv", encoding="utf-8-sig")

    if 'figure' in result and result['figure'] is not None:
        result['figure'].savefig(path + "_plot.png", dpi=150, bbox_inches='tight')

    print(f"Experiment saved to {path}_*")
    return path

def _mape(actual: Optional[pd.Series], forecast: Optional[pd.Series]) -> float:
    """Считает среднюю абсолютную процентную ошибку для двух рядов на общем индексе"""
    actual = pd.Series(actual).dropna()
    forecast = pd.Series(forecast).dropna()
    common = actual.index.intersection(forecast.index)
    if len(common) == 0:
        return np.nan
    denom = actual.loc[common].replace(0, np.nan)
    err = (actual.loc[common] - forecast.loc[common]).abs() / denom * 100
    return err.replace([np.inf, -np.inf], np.nan).mean()

def _accuracy(actual: Optional[pd.Series], forecast: Optional[pd.Series]) -> float:
    """Считает среднюю точность прогноза для двух рядов на общем индексе"""
    actual = pd.Series(actual).dropna()
    forecast = pd.Series(forecast).dropna()
    common = actual.index.intersection(forecast.index)
    if len(common) == 0:
        return np.nan
    denom = actual.loc[common].replace(0, np.nan).abs()
    err = (actual.loc[common] - forecast.loc[common]).abs() / denom * 100
    return (100 - err.replace([np.inf, -np.inf], np.nan)).mean()

def _mae(actual: Optional[pd.Series], forecast: Optional[pd.Series]) -> float:
    """Считает среднюю абсолютную ошибку для двух рядов на общем индексе"""
    actual = pd.Series(actual).dropna()
    forecast = pd.Series(forecast).dropna()
    common = actual.index.intersection(forecast.index)
    if len(common) == 0:
        return np.nan
    return (actual.loc[common] - forecast.loc[common]).abs().mean()

def _add_actual_components(full_data: pd.DataFrame, window_extrap: int) -> pd.DataFrame:
    """
    Добавляет в таблицу фактические компоненты балансовой модели

    Функция рассчитывает годовое изменение населения, интегральные притоки и оттоки,
    процентные приросты интегральных рядов и скорости изменения этих приростов
    Полученные столбцы используются в Excel-выгрузке для сравнения фактических
    и прогнозных компонент модели
    """
    df = full_data.copy()
    df.index = df.index.astype(int)
    df = df.sort_index()

    df['actual_net_change'] = df['inflow'] - df['outflow']

    if 'total_actual' not in df.columns:
        df['total_actual'] = df['population'] + df['actual_net_change']

    df['inflow_int_actual'] = df['inflow'].cumsum()
    df['outflow_int_actual'] = df['outflow'].cumsum()

    df['r_inflow_actual'] = df['inflow_int_actual'].pct_change()
    df['r_outflow_actual'] = df['outflow_int_actual'].pct_change()
    df['r_inflow_actual_pct'] = df['r_inflow_actual'] * 100
    df['r_outflow_actual_pct'] = df['r_outflow_actual'] * 100

    df['alpha_inflow_actual'] = np.nan
    df['alpha_outflow_actual'] = np.nan
    w = int(window_extrap)

    for pos in range(w, len(df)):
        df.iloc[pos, df.columns.get_loc('alpha_inflow_actual')] = (
            df['r_inflow_actual'].iloc[pos - w + 1] - df['r_inflow_actual'].iloc[pos]
        ) / w
        df.iloc[pos, df.columns.get_loc('alpha_outflow_actual')] = (
            df['r_outflow_actual'].iloc[pos - w + 1] - df['r_outflow_actual'].iloc[pos]
        ) / w

    df['alpha_inflow_actual_pct'] = df['alpha_inflow_actual'] * 100
    df['alpha_outflow_actual_pct'] = df['alpha_outflow_actual'] * 100
    return df

def _add_forecast_series(detail_df: pd.DataFrame, res: Dict[str, Any]) -> pd.DataFrame:
    """Переносит прогнозные ряды текущего окна в детальную таблицу Excel"""
    forecast_pop = res['forecast']
    detail_df.loc[forecast_pop.index, 'Прогноз_население_конец_года'] = forecast_pop

    forecast_dict = res.get('forecast_dict', {}) or {}
    name_map = {
        'inflow_forecast': 'Прогноз_годовой_приток',
        'outflow_forecast': 'Прогноз_годовой_отток',
        'r_inflow_forecast': 'Прогноз_r_ИП_доля',
        'r_outflow_forecast': 'Прогноз_r_ИО_доля',
        'inflow_int_forecast': 'Прогноз_ИП',
        'outflow_int_forecast': 'Прогноз_ИО',
        'alpha_inflow_forecast': 'Прогноз_alpha_ИП_доля',
        'alpha_outflow_forecast': 'Прогноз_alpha_ИО_доля',
        'used_func_inflow': 'Функция_притока',
        'used_func_outflow': 'Функция_оттока'
    }
    for key, series in forecast_dict.items():
        if series is not None:
            col_name = name_map.get(key, key)
            detail_df.loc[series.index, col_name] = series

    if 'Прогноз_r_ИП_доля' in detail_df.columns:
        detail_df['Прогноз_r_ИП_%'] = detail_df['Прогноз_r_ИП_доля'] * 100
    if 'Прогноз_r_ИО_доля' in detail_df.columns:
        detail_df['Прогноз_r_ИО_%'] = detail_df['Прогноз_r_ИО_доля'] * 100
    if 'Прогноз_alpha_ИП_доля' in detail_df.columns:
        detail_df['Прогноз_alpha_ИП_%'] = detail_df['Прогноз_alpha_ИП_доля'] * 100
    if 'Прогноз_alpha_ИО_доля' in detail_df.columns:
        detail_df['Прогноз_alpha_ИО_%'] = detail_df['Прогноз_alpha_ИО_доля'] * 100

    return detail_df

def _prepare_un_forecast_excel(un_forecast: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Подготавливает прогноз ООН к объединению с детальной таблицей окна"""
    if un_forecast is None:
        return None
    df = un_forecast.copy()
    if df.empty:
        return None
    if 'year' in df.columns:
        df = df.set_index('year')
    df.index = df.index.astype(int)
    return df.rename(columns={
        'un_population_start': 'ООН_население_начало_года',
        'un_births': 'ООН_рождаемость',
        'un_deaths': 'ООН_смертность',
        'un_net_migration': 'ООН_чистая_миграция',
        'un_inflow': 'ООН_годовой_приток',
        'un_outflow': 'ООН_годовой_отток',
        'un_total_forecast': 'ООН_население_конец_года',
        'variant': 'ООН_сценарий',
        'country': 'ООН_страна',
    })

def _join_un_forecast(detail_df: pd.DataFrame, un_forecast_excel: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Добавляет к детальной таблице значения выбранного сценария ООН"""
    if un_forecast_excel is None or un_forecast_excel.empty:
        return detail_df
    cols = [
        'ООН_сценарий', 'ООН_население_начало_года', 'ООН_рождаемость',
        'ООН_смертность', 'ООН_чистая_миграция', 'ООН_годовой_приток',
        'ООН_годовой_отток', 'ООН_население_конец_года'
    ]
    present = [c for c in cols if c in un_forecast_excel.columns]
    return detail_df.join(un_forecast_excel[present], how='left')

def _add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в таблицу столбцы ошибок и точности для доступных пар факт-прогноз"""
    pairs = [
        ('Факт_население_конец_года', 'Прогноз_население_конец_года', 'Ошибка_население_%'),
        ('Факт_годовой_приток', 'Прогноз_годовой_приток', 'Ошибка_годовой_приток_%'),
        ('Факт_годовой_отток', 'Прогноз_годовой_отток', 'Ошибка_годовой_отток_%'),
        ('ООН_население_конец_года', 'Прогноз_население_конец_года', 'Ошибка_население_к_ООН_%'),
        ('ООН_годовой_приток', 'Прогноз_годовой_приток', 'Ошибка_годовой_приток_к_ООН_%'),
        ('ООН_годовой_отток', 'Прогноз_годовой_отток', 'Ошибка_годовой_отток_к_ООН_%'),
        ('Факт_ИП', 'Прогноз_ИП', 'Ошибка_ИП_%'),
        ('Факт_ИО', 'Прогноз_ИО', 'Ошибка_ИО_%'),
        ('Факт_r_ИП_доля', 'Прогноз_r_ИП_доля', 'Ошибка_r_ИП_%'),
        ('Факт_r_ИО_доля', 'Прогноз_r_ИО_доля', 'Ошибка_r_ИО_%'),
        ('Факт_alpha_ИП_доля', 'Прогноз_alpha_ИП_доля', 'Ошибка_alpha_ИП_%'),
        ('Факт_alpha_ИО_доля', 'Прогноз_alpha_ИО_доля', 'Ошибка_alpha_ИО_%'),
    ]
    for actual_col, forecast_col, error_col in pairs:
        if actual_col in df.columns and forecast_col in df.columns:
            denom = df[actual_col].replace(0, np.nan)
            df[error_col] = np.where(
                df[actual_col].notna() & df[forecast_col].notna(),
                100 * (df[forecast_col] - df[actual_col]).abs() / denom.abs(),
                np.nan
            )
            accuracy_col = error_col.replace('Ошибка', 'Точность')
            df[accuracy_col] = 100 - df[error_col]
    return df

def _format_workbook(writer: pd.ExcelWriter, sheet_names: Iterable[str]) -> None:
    """
    Лёгкое оформление Excel через openpyxl, потому что проект уже использует
    pd.ExcelWriter(engine='openpyxl') для выгрузки результатов
    """
    wb = writer.book
    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col[:80]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 34)

def save_backtest(results: Sequence[Dict[str, Any]], full_data: pd.DataFrame, backtest_type: str, country: str, method: str,
                  window_approx: int, window_extrap: int, migration_policy: int,
                  approx_method: str, use_moving_window: bool, name: Optional[str] = None,
                  un_forecast: Optional[pd.DataFrame] = None, un_variant: Optional[str] = None) -> str:
    """
    Сохраняет подробные результаты прогнозирования по окнам в Excel-файл

    В файл попадают сводная таблица по окнам и отдельные листы с фактическими
    и прогнозными компонентами модели
    Такая выгрузка удобна для последующего анализа численности населения,
    притока, оттока и промежуточных расчётных рядов
    """
    if backtest_type != 'inflow':
        raise ValueError("Only 'inflow' backtest type is supported now")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{country}_{method}_windows_{timestamp}"
    path = os.path.join(OUTPUT_DIR, name + ".xlsx")

    full_data = _add_actual_components(full_data, window_extrap)
    un_forecast_excel = _prepare_un_forecast_excel(un_forecast)

    summary = []
    for i, res in enumerate(results):
        actual = res['actual'] if res.get('actual') is not None else pd.DataFrame()
        forecast_dict = res.get('forecast_dict', {}) or {}

        actual_full = full_data.loc[full_data.index.intersection(res['forecast'].index)]
        mape_pop = _mape(actual_full.get('total_actual'), res['forecast'])
        acc_pop = _accuracy(actual_full.get('total_actual'), res['forecast'])
        mape_inf = _mape(actual_full.get('inflow'), forecast_dict.get('inflow_forecast'))
        mape_out = _mape(actual_full.get('outflow'), forecast_dict.get('outflow_forecast'))
        mape_I = _mape(actual_full.get('inflow_int_actual'), forecast_dict.get('inflow_int_forecast'))
        mape_O = _mape(actual_full.get('outflow_int_actual'), forecast_dict.get('outflow_int_forecast'))
        mae_rI = _mae(actual_full.get('r_inflow_actual'), forecast_dict.get('r_inflow_forecast'))
        mae_rO = _mae(actual_full.get('r_outflow_actual'), forecast_dict.get('r_outflow_forecast'))
        mae_aI = _mae(actual_full.get('alpha_inflow_actual'), forecast_dict.get('alpha_inflow_forecast'))
        mae_aO = _mae(actual_full.get('alpha_outflow_actual'), forecast_dict.get('alpha_outflow_forecast'))

        if un_forecast_excel is not None and not un_forecast_excel.empty:
            un_full = un_forecast_excel.loc[un_forecast_excel.index.intersection(res['forecast'].index)]
            mape_pop_un = _mape(un_full.get('ООН_население_конец_года'), res['forecast'])
            acc_pop_un = _accuracy(un_full.get('ООН_население_конец_года'), res['forecast'])
            mape_inf_un = _mape(un_full.get('ООН_годовой_приток'), forecast_dict.get('inflow_forecast'))
            mape_out_un = _mape(un_full.get('ООН_годовой_отток'), forecast_dict.get('outflow_forecast'))
        else:
            mape_pop_un = mape_inf_un = mape_out_un = np.nan
            acc_pop_un = np.nan

        summary.append({
            'Окно': i + 1,
            'Годы аппроксимации': f"{res['approx_start']}-{res['approx_end']}",
            'Годы прогноза': f"{res['forecast_start']}-{res['forecast_end']}",
            'MAPE населения (%)': round(mape_pop, 4) if not np.isnan(mape_pop) else '-',
            'Точность населения (%)': round(acc_pop, 4) if not np.isnan(acc_pop) else '-',
            'MAPE годового притока (%)': round(mape_inf, 4) if not np.isnan(mape_inf) else '-',
            'MAPE годового оттока (%)': round(mape_out, 4) if not np.isnan(mape_out) else '-',
            'MAPE населения к ООН (%)': round(mape_pop_un, 4) if not np.isnan(mape_pop_un) else '-',
            'Точность населения к ООН (%)': round(acc_pop_un, 4) if not np.isnan(acc_pop_un) else '-',
            'MAPE годового притока к ООН (%)': round(mape_inf_un, 4) if not np.isnan(mape_inf_un) else '-',
            'MAPE годового оттока к ООН (%)': round(mape_out_un, 4) if not np.isnan(mape_out_un) else '-',
            'MAPE ИП (%)': round(mape_I, 4) if not np.isnan(mape_I) else '-',
            'MAPE ИО (%)': round(mape_O, 4) if not np.isnan(mape_O) else '-',
            'MAE r ИП (доля)': round(mae_rI, 8) if not np.isnan(mae_rI) else '-',
            'MAE r ИО (доля)': round(mae_rO, 8) if not np.isnan(mae_rO) else '-',
            'MAE alpha ИП (доля)': round(mae_aI, 8) if not np.isnan(mae_aI) else '-',
            'MAE alpha ИО (доля)': round(mae_aO, 8) if not np.isnan(mae_aO) else '-'
        })

    sheet_names = []
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        params_data = {
            'Параметр': [
                'Страна', 'Метод', 'Окно аппроксимации (P)', 'Окно экстраполяции (K)',
                'Политика миграции', 'Метод аппроксимации', 'Использовать СОА',
                'Сценарий ООН для сравнения', 'База интегралов', 'Единицы r и alpha в коде'
            ],
            'Значение': [
                country, method, window_approx, window_extrap, migration_policy,
                approx_method, 'Да' if use_moving_window else 'Нет',
                un_variant or '-', f"от {int(full_data.index.min())} года",
                'доли; столбцы с % добавлены отдельно'
            ]
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name='Параметры', index=False)
        sheet_names.append('Параметры')

        pd.DataFrame(summary).to_excel(writer, sheet_name='Сводка MAPE', index=False)
        sheet_names.append('Сводка MAPE')

        actual_cols = [
            'population', 'births', 'deaths', 'net_migration', 'inflow', 'outflow',
            'actual_net_change', 'total_actual', 'inflow_int_actual', 'outflow_int_actual',
            'r_inflow_actual', 'r_outflow_actual', 'r_inflow_actual_pct', 'r_outflow_actual_pct',
            'alpha_inflow_actual', 'alpha_outflow_actual',
            'alpha_inflow_actual_pct', 'alpha_outflow_actual_pct'
        ]
        present_cols = [c for c in actual_cols if c in full_data.columns]
        all_actual = full_data[present_cols].rename(columns={
            'population': 'Факт_население_начало_года',
            'births': 'Факт_рождаемость',
            'deaths': 'Факт_смертность',
            'net_migration': 'Факт_чистая_миграция',
            'inflow': 'Факт_годовой_приток',
            'outflow': 'Факт_годовой_отток',
            'actual_net_change': 'Факт_чистый_прирост',
            'total_actual': 'Факт_население_конец_года',
            'inflow_int_actual': 'Факт_ИП',
            'outflow_int_actual': 'Факт_ИО',
            'r_inflow_actual': 'Факт_r_ИП_доля',
            'r_outflow_actual': 'Факт_r_ИО_доля',
            'r_inflow_actual_pct': 'Факт_r_ИП_%',
            'r_outflow_actual_pct': 'Факт_r_ИО_%',
            'alpha_inflow_actual': 'Факт_alpha_ИП_доля',
            'alpha_outflow_actual': 'Факт_alpha_ИО_доля',
            'alpha_inflow_actual_pct': 'Факт_alpha_ИП_%',
            'alpha_outflow_actual_pct': 'Факт_alpha_ИО_%'
        })
        all_actual.index.name = 'Год'
        all_actual.to_excel(writer, sheet_name='Факт компоненты')
        sheet_names.append('Факт компоненты')

        if un_forecast_excel is not None and not un_forecast_excel.empty:
            un_forecast_excel.to_excel(writer, sheet_name='ООН прогноз')
            sheet_names.append('ООН прогноз')

        for i, res in enumerate(results):
            sheet_name = f"Окно_{i+1}"
            start_year = int(full_data.index.min())
            end_year = int(res['forecast_end'])
            years = list(range(start_year, end_year + 1))
            detail = pd.DataFrame(index=years)
            detail.index.name = 'Год'

            detail = detail.join(all_actual, how='left')
            detail = _join_un_forecast(detail, un_forecast_excel)
            detail = _add_forecast_series(detail, res)
            detail = _add_error_columns(detail)

            if 'Прогноз_ИП' in detail.columns and 'Прогноз_ИО' in detail.columns and 'Прогноз_население_конец_года' in detail.columns:
                base_total = full_data['population'].iloc[0]
                detail['Проверка_баланса_прогноз'] = detail['Прогноз_население_конец_года'] - (base_total + detail['Прогноз_ИП'] - detail['Прогноз_ИО'])
            if 'Прогноз_ИП' in detail.columns and 'Прогноз_годовой_приток' in detail.columns:
                detail['Проверка_годовой_приток'] = detail['Прогноз_ИП'].diff() - detail['Прогноз_годовой_приток']
                first_forecast_year = int(res['forecast_start'])
                prev_year = first_forecast_year - 1
                if prev_year in full_data.index and first_forecast_year in detail.index:
                    detail.loc[first_forecast_year, 'Проверка_годовой_приток'] = (
                        detail.loc[first_forecast_year, 'Прогноз_ИП'] - full_data.loc[prev_year, 'inflow_int_actual']
                    ) - detail.loc[first_forecast_year, 'Прогноз_годовой_приток']
            if 'Прогноз_ИО' in detail.columns and 'Прогноз_годовой_отток' in detail.columns:
                detail['Проверка_годовой_отток'] = detail['Прогноз_ИО'].diff() - detail['Прогноз_годовой_отток']
                first_forecast_year = int(res['forecast_start'])
                prev_year = first_forecast_year - 1
                if prev_year in full_data.index and first_forecast_year in detail.index:
                    detail.loc[first_forecast_year, 'Проверка_годовой_отток'] = (
                        detail.loc[first_forecast_year, 'Прогноз_ИО'] - full_data.loc[prev_year, 'outflow_int_actual']
                    ) - detail.loc[first_forecast_year, 'Прогноз_годовой_отток']

            detail.to_excel(writer, sheet_name=sheet_name)
            sheet_names.append(sheet_name)

        _format_workbook(writer, sheet_names)

    return path
