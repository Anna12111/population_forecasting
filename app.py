import glob
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_processing.storage import (
    get_available_countries,
    get_available_years_for_country,
    get_inflow_outflow,
)
from src.data_processing.un_forecast import (
    get_un_forecast_description,
    get_un_forecast_variants,
)
from src.experiment.config import ExperimentConfig
from src.experiment.runner import run_rolling_forecast
from src.experiment.saver import build_combined_export_table, save_backtest, save_experiment

METHOD_LABELS = {
    "inflow_outflow_direct": "Прямой прогноз потоков",
    "integrated_growth_rate": "Темпы прироста интегралов",
    "growth_rate_speed": "Скорость темпов",
}

METHOD_NOTES = {
    "inflow_outflow_direct": "Прогнозируются интегральные ряды притока и оттока, затем по балансу восстанавливается численность населения.",
    "integrated_growth_rate": "Прогнозируются процентные приросты интегральных рядов притока и оттока.",
    "growth_rate_speed": "Прогнозируется скорость изменения процентных приростов интегральных рядов.",
}

MIGRATION_POLICIES = {
    0: "0 - вся миграция в приток",
    1: "1 - положительная в приток, отрицательная в отток",
    2: "2 - вся миграция в отток",
}

APPROX_LABELS = {
    "auto": "Авто",
    "power": "Степенная",
    "linear": "Линейная",
}

PLOT_COLORS = {
    "history": "#3b82f6",
    "forecast": "#f97316",
    "actual": "#10b981",
    "un": "#a855f7",
    "inflow": "#22c55e",
    "outflow": "#ef4444",
    "muted": "#94a3b8",
}

CHART_BG = "rgba(0, 0, 0, 0)"
CHART_GRID = "rgba(148, 163, 184, 0.35)"
CHART_AXIS = "#94a3b8"


st.set_page_config(
    page_title="Прогнозирование численности населения",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_style() -> None:
    """Подключает оформление, близкое к первоначальной версии Streamlit-интерфейса"""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1400px;
            }

            section[data-testid="stSidebar"] {
                border-right: 1px solid rgba(148, 163, 184, 0.25);
            }

            .start-panel,
            .forecast-panel,
            .soft-card,
            .experiment-card {
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 0.75rem;
                padding: 1rem 1.1rem;
                margin: 0.75rem 0 1rem 0;
                background: rgba(148, 163, 184, 0.08);
            }

            .start-panel h1,
            .forecast-panel h2 {
                margin: 0.25rem 0 0.35rem 0;
            }

            .start-panel p,
            .forecast-panel p,
            .soft-card p,
            .experiment-card p {
                margin: 0.15rem 0;
                line-height: 1.5;
            }

            .eyebrow {
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                opacity: 0.75;
            }

            .start-grid,
            .forecast-grid,
            .run-summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.7rem;
                margin-top: 0.8rem;
            }

            .start-card,
            .forecast-card,
            .run-summary-card {
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-left: 4px solid rgba(59, 130, 246, 0.65);
                border-radius: 0.7rem;
                padding: 0.8rem;
                background: rgba(255, 255, 255, 0.03);
            }

            .start-card b,
            .forecast-card b,
            .run-summary-card .label {
                display: block;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .forecast-card .value {
                display: block;
                font-size: 1.15rem;
                font-weight: 700;
                margin-top: 0.15rem;
            }

            .forecast-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.8rem;
            }

            .forecast-meta span {
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 999px;
                padding: 0.25rem 0.55rem;
                font-size: 0.85rem;
            }

            .section-title {
                font-weight: 700;
                font-size: 1.1rem;
                margin: 1.1rem 0 0.6rem 0;
            }

            .status-note,
            .warning-note {
                border-radius: 0.7rem;
                padding: 0.8rem 0.9rem;
                margin: 0.7rem 0 1rem 0;
                line-height: 1.5;
            }

            .status-note {
                background: rgba(59, 130, 246, 0.12);
                border: 1px solid rgba(59, 130, 246, 0.35);
            }

            .warning-note {
                background: rgba(249, 115, 22, 0.12);
                border: 1px solid rgba(249, 115, 22, 0.35);
            }

            [data-testid="stMetric"] {
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 0.75rem;
                padding: 0.75rem;
                background: rgba(148, 163, 184, 0.06);
            }

            @media (max-width: 1100px) {
                .start-grid,
                .forecast-grid,
                .run-summary-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 720px) {
                .start-grid,
                .forecast-grid,
                .run-summary-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_countries() -> list[str]:
    """Загружает список стран из локального набора демографических данных"""
    return get_available_countries()

@st.cache_data(show_spinner=False)
def load_years_for_country(country: str) -> list[int]:
    """Возвращает доступные годы наблюдений для выбранной страны"""
    return get_available_years_for_country(country)

@st.cache_data(show_spinner=False)
def load_inflow_outflow(country: str, migration_policy: int) -> pd.DataFrame:
    """Подготавливает баланс притока и оттока для страны с учётом выбранной политики миграции"""
    return get_inflow_outflow(country, migration_policy)

@st.cache_data(show_spinner=False)
def load_un_forecast_variants() -> list[str]:
    """Возвращает перечень сценариев WPP2024, доступных для внешнего сравнения"""
    return get_un_forecast_variants()

def format_number(value: Any, decimals: int = 2, suffix: str = "") -> str:
    """Форматирует числовое значение для карточек и таблиц, не падая на пустых значениях"""
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(value) >= 1000 and not suffix:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.{decimals}f}{suffix}"

def variant_label(variant: Optional[str]) -> str:
    """Преобразует техническое название сценария ООН в более читаемую подпись"""
    if not variant:
        return "не выбран"
    return variant.replace(" variant", "").replace("-", " ")

def render_start_panel() -> None:
    """Показывает стартовую информационную панель до запуска первого расчёта"""
    st.markdown(
        """
        <div class="start-panel">
            <div class="eyebrow">Параметры эксперимента</div>
            <h1>Начните новый эксперимент</h1>
            <p>Настройте параметры в боковой панели и запустите расчёт.</p>
            <div class="start-grid">
                <div class="start-card"><b>Страна</b><span>Выберете страну для прогнозирования</span></div>
                <div class="start-card"><b>Метод</b><span>Выберете метод прогнозирования</span></div>
                <div class="start-card"><b>Окна</b><span>аппроксимации и экстраполяции</span></div>
                <div class="start-card"><b>Метрики</b><span>MAPE, точность, SMAPE, RMSE и MAE</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_info_card(title: str, text: str) -> None:
    """Выводит компактную информационную карточку с пояснением текущего блока интерфейса"""
    st.markdown(
        f"""
        <div class="soft-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_note(text: str, kind: str = "status") -> None:
    """Показывает короткое пояснение в стиле системной заметки"""
    css_class = "warning-note" if kind == "warning" else "status-note"
    st.markdown(f"<div class=\"{css_class}\">{text}</div>", unsafe_allow_html=True)

def wpp_is_used(res: Dict[str, Any]) -> bool:
    """Проверяет, были ли реально загружены и использованы данные WPP2024 для сравнения"""
    un_forecast = res.get("combined_un_forecast")
    if un_forecast is None:
        return False
    try:
        return not pd.Series(un_forecast).dropna().empty
    except Exception:
        return False

def render_current_run_panel(res: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Показывает краткую панель с параметрами и итоговыми значениями построенного прогноза"""
    config = res.get("config")
    country = getattr(config, "country", params.get("country", "-"))
    method = getattr(config, "method", params.get("method", "-"))
    start_year = int(getattr(config, "t_start", params.get("t_start", 0)))
    horizon = int(getattr(config, "horizon", params.get("horizon", 0)))
    end_year = start_year + horizon - 1 if start_year and horizon else params.get("end_forecast", "-")
    migration_policy = getattr(config, "migration_policy", params.get("migration_policy", "-"))
    un_variant = getattr(config, "un_forecast_variant", params.get("un_forecast_variant", None))
    window_approx = getattr(config, "window_approx", params.get("window_approx", "-"))
    window_extrap = getattr(config, "window_extrap", params.get("window_extrap", "-"))

    forecast = pd.Series(res.get("combined_forecast", pd.Series(dtype=float))).dropna()
    if forecast.empty:
        first_year = start_year
        last_year = end_year
        first_value = None
        last_value = None
        change_value = None
        change_pct = None
    else:
        first_year = int(forecast.index.min())
        last_year = int(forecast.index.max())
        first_value = float(forecast.iloc[0])
        last_value = float(forecast.iloc[-1])
        change_value = last_value - first_value
        change_pct = change_value / abs(first_value) * 100 if first_value else None

    actual_metrics = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
    un_pop_metrics = (res.get("un_metrics") or {}).get("population")
    if wpp_is_used(res):
        if isinstance(actual_metrics, dict) and actual_metrics.get("mape") is not None:
            metric_value = f"Факт MAPE {format_number(actual_metrics.get('mape'), suffix='%')}"
            metric_note = f"точность {format_number(actual_metrics.get('accuracy'), suffix='%')}"
            if un_pop_metrics and un_pop_metrics.get("mape") is not None:
                metric_note += f"; ООН MAPE {format_number(un_pop_metrics.get('mape'), suffix='%')}"
        elif un_pop_metrics and un_pop_metrics.get("mape") is not None:
            metric_value = f"ООН MAPE {format_number(un_pop_metrics.get('mape'), suffix='%')}"
            metric_note = f"точность {format_number(un_pop_metrics.get('accuracy'), suffix='%')}"
        else:
            metric_value = "MAPE -"
            metric_note = "нет сопоставимых значений"
    else:
        metric_value = f"MAPE {format_number(actual_metrics.get('mape'), suffix='%')}"
        metric_note = f"точность {format_number(actual_metrics.get('accuracy'), suffix='%')}"

    change_sign = "+" if change_value is not None and change_value >= 0 else ""
    change_text = "-"
    if change_value is not None:
        change_text = f"{change_sign}{change_value:,.0f}".replace(",", " ")
        if change_pct is not None:
            change_text += f" ({change_sign}{change_pct:.2f}%)"

    meta_items = [
        str(METHOD_LABELS.get(method, method)),
        str(MIGRATION_POLICIES.get(migration_policy, migration_policy)),
    ]
    if wpp_is_used(res):
        meta_items.append(f"ООН: {variant_label(un_variant)}")
    meta_html = "".join(f"<span>{item}</span>" for item in meta_items if item and item != "None")

    panel_html = f"""
    <div class="forecast-panel">
        <div class="eyebrow">Результаты прогноза</div>
        <h2>{country}: {first_year}-{last_year}</h2>
        <p>
            Ниже показаны основные результаты построенного прогноза и параметры выполненного эксперимента.
        </p>
        <div class="forecast-grid">
            <div class="forecast-card">
                <b>Численность населения</b>
                <span class="value">{format_number(last_value, decimals=0)}</span>
                <small>прогноз на {last_year} год</small>
            </div>
            <div class="forecast-card">
                <b>Изменение за горизонт</b>
                <span class="value">{change_text}</span>
                <small>от первого прогнозного года</small>
            </div>
            <div class="forecast-card">
                <b>Метрики</b>
                <span class="value">{metric_value}</span>
                <small>{metric_note}</small>
            </div>
            <div class="forecast-card">
                <b>Окна прогноза</b>
                <span class="value">{len(res.get("results", []))} окон</span>
                <small>аппроксимация {window_approx}, экстраполяция {window_extrap}</small>
            </div>
        </div>
        <div class="forecast-meta">{meta_html}</div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)

def render_metric_cards(metrics: Optional[Dict[str, Any]], prefix: str = "") -> None:
    """Отображает стандартный набор метрик качества прогноза в одном ряду"""
    metrics = metrics or {}
    cols = st.columns(5)
    values = [
        ("MAPE", format_number(metrics.get("mape"), suffix="%")),
        ("Точность", format_number(metrics.get("accuracy"), suffix="%")),
        ("SMAPE", format_number(metrics.get("smape"), suffix="%")),
        ("RMSE", format_number(metrics.get("rmse"), decimals=0)),
        ("MAE", format_number(metrics.get("mae"), decimals=0)),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(f"{prefix}{label}", value)

def build_sidebar_config() -> tuple[Optional[ExperimentConfig], Dict[str, Any], bool]:
    """Собирает параметры эксперимента из боковой панели и формирует объект конфигурации"""
    with st.sidebar:
        st.markdown("## **Параметры эксперимента**")
        st.caption(
            "Выберите страну, метод и параметры прогноза."
        )

        countries = load_countries()
        country = st.selectbox(
            "Страна",
            countries,
            index=countries.index("World") if "World" in countries else 0,
            help="Выберете страну из выпадающего списка. Доступные страны зависят от набора данных, используемого для прогноза",
        )

        method = st.selectbox(
            "Метод",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda value: METHOD_LABELS[value],
            help="Выберете метод прогнозирования",
        )
        st.caption(METHOD_NOTES[method])

        migration_policy = st.selectbox(
            "Миграционная политика",
            options=list(MIGRATION_POLICIES.keys()),
            index=2,
            format_func=lambda value: MIGRATION_POLICIES[value],
            help="Эта настройка применяется до прогноза и меняет учет миграции в исторических данных. Она не влияет на алгоритм, который не использует приток и отток, но может существенно менять ряды для методов, которые их используют",
        )

        st.divider()
        st.markdown("### Размеры окон")
        col_left, col_right = st.columns(2)
        with col_left:
            window_approx = st.number_input(
                "Аппроксимация (P)",
                min_value=2,
                max_value=50,
                value=10,
                help="Количество последних лет, на которых подбирается аппроксимирующая функция.",
            )
        with col_right:
            window_extrap = st.number_input(
                "Экстраполяция (K)",
                min_value=1,
                max_value=20,
                value=5,
                help="Размер одного прогнозного блока. После блока алгоритм переходит к следующему окну.",
            )

        available_years = load_years_for_country(country)
        config = None
        params: Dict[str, Any] = {
            "country": country,
            "method": method,
            "migration_policy": migration_policy,
            "window_approx": window_approx,
            "window_extrap": window_extrap,
        }

        max_year = None
        if available_years:
            min_year = int(min(available_years))
            max_year = int(max(available_years))
            min_t_start = min_year + int(window_approx)
            if min_t_start > max_year:
                st.error(f"Недостаточно данных: для окна {window_approx} лет доступный ряд слишком короткий.")
                t_start = max_year
                end_forecast = max_year
            else:
                default_start = min(max_year - 5, max_year - int(window_approx))
                default_start = max(min_t_start, default_start)
                t_start = st.number_input(
                    "Год начала прогноза",
                    min_value=min_t_start,
                    max_value=max_year + 100,
                    value=default_start,
                    step=1,
                    help="Первый год, для которого модель рассчитывает прогнозное значение.",
                )
                end_forecast = st.number_input(
                    "Год конца прогноза",
                    min_value=int(t_start) + 1,
                    max_value=max_year + 100,
                    value=max(max_year, int(t_start) + 1),
                    step=1,
                    help="Последний год горизонта. Если он выходит за пределы фактических данных, для оценки можно подключить прогноз ООН.",
                )
            horizon = int(end_forecast) - int(t_start) + 1
        else:
            st.error("Для выбранной страны не найден исторический ряд.")
            t_start = 2000
            end_forecast = 2001
            horizon = 2

        st.divider()
        st.markdown("### Прогноз ООН для сравнения")
        forecast_has_future = bool(max_year is not None and int(end_forecast) > max_year)
        compare_with_un = st.checkbox(
            "Использовать прогноз ООН",
            value=forecast_has_future,
            help=(
                "Если прогноз строится только на историческом периоде, этот пункт можно отключить: расчёт станет заметно быстрее. "
                "Для будущих лет прогноз ООН нужен как внешний ориентир, потому что фактических значений ещё нет."
            ),
        )

        un_forecast_variant: Optional[str] = None
        if compare_with_un:
            un_variant_options = load_un_forecast_variants()
            un_forecast_variant = st.selectbox(
                "Сценарий WPP2024",
                options=un_variant_options,
                index=un_variant_options.index("Medium variant") if "Medium variant" in un_variant_options else 0,
                format_func=variant_label,
                help="Сценарий используется только для сравнения после построения собственного прогноза.",
            )
            st.caption(get_un_forecast_description(un_forecast_variant))
            if max_year is not None and int(end_forecast) <= max_year:
                render_note(
                    "Выбран исторический период. Сравнение с WPP2024 не обязательно и может замедлять первый запуск из-за чтения большого Excel-файла.",
                    kind="warning",
                )
        else:
            st.caption("Сравнение с прогнозом ООН отключено. Метрики будут рассчитаны по фактическим данным, если они доступны.")

        if max_year is not None and int(end_forecast) > max_year:
            render_note(
                f"Фактические данные доступны до {max_year} года. Для будущих лет опорный ряд будет использовать сценарий ООН, если он подключён.",
                kind="warning",
            )

        st.divider()
        st.markdown("### Опции")
        approx_method = st.selectbox(
            "Тип аппроксимации",
            options=list(APPROX_LABELS.keys()),
            index=0,
            format_func=lambda value: APPROX_LABELS[value],
            help="Автоматический режим использует степенную модель только для положительных рядов, иначе переключается на линейную.",
        )
        use_moving_window = st.checkbox(
            "Использовать СОА (рекуррентное окно внутри каждого шага)",
            value=False,
            help="Если включено, прогноз строится пошагово: новое прогнозное значение добавляется в окно и влияет на следующий шаг.",
        )

        params.update(
            {
                "t_start": int(t_start),
                "end_forecast": int(end_forecast),
                "horizon": int(horizon),
                "approx_method": approx_method,
                "use_moving_window": use_moving_window,
                "un_forecast_variant": un_forecast_variant,
            }
        )

        if available_years and min_year + int(window_approx) <= max_year:
            config = ExperimentConfig(
                country=country,
                method=method,
                window_approx=int(window_approx),
                window_extrap=int(window_extrap),
                t_start=int(t_start),
                horizon=int(horizon),
                migration_policy=int(migration_policy),
                approx_method=approx_method,
                use_moving_window=use_moving_window,
                un_forecast_variant=un_forecast_variant,
            )

        run_button = st.button("ЗАПУСТИТЬ ПРОГНОЗ", type="primary", use_container_width=True)

    return config, params, run_button


def apply_forecast_chart_layout(fig: go.Figure, *, height: int, y_title: str) -> go.Figure:
    """Применяет единое оформление к графикам прогноза без жёсткого фонового цвета"""
    fig.update_layout(
        height=height,
        template="plotly_white",
        hovermode="x unified",
        font=dict(color=CHART_AXIS),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=18, r=18, t=54, b=36),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    fig.update_xaxes(
        title_text="Год",
        showgrid=True,
        gridcolor=CHART_GRID,
        zeroline=False,
        linecolor=CHART_GRID,
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=CHART_GRID,
        zeroline=False,
        linecolor=CHART_GRID,
    )
    return fig


def build_population_figure(res: Dict[str, Any]) -> go.Figure:
    """Строит отдельный график динамики численности населения"""
    fig = go.Figure()
    train = res["train"]
    hist_pop = train["total_actual"] if "total_actual" in train.columns else train["population"]

    fig.add_trace(
        go.Scatter(
            x=hist_pop.index,
            y=hist_pop.values,
            mode="lines+markers",
            name="Исторические",
            line=dict(color=PLOT_COLORS["history"], width=2.4),
            marker=dict(size=4),
        )
    )

    forecast_pop = res["combined_forecast"]
    fig.add_trace(
        go.Scatter(
            x=forecast_pop.index,
            y=forecast_pop.values,
            mode="lines+markers",
            name="Прогноз",
            line=dict(color=PLOT_COLORS["forecast"], width=2.6, dash="dash"),
            marker=dict(size=6, symbol="square"),
        )
    )

    actual_pop = res.get("combined_actual")
    if actual_pop is not None and not actual_pop.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_pop.index,
                y=actual_pop.values,
                mode="lines+markers",
                name="Факт",
                line=dict(color=PLOT_COLORS["actual"], width=2.2),
                marker=dict(size=6, symbol="triangle-up"),
            )
        )

    un_pop = res.get("combined_un_forecast")
    if un_pop is not None and not un_pop.empty:
        fig.add_trace(
            go.Scatter(
                x=un_pop.index,
                y=un_pop.values,
                mode="lines+markers",
                name=f"ООН: {variant_label(res.get('un_forecast_variant'))}",
                line=dict(color=PLOT_COLORS["un"], width=2.2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            )
        )

    return apply_forecast_chart_layout(fig, height=500, y_title="Население")


def build_flow_figure(res: Dict[str, Any]) -> go.Figure:
    """Строит отдельный график рядов притока и оттока"""
    fig = go.Figure()
    train = res["train"]

    if "inflow" in train.columns:
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train["inflow"],
                mode="lines",
                name="Приток (ист.)",
                line=dict(color=PLOT_COLORS["inflow"], width=1.8),
            )
        )
    if "outflow" in train.columns:
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train["outflow"],
                mode="lines",
                name="Отток (ист.)",
                line=dict(color=PLOT_COLORS["outflow"], width=1.8),
            )
        )

    if res.get("combined_inflow_forecast") is not None:
        fig.add_trace(
            go.Scatter(
                x=res["combined_inflow_forecast"].index,
                y=res["combined_inflow_forecast"].values,
                mode="lines",
                name="Приток (прогноз)",
                line=dict(color=PLOT_COLORS["inflow"], width=2.2, dash="dash"),
            )
        )
    if res.get("combined_outflow_forecast") is not None:
        fig.add_trace(
            go.Scatter(
                x=res["combined_outflow_forecast"].index,
                y=res["combined_outflow_forecast"].values,
                mode="lines",
                name="Отток (прогноз)",
                line=dict(color=PLOT_COLORS["outflow"], width=2.2, dash="dash"),
            )
        )

    if res.get("combined_un_inflow_forecast") is not None and not res["combined_un_inflow_forecast"].empty:
        fig.add_trace(
            go.Scatter(
                x=res["combined_un_inflow_forecast"].index,
                y=res["combined_un_inflow_forecast"].values,
                mode="lines",
                name="Приток (ООН)",
                line=dict(color="#14b8a6", width=2, dash="dot"),
            )
        )
    if res.get("combined_un_outflow_forecast") is not None and not res["combined_un_outflow_forecast"].empty:
        fig.add_trace(
            go.Scatter(
                x=res["combined_un_outflow_forecast"].index,
                y=res["combined_un_outflow_forecast"].values,
                mode="lines",
                name="Отток (ООН)",
                line=dict(color="#fb7185", width=2, dash="dot"),
            )
        )

    if not fig.data:
        fig.add_annotation(
            text="Нет данных для отображения притока и оттока",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=CHART_AXIS),
        )

    return apply_forecast_chart_layout(fig, height=430, y_title="Число людей")

def build_error_series_figure(error_df: pd.DataFrame, column: str, title: str, color: str, y_title: str) -> go.Figure:
    """Строит отдельный график для одного показателя ошибки или точности"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=error_df["Год"],
            y=error_df[column],
            mode="lines+markers",
            name=title,
            line=dict(color=color, width=2.3),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=360,
        template="plotly_white",
        xaxis_title="Год",
        yaxis_title=y_title,
        hovermode="x unified",
        font=dict(color=CHART_AXIS),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=18, r=18, t=58, b=34),
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)
    return fig

def render_forecast_tab(res: Dict[str, Any]) -> None:
    """Отображает сводку прогноза, основной график и таблицу прогнозных значений"""
    uses_wpp = wpp_is_used(res)
    actual_metrics = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
    un_pop_metrics = (res.get("un_metrics") or {}).get("population")

    if uses_wpp:
        render_info_card(
            "Результаты прогноза",
            "Оценки качества показаны раздельно: отдельно по фактическим значениям, если они доступны, "
            "и отдельно по выбранному сценарию WPP2024. Опорный объединённый ряд в этом блоке не используется.",
        )

        has_actual_metrics = isinstance(actual_metrics, dict) and actual_metrics.get("mape") is not None
        if has_actual_metrics:
            st.markdown('<div class="section-title">Метрики к фактическим значениям</div>', unsafe_allow_html=True)
            render_metric_cards(actual_metrics, prefix="Факт: ")
        else:
            st.info("Для выбранного периода нет фактических значений, с которыми можно сравнить прогноз.")

        if un_pop_metrics:
            st.markdown(
                f'<div class="section-title">Метрики к сценарию ООН: {variant_label(res.get("un_forecast_variant"))}</div>',
                unsafe_allow_html=True,
            )
            render_metric_cards(un_pop_metrics, prefix="ООН: ")
        elif res.get("un_load_error"):
            st.warning(f"Прогноз ООН не загружен: {res.get('un_load_error')}")
        else:
            st.info("Для выбранного периода нет сопоставимых строк прогноза WPP2024.")
    else:
        render_info_card(
            "Результаты прогноза",
            "Основные показатели рассчитаны по фактическим значениям на тех годах, где факт доступен для сравнения.",
        )
        st.markdown('<div class="section-title">Метрики к фактическим значениям</div>', unsafe_allow_html=True)
        render_metric_cards(actual_metrics)
        if res.get("un_load_error"):
            st.warning(f"Прогноз ООН не загружен: {res.get('un_load_error')}")

    st.markdown('<div class="section-title">Динамика населения</div>', unsafe_allow_html=True)
    st.plotly_chart(build_population_figure(res), use_container_width=True)

    st.markdown('<div class="section-title">Динамика притока и оттока</div>', unsafe_allow_html=True)
    st.plotly_chart(build_flow_figure(res), use_container_width=True)

    st.markdown('<div class="section-title">Прогнозные значения</div>', unsafe_allow_html=True)
    forecast_df = res["combined_forecast"].to_frame(name="Прогноз населения")
    if res.get("combined_un_forecast") is not None:
        forecast_df["Прогноз ООН населения"] = res["combined_un_forecast"]
        denom = forecast_df["Прогноз ООН населения"].replace(0, np.nan).abs()
        forecast_df["Ошибка к ООН населения (%)"] = (
            forecast_df["Прогноз населения"] - forecast_df["Прогноз ООН населения"]
        ).abs() / denom * 100
        forecast_df["Точность к ООН населения (%)"] = 100 - forecast_df["Ошибка к ООН населения (%)"]
    if res.get("combined_inflow_forecast") is not None:
        forecast_df["Прогноз притока"] = res["combined_inflow_forecast"]
    if res.get("combined_un_inflow_forecast") is not None:
        forecast_df["Прогноз ООН притока"] = res["combined_un_inflow_forecast"]
    if res.get("combined_outflow_forecast") is not None:
        forecast_df["Прогноз оттока"] = res["combined_outflow_forecast"]
    if res.get("combined_un_outflow_forecast") is not None:
        forecast_df["Прогноз ООН оттока"] = res["combined_un_outflow_forecast"]

    fmt = {col: "{:,.0f}" for col in forecast_df.columns if not col.endswith("(%)")}
    fmt.update({col: "{:.2f}" for col in forecast_df.columns if col.endswith("(%)")})
    st.dataframe(forecast_df.style.format(fmt), use_container_width=True)

    full_table = build_combined_export_table(res)
    csv_bytes = full_table.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Скачать полную таблицу расчёта",
        data=csv_bytes,
        file_name=f"forecast_{res['config'].country}_{res['config'].t_start}_{res['config'].t_start + res['config'].horizon - 1}.csv",
        mime="text/csv",
        use_container_width=True,
        help="В файл входят исторические значения, прогноз модели, прогноз ООН, опорный ряд, ошибки и точность.",
    )

def render_errors_tab(res: Dict[str, Any]) -> None:
    """Отображает динамику ошибки и точности по годам прогнозного горизонта"""
    render_info_card(
        "Динамика ошибки",
        "Пошаговая ошибка показывает качество в отдельном году, накопленная MAPE - среднюю ошибку от начала прогнозного периода до текущего года.",
    )

    has_reference_errors = res.get("reference_error_series") is not None and not res["reference_error_series"].empty
    has_actual_errors = res.get("error_series") is not None and not res["error_series"].empty
    has_un_errors = res.get("un_error_series") is not None and not res["un_error_series"].empty

    if not has_reference_errors:
        st.info("Нет данных для построения ряда ошибок: не найдено сопоставимых фактических значений или строк прогноза ООН.")
        return

    ref_source = res.get("combined_reference_source")
    error_df = pd.DataFrame(
        {
            "Год": res["reference_error_series"].index,
            "Источник N(t)": ref_source.reindex(res["reference_error_series"].index).values if ref_source is not None else None,
            "Пошаговая ошибка (%)": res["reference_error_series"].values,
            "Точность (%)": res["reference_accuracy_series"].reindex(res["reference_error_series"].index).values,
            "Накопленная MAPE (%)": res["reference_mape_series"].values,
            "Накопленная средняя точность (%)": res["reference_accuracy_series"].reindex(res["reference_error_series"].index).expanding().mean().values,
        }
    )

    st.plotly_chart(
        build_error_series_figure(
            error_df,
            "Пошаговая ошибка (%)",
            "Пошаговая ошибка по годам",
            PLOT_COLORS["forecast"],
            "Ошибка, %",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        build_error_series_figure(
            error_df,
            "Накопленная MAPE (%)",
            "Накопленная средняя ошибка MAPE",
            PLOT_COLORS["history"],
            "MAPE, %",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        build_error_series_figure(
            error_df,
            "Точность (%)",
            "Точность прогноза по годам",
            PLOT_COLORS["actual"],
            "Точность, %",
        ),
        use_container_width=True,
    )
    with st.expander("Таблица значений ошибок"):
        st.dataframe(
            error_df.set_index("Год").style.format(
                {
                    "Пошаговая ошибка (%)": "{:.2f}",
                    "Точность (%)": "{:.2f}",
                    "Накопленная MAPE (%)": "{:.2f}",
                    "Накопленная средняя точность (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    with st.expander("Раздельные ошибки к факту и к ООН"):
        table_parts = []
        if has_actual_errors:
            table_parts.append(
                pd.DataFrame(
                    {
                        "Год": res["error_series"].index,
                        "Ошибка к факту (%)": res["error_series"].values,
                        "Точность к факту (%)": res["accuracy_series"].reindex(res["error_series"].index).values,
                        "Накопленная MAPE к факту (%)": res["mape_series"].values,
                    }
                ).set_index("Год")
            )
        if has_un_errors:
            table_parts.append(
                pd.DataFrame(
                    {
                        "Год": res["un_error_series"].index,
                        "Ошибка к ООН (%)": res["un_error_series"].values,
                        "Точность к ООН (%)": res["un_accuracy_series"].reindex(res["un_error_series"].index).values,
                        "Накопленная MAPE к ООН (%)": res["un_mape_series"].values,
                    }
                ).set_index("Год")
            )
        if table_parts:
            st.dataframe(pd.concat(table_parts, axis=1).sort_index().style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("Раздельные ряды ошибок недоступны для выбранного периода.")

def get_outputs_dir() -> str:
    """Возвращает путь к каталогу сохранённых результатов приложения"""
    return os.path.join(os.path.dirname(__file__), "outputs")

def list_saved_experiments() -> list[tuple[str, str]]:
    """Ищет сохранённые эксперименты по файлам конфигурации в каталоге outputs"""
    output_dir = get_outputs_dir()
    if not os.path.exists(output_dir):
        return []
    experiments = []
    for filename in glob.glob(os.path.join(output_dir, "*_config.json")):
        base = filename.replace("_config.json", "")
        experiments.append((os.path.basename(base), base))
    return sorted(experiments, reverse=True)

def params_from_config(config: ExperimentConfig) -> Dict[str, Any]:
    """Преобразует объект конфигурации в словарь параметров для состояния интерфейса"""
    data = config.dict()
    data["end_forecast"] = int(config.t_start) + int(config.horizon) - 1
    return data

def render_experiments_tab(res: Dict[str, Any]) -> None:
    """Даёт возможность сохранить текущий запуск и повторить ранее сохранённый эксперимент"""
    render_info_card(
        "Сохранённые эксперименты",
        "Сохранённый запуск содержит параметры модели. При загрузке комплекс заново выполняет расчёт с теми же настройками и подставляет результат в интерфейс.",
    )

    col_save, col_load = st.columns([0.9, 1.1])
    with col_save:
        st.markdown("**Сохранить текущий расчёт**")
        st.caption("В каталог outputs записываются конфигурация, метрики, прогноз и итоговая таблица.")
        if st.button("Сохранить результаты", use_container_width=True):
            try:
                path = save_experiment(res)
                st.success(f"Результаты сохранены: `{path}`")
            except Exception as exc:
                st.error(f"Не удалось сохранить результаты: {exc}")

    with col_load:
        st.markdown("**Повторить сохранённый расчёт**")
        saved = list_saved_experiments()
        if not saved:
            st.info("Сохранённые эксперименты пока не найдены.")
            return

        exp_names = [item[0] for item in saved]
        selected = st.selectbox("Сохранённый эксперимент", exp_names)
        base_path = dict(saved)[selected]
        config_path = base_path + "_config.json"

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config_dict = json.load(file)
            saved_config = ExperimentConfig(**config_dict)
        except Exception as exc:
            st.error(f"Не удалось прочитать конфигурацию эксперимента: {exc}")
            return

        st.markdown(
            f"""
            <div class="experiment-card">
                <p><b>Страна:</b> {saved_config.country}</p>
                <p><b>Метод:</b> {METHOD_LABELS.get(saved_config.method, saved_config.method)}</p>
                <p><b>Период прогноза:</b> {saved_config.t_start}-{saved_config.t_start + saved_config.horizon - 1}</p>
                <p><b>Окна:</b> аппроксимация {saved_config.window_approx}, экстраполяция {saved_config.window_extrap}</p>
                <p><b>Сценарий ООН:</b> {variant_label(saved_config.un_forecast_variant)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Выполнить расчёт с этими параметрами", type="primary", use_container_width=True):
            result = run_experiment(saved_config)
            if result is not None:
                st.session_state.result = result
                st.session_state.params = params_from_config(saved_config)
                st.success("Эксперимент загружен и пересчитан. Результаты обновлены в интерфейсе.")
                st.rerun()

def calculate_window_summary(res: Dict[str, Any]) -> pd.DataFrame:
    """Формирует сводную таблицу по каждому окну аппроксимации и экстраполяции"""
    summary = []
    for index, item in enumerate(res["results"]):
        actual_pop = item["actual"]["total_actual"] if "total_actual" in item["actual"].columns else None
        forecast_pop = item["forecast"]
        mape_pop = None
        acc_pop = None
        if actual_pop is not None and not actual_pop.empty:
            common = actual_pop.index.intersection(forecast_pop.index)
            if not common.empty:
                mape_pop = np.mean(np.abs(actual_pop.loc[common] - forecast_pop.loc[common]) / actual_pop.loc[common].abs() * 100)
                acc_pop = 100 - mape_pop

        mape_inf = None
        if "inflow" in item["actual"].columns and item.get("inflow_forecast") is not None:
            actual_inf = item["actual"]["inflow"]
            forecast_inf = item["inflow_forecast"]
            common = actual_inf.index.intersection(forecast_inf.index)
            if not common.empty:
                mape_inf = np.mean(np.abs(actual_inf.loc[common] - forecast_inf.loc[common]) / actual_inf.loc[common].abs() * 100)

        mape_out = None
        if "outflow" in item["actual"].columns and item.get("outflow_forecast") is not None:
            actual_out = item["actual"]["outflow"]
            forecast_out = item["outflow_forecast"]
            common = actual_out.index.intersection(forecast_out.index)
            if not common.empty:
                mape_out = np.mean(np.abs(actual_out.loc[common] - forecast_out.loc[common]) / actual_out.loc[common].abs() * 100)

        mape_pop_un = None
        acc_pop_un = None
        if res.get("combined_un_forecast") is not None:
            common = forecast_pop.index.intersection(res["combined_un_forecast"].index)
            if not common.empty:
                denom = res["combined_un_forecast"].loc[common].replace(0, np.nan).abs()
                mape_pop_un = np.mean(np.abs(res["combined_un_forecast"].loc[common] - forecast_pop.loc[common]) / denom * 100)
                acc_pop_un = 100 - mape_pop_un

        summary.append(
            {
                "Окно": index + 1,
                "Период аппроксимации": f"{item['approx_start']}-{item['approx_end']}",
                "Период прогноза": f"{item['forecast_start']}-{item['forecast_end']}",
                "MAPE населения (%)": round(mape_pop, 2) if mape_pop is not None else "-",
                "Точность населения (%)": round(acc_pop, 2) if acc_pop is not None else "-",
                "MAPE притока (%)": round(mape_inf, 2) if mape_inf is not None else "-",
                "MAPE оттока (%)": round(mape_out, 2) if mape_out is not None else "-",
                "MAPE населения к ООН (%)": round(mape_pop_un, 2) if mape_pop_un is not None else "-",
                "Точность населения к ООН (%)": round(acc_pop_un, 2) if acc_pop_un is not None else "-",
            }
        )
    return pd.DataFrame(summary)

def render_window_details(res: Dict[str, Any]) -> None:
    """Показывает подробные таблицы факта и прогноза для каждого окна эксперимента"""
    with st.expander("Показать таблицы по отдельным окнам"):
        for index, item in enumerate(res["results"]):
            st.markdown(
                f"**Окно {index + 1}: {item['approx_start']}-{item['approx_end']} / {item['forecast_start']}-{item['forecast_end']}**"
            )
            col_actual, col_forecast = st.columns(2)
            with col_actual:
                st.write("Фактические значения")
                st.dataframe(item["actual"].style.format("{:,.0f}"), use_container_width=True)
            with col_forecast:
                st.write("Прогнозные значения")
                forecast_dict = item.get("forecast_dict", {})
                df_fc = pd.DataFrame(index=item["forecast"].index)
                df_fc["Население"] = item["forecast"]
                if "inflow_forecast" in forecast_dict:
                    df_fc["Приток"] = forecast_dict["inflow_forecast"]
                if "outflow_forecast" in forecast_dict:
                    df_fc["Отток"] = forecast_dict["outflow_forecast"]
                if res.get("combined_un_forecast") is not None:
                    df_fc["Население ООН"] = res["combined_un_forecast"].reindex(df_fc.index)
                    denom = df_fc["Население ООН"].replace(0, np.nan).abs()
                    df_fc["Ошибка к ООН населения (%)"] = (df_fc["Население"] - df_fc["Население ООН"]).abs() / denom * 100
                    df_fc["Точность к ООН населения (%)"] = 100 - df_fc["Ошибка к ООН населения (%)"]
                for key in [
                    "r_inflow_forecast",
                    "r_outflow_forecast",
                    "inflow_int_forecast",
                    "outflow_int_forecast",
                    "alpha_inflow_forecast",
                    "alpha_outflow_forecast",
                ]:
                    if key in forecast_dict and forecast_dict[key] is not None:
                        df_fc[key] = forecast_dict[key]
                fmt_detail = {col: "{:,.0f}" for col in df_fc.columns if not col.endswith("(%)")}
                fmt_detail.update({col: "{:.2f}" for col in df_fc.columns if col.endswith("(%)")})
                st.dataframe(df_fc.style.format(fmt_detail), use_container_width=True)

def render_windows_tab(res: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Отображает сводку по прогнозным окнам и позволяет сохранить их в Excel"""
    render_info_card(
        "Детальные результаты по окнам",
        "В этом разделе видно, как общий горизонт разбит на отдельные интервалы аппроксимации и экстраполяции. Это полезно для ретроспективного анализа устойчивости метода.",
    )

    if not res.get("results"):
        st.info("Детальные данные по окнам отсутствуют.")
        return

    df_summary = calculate_window_summary(res)
    st.dataframe(df_summary, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_summary["Окно"],
            y=pd.to_numeric(df_summary["MAPE населения (%)"], errors="coerce"),
            mode="lines+markers",
            name="MAPE населения к факту",
            line=dict(color=PLOT_COLORS["history"], width=2.2),
        )
    )
    if "MAPE населения к ООН (%)" in df_summary.columns:
        fig.add_trace(
            go.Scatter(
                x=df_summary["Окно"],
                y=pd.to_numeric(df_summary["MAPE населения к ООН (%)"], errors="coerce"),
                mode="lines+markers",
                name="MAPE населения к ООН",
                line=dict(color=PLOT_COLORS["un"], width=2.2, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Номер окна",
        yaxis_title="MAPE, %",
        hovermode="x unified",
        font=dict(color=CHART_AXIS),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=18, r=18, t=35, b=32),
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)
    st.plotly_chart(fig, use_container_width=True)

    render_window_details(res)

    if st.button("Сохранить результаты окон в Excel", use_container_width=True):
        full_data = load_inflow_outflow(params["country"], int(params["migration_policy"]))
        if full_data is None:
            st.error("Не удалось загрузить полный набор данных для сохранения.")
            return
        full_data = full_data.set_index("year")
        try:
            path = save_backtest(
                results=res["results"],
                full_data=full_data,
                backtest_type="inflow",
                country=params["country"],
                method=params["method"],
                window_approx=int(params["window_approx"]),
                window_extrap=int(params["window_extrap"]),
                migration_policy=int(params["migration_policy"]),
                approx_method=params["approx_method"],
                use_moving_window=bool(params["use_moving_window"]),
                un_forecast=res.get("un_forecast_data"),
                un_variant=res.get("un_forecast_variant"),
            )
            st.success(f"Файл сохранён: `{path}`")
        except Exception as exc:
            st.error(f"Не удалось сохранить Excel-файл: {exc}")

@st.cache_data(show_spinner=False, max_entries=64)
def cached_run_rolling_forecast(config_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет расчёт с кэшированием, чтобы повторные запуски с теми же параметрами не пересчитывались заново"""
    return run_rolling_forecast(ExperimentConfig(**config_payload))

def run_experiment(config: ExperimentConfig) -> Optional[Dict[str, Any]]:
    """Запускает вычислительный эксперимент и возвращает результат для сохранения в состоянии сессии"""
    config_payload = config.dict()
    with st.spinner("Выполняется расчёт прогноза, метрик и сравнительных рядов..."):
        try:
            result = cached_run_rolling_forecast(config_payload)
            st.success(f"Прогноз построен. Количество окон: {len(result['results'])}.")
            return result
        except Exception as exc:
            st.error(f"Ошибка при выполнении прогноза: {exc}")
            return None

def main() -> None:
    """Точка входа Streamlit-приложения"""
    inject_style()
    st.title("Прогнозирование численности населения")
    st.caption("Программный комплекс для прогнозирования численности населения")

    if "result" not in st.session_state:
        st.session_state.result = None
    if "params" not in st.session_state:
        st.session_state.params = {}

    config, params, run_button = build_sidebar_config()
    if run_button and config is not None:
        result = run_experiment(config)
        if result is not None:
            st.session_state.result = result
            st.session_state.params = params

    if st.session_state.result is None:
        render_start_panel()
        render_note(
            "Настройте параметры в боковой панели и запустите расчёт. <br>После выполнения стартовая панель будет заменена сводкой по построенному прогнозу.",
            kind="status",
        )
        return

    res = st.session_state.result
    current_params = st.session_state.params or params
    render_current_run_panel(res, current_params)
    tabs = st.tabs(["Прогноз", "Ошибки", "Эксперименты", "Детали по окнам"])
    with tabs[0]:
        render_forecast_tab(res)
    with tabs[1]:
        render_errors_tab(res)
    with tabs[2]:
        render_experiments_tab(res)
    with tabs[3]:
        render_windows_tab(res, current_params)

if __name__ == "__main__":
    main()
