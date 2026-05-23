"""Загрузка исторических демографических данных и подготовка балансовых рядов"""
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
POPULATION_CSV = "population_data.csv"

WPP_COLUMN_MAP = {
    "Variant": "variant",
    "Region, subregion, country or area *": "country",
    "Year": "year",
    "Total Population, as of 1 January (thousands)": "population",
    "Births (thousands)": "births",
    "Total Deaths (thousands)": "deaths",
    "Net Number of Migrants (thousands)": "net_migration",
}

CANONICAL_COLUMNS = ["country", "year", "population", "births", "deaths", "net_migration"]
NUMERIC_COLUMNS = ["population", "births", "deaths", "net_migration"]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Преобразует числовой столбец WPP к типу float с учетом пропусков и разделителей"""
    cleaned = (
        series.replace("…", pd.NA)
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_wpp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит таблицу WPP к внутреннему формату, который используется методами прогноза"""
    if all(column in df.columns for column in CANONICAL_COLUMNS):
        result = df.copy()
        if "variant" in result.columns:
            result = result[result["variant"].fillna("").astype(str).str.lower().eq("estimates")]
        for column in NUMERIC_COLUMNS + ["year"]:
            result[column] = _to_numeric(result[column])
        result = result.dropna(subset=["country", "year", "population"])
        result["year"] = result["year"].astype(int)
        return result[CANONICAL_COLUMNS].sort_values(["country", "year"]).reset_index(drop=True)

    missing = [column for column in WPP_COLUMN_MAP if column not in df.columns]
    if missing:
        raise ValueError(
            "Файл population_data.csv имеет неподдерживаемую структуру. "
            f"Не найдены столбцы: {missing}"
        )

    result = df.rename(columns=WPP_COLUMN_MAP).copy()
    result = result[result["variant"].fillna("").astype(str).str.lower().eq("estimates")]
    for column in NUMERIC_COLUMNS + ["year"]:
        result[column] = _to_numeric(result[column])

    result = result.dropna(subset=["country", "year", "population"])
    result["year"] = result["year"].astype(int)

    for column in NUMERIC_COLUMNS:
        result[column] = result[column] * 1000

    return result[CANONICAL_COLUMNS].sort_values(["country", "year"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _load_population_data(filename: str = POPULATION_CSV) -> pd.DataFrame:
    """Загружает локальный CSV-файл и нормализует его под балансовую модель"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}. Запустите launch.pyw для первичной инициализации данных."
        )
    return _normalise_wpp_frame(pd.read_csv(path, low_memory=False))


def get_series(country: str = "World", indicator: str = "population") -> Optional[pd.Series]:
    """Возвращает временной ряд выбранного показателя для указанной страны"""
    df = _load_population_data()
    df_country = df[df["country"] == country]
    if df_country.empty:
        return None
    if indicator not in df_country.columns:
        indicator = "population"
    return df_country.set_index("year")[indicator].sort_index()


def get_available_countries() -> list[str]:
    """Формирует отсортированный список стран и территорий из локального набора данных"""
    df = _load_population_data()
    return sorted(df["country"].dropna().unique().tolist())


def get_available_years_for_country(country: str) -> list[int]:
    """Возвращает годы наблюдений, доступные для выбранной страны"""
    df = _load_population_data()
    df_country = df[df["country"] == country]
    if df_country.empty:
        return []
    return sorted(df_country["year"].astype(int).unique().tolist())


def get_inflow_outflow(country: str, migration_policy: int = 2) -> Optional[pd.DataFrame]:
    """Рассчитывает ряды притока, оттока и фактическую численность на конец года"""
    df = _load_population_data()
    df_country = df[df["country"] == country].copy()
    if df_country.empty:
        return None

    required = ["population", "births", "deaths", "net_migration"]
    if not all(column in df_country.columns for column in required):
        return None

    if migration_policy == 0:
        df_country["inflow"] = df_country["births"] + df_country["net_migration"]
        df_country["outflow"] = df_country["deaths"]
    elif migration_policy == 1:
        df_country["inflow"] = df_country["births"] + df_country["net_migration"].clip(lower=0)
        df_country["outflow"] = df_country["deaths"] + (-df_country["net_migration"]).clip(lower=0)
    else:
        df_country["inflow"] = df_country["births"]
        df_country["outflow"] = df_country["deaths"] - df_country["net_migration"]

    df_country["total_actual"] = df_country["population"] + df_country["inflow"] - df_country["outflow"]

    return df_country[
        ["year", "population", "births", "deaths", "net_migration", "inflow", "outflow", "total_actual"]
    ].sort_values("year")
