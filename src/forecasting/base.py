"""Базовые интерфейсы методов прогнозирования временных рядов"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Optional
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype

class ForecastingMethod(ABC):
    """Базовый интерфейс для всех методов прогнозирования в проекте"""
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        """Инициализирует объект и сохраняет параметры, которые используются при дальнейшем расчёте"""
        self.params = params or {}
        self.fitted_series = None
        self.last_year = None

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Подготавливает внутренние ряды метода на основе обучающего фрагмента данных"""
        self.fitted_series = series
        idx = series.index
        if is_datetime64_any_dtype(idx):
            self.last_year = idx[-1].year
        elif is_integer_dtype(idx):
            self.last_year = idx[-1]
        else:
            try:
                self.last_year = int(idx[-1])
            except (ValueError, TypeError):
                self.last_year = len(series) - 1

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Строит прогноз на заданное число шагов вперёд"""
        pass
