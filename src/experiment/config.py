"""Конфигурационная модель вычислительного эксперимента"""
from pydantic import BaseModel
from typing import Literal, Optional

class ExperimentConfig(BaseModel):
    """Набор параметров, полностью описывающий один запуск вычислительного эксперимента"""
    country: str = "World"
    method: Literal[
        "inflow_outflow_direct", "integrated_growth_rate", "growth_rate_speed"
    ] = "inflow_outflow_direct"
    window_approx: int = 10
    window_extrap: int = 5
    t_start: int
    horizon: int = 5
    migration_policy: int = 2
    approx_method: Literal["auto", "linear", "power"] = "auto"
    use_moving_window: bool = False

    un_forecast_variant: Optional[str] = "Medium variant"
