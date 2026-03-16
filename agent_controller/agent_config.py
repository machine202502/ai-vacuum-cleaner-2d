"""
Конфиг агента: размер (радиус), скорости. Радиус в метрах, скорости в м/с.
"""
from dataclasses import dataclass


@dataclass
class AgentConfig:
    radius: float = 0.2
    speed: float = 0.3
    backward_speed: float = 0.15
    turn_speed: float = 2.5
