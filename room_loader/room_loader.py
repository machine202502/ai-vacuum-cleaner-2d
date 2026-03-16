"""
Типы комнаты и загрузка из файлов rooms/{name}.json.
Поддержка units: "meters" — значения в JSON в метрах, при загрузке конвертируются в пиксели.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from units import meters_to_pixels

ROOMS_DIR = Path(__file__).resolve().parent.parent / "rooms"


@dataclass(frozen=True)
class Rect:
    """Прямоугольник: x, y — левый верхний угол, w, h — размеры."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Point:
    """Точка на плоскости."""
    x: int
    y: int


DEFAULT_WALL_COLOR = (70, 70, 70)


@dataclass(frozen=True)
class Zone:
    """Зона: прямоугольник с названием и цветом (комната, кухня и т.д.)."""
    name: str
    rect: Rect
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Wall:
    """Стена: прямоугольник и опциональный цвет. Если цвет не задан — DEFAULT_WALL_COLOR."""
    rect: Rect
    color: tuple[int, int, int] = DEFAULT_WALL_COLOR


@dataclass
class Room:
    """Комната: старт агента, зоны, стены. Координаты в пикселях, могут быть отрицательными."""
    agent: Point
    zones: list[Zone]
    walls: list[Wall]


def _to_px(v: float, scale: float) -> int:
    return int(round(v * scale))


def _parse_point(data: list, scale: float) -> Point:
    x, y = float(data[0]), float(data[1])
    return Point(_to_px(x, scale), _to_px(y, scale))


def _parse_rect(data: list, scale: float) -> Rect:
    x, y, w, h = float(data[0]), float(data[1]), float(data[2]), float(data[3])
    return Rect(
        _to_px(x, scale),
        _to_px(y, scale),
        _to_px(w, scale),
        _to_px(h, scale),
    )


def _parse_wall(data: list, scale: float) -> Wall:
    rect = _parse_rect(data[:4], scale)
    color = DEFAULT_WALL_COLOR
    if len(data) >= 7:
        color = (int(data[4]), int(data[5]), int(data[6]))
    return Wall(rect=rect, color=color)


def load_room_from_file(path: Path | str) -> Room:
    """Загружает комнату из JSON. При units: \"meters\" конвертирует в пиксели."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Room file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    use_meters = data.get("units") == "meters"
    scale = meters_to_pixels(1.0) if use_meters else 1.0

    agent = _parse_point(data.get("agent", [0, 0]), scale)
    walls = [_parse_wall(r, scale) for r in data.get("walls", [])]
    zones = []
    for z in data.get("zones", []):
        rect = _parse_rect(z["rect"], scale)
        color = tuple(int(z["color"][i]) for i in range(3))
        zones.append(Zone(name=z["name"], rect=rect, color=color))
    return Room(agent=agent, zones=zones, walls=walls)


def load_room(name: str) -> Room:
    """Загружает комнату по имени из каталога rooms: rooms/{name}.json."""
    return load_room_from_file(ROOMS_DIR / f"{name}.json")
