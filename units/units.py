"""
Перевод метров ↔ пиксели. Меняй PIXELS_PER_METER — от него зависят и рендер, и столкновения.
"""
PIXELS_PER_METER = 80


def meters_to_pixels(m: float) -> float:
    return m * PIXELS_PER_METER


def pixels_to_meters(p: float) -> float:
    """Пиксели → метры."""
    return p / PIXELS_PER_METER
