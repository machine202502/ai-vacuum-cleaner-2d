"""
Отрисовка карты посещений: ячейки с 1 посещением — зелёный, 2–3 — жёлтый, больше — красный.
Рисуются полупрозрачно поверх мира.
"""
from __future__ import annotations

import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygame import Surface

# По числу посещений (RGB); при отрисовке добавляется альфа
COLOR_1 = (60, 220, 80)    # 1 посещение — зелёный
COLOR_2_3 = (220, 220, 60)  # 2–3 — жёлтый
COLOR_4_PLUS = (220, 60, 60)  # 4+ — красный
VISIT_MAP_ALPHA = 110  # 0..255, прозрачность наложения


def _color_for_count(count: int) -> tuple[int, int, int]:
    if count <= 0:
        return (0, 0, 0)
    if count == 1:
        return COLOR_1
    if count <= 3:
        return COLOR_2_3
    return COLOR_4_PLUS


def draw_visit_map(surface: "Surface", draw_data: dict) -> None:
    """
    Рисует посещённые ячейки полупрозрачно на поверхности.
    draw_data — результат get_draw_data() из VisitMap: left_px, top_px, cell_px, counts, reachable.
    """
    if not draw_data:
        return
    left_px = draw_data["left_px"]
    top_px = draw_data["top_px"]
    cell_px = draw_data.get("cell_px", 2)
    counts = draw_data.get("counts", {})
    reachable = draw_data.get("reachable", set())
    if not reachable:
        return
    max_i = max(i for i, _ in reachable)
    max_j = max(j for _, j in reachable)
    w = (max_i + 1) * cell_px
    h = (max_j + 1) * cell_px
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    for (i, j), count in counts.items():
        if count <= 0:
            continue
        r, g, b = _color_for_count(count)
        layer.fill((r, g, b, VISIT_MAP_ALPHA), (i * cell_px, j * cell_px, cell_px, cell_px))
    surface.blit(layer, (left_px, top_px))
