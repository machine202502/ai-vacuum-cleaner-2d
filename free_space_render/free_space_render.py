"""
Отрисовка рассчитанного свободного пространства: мигающие зелёные ячейки на поверхности.
"""
from __future__ import annotations

import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygame import Surface

BLINK_MS = 400
GREEN_BRIGHT = (80, 255, 100)
GREEN_DIM = (40, 200, 60)


def draw_free_space(
    surface: "Surface",
    draw_data: dict,
    ticks_ms: int = 0,
) -> None:
    """
    Рисует достижимые ячейки свободного пространства мигающим зелёным.
    draw_data — результат get_draw_data() из FreeSpaceMap: left_px, top_px, cells, cell_px.
    ticks_ms — время в мс (например pygame.time.get_ticks()) для мигания.
    """
    if not draw_data:
        return
    left_px = draw_data["left_px"]
    top_px = draw_data["top_px"]
    cells = draw_data["cells"]
    cell_px = draw_data.get("cell_px", 2)
    blink = (ticks_ms // BLINK_MS) % 2
    color = GREEN_BRIGHT if blink else GREEN_DIM
    for (i, j) in cells:
        px = left_px + i * cell_px
        py = top_px + j * cell_px
        pygame.draw.rect(surface, color, (px, py, cell_px, cell_px))
