"""
Карта свободного пространства комнаты: сетка 2×2 px, расчёт достижимых ячеек от агента (BFS).
Хранит карту занятости (стены) и результат обхода — множество достижимых ячеек и площадь.
"""
from __future__ import annotations

from collections import deque


class FreeSpaceMap:
    """Карта свободного пространства в границах комнаты. Строит сетку по стенам, считает достижимое от агента."""

    CELL_PX = 2  # размер ячейки карты в пикселях (2×2)

    def __init__(self) -> None:
        self._bounds: tuple[float, float, float, float] | None = None
        self._scale: float = 0.0
        self._origin_x: int = 0
        self._origin_y: int = 0
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._blocked: set[tuple[int, int]] | None = None  # занятые стенами ячейки (i, j)
        self._reachable: set[tuple[int, int]] | None = None  # достижимые от агента ячейки
        self._free_pixels: int = 0
        self._free_m2: float = 0.0
        self._left_px: int = 0
        self._top_px: int = 0

    @staticmethod
    def _overlaps_m(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    def calculate(
        self,
        room_data: dict,
        bounds: tuple[float, float, float, float],
        scale: float,
        origin_x: int,
        origin_y: int,
    ) -> bool:
        """
        Строит карту по стенам комнаты и вычисляет достижимое пространство от позиции агента (BFS).
        Возвращает True при успехе, False если агент вне сетки или в стене.
        """
        bx, by, bw, bh = bounds
        left_px = origin_x + int(bx * scale)
        top_px = origin_y + int(by * scale)
        w_px = max(1, int(bw * scale))
        h_px = max(1, int(bh * scale))
        grid_w = w_px // self.CELL_PX
        grid_h = h_px // self.CELL_PX
        if grid_w <= 0 or grid_h <= 0:
            self.clear()
            return False

        cell_size_m = self.CELL_PX / scale

        def cell_to_rect_m(i: int, j: int) -> tuple[float, float, float, float]:
            return (bx + i * cell_size_m, by + j * cell_size_m, cell_size_m, cell_size_m)

        blocked: set[tuple[int, int]] = set()
        for i in range(grid_w):
            for j in range(grid_h):
                cell_m = cell_to_rect_m(i, j)
                for wall in room_data.get("walls", []):
                    wm = (wall[0], wall[1], wall[2], wall[3])
                    if self._overlaps_m(cell_m, wm):
                        blocked.add((i, j))
                        break

        agent = room_data.get("agent", [0, 0])
        ax, ay = agent[0], agent[1]
        start_i = int((ax - bx) / cell_size_m)
        start_j = int((ay - by) / cell_size_m)
        if start_i < 0 or start_i >= grid_w or start_j < 0 or start_j >= grid_h:
            self.clear()
            return False
        if (start_i, start_j) in blocked:
            self.clear()
            return False

        reachable: set[tuple[int, int]] = set()
        q: deque[tuple[int, int]] = deque([(start_i, start_j)])
        reachable.add((start_i, start_j))
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < grid_w
                    and 0 <= nj < grid_h
                    and (ni, nj) not in blocked
                    and (ni, nj) not in reachable
                ):
                    reachable.add((ni, nj))
                    q.append((ni, nj))

        self._bounds = bounds
        self._scale = scale
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._grid_w = grid_w
        self._grid_h = grid_h
        self._blocked = blocked
        self._reachable = reachable
        self._free_pixels = len(reachable) * (self.CELL_PX * self.CELL_PX)
        self._free_m2 = len(reachable) * (cell_size_m * cell_size_m)
        self._left_px = left_px
        self._top_px = top_px
        return True

    def clear(self) -> None:
        """Сбросить результат расчёта (достижимые ячейки и статистику). Карта занятости тоже сбрасывается."""
        self._bounds = None
        self._blocked = None
        self._reachable = None
        self._free_pixels = 0
        self._free_m2 = 0.0

    @property
    def has_result(self) -> bool:
        return self._reachable is not None

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        return self._bounds

    @property
    def free_pixels(self) -> int:
        return self._free_pixels

    @property
    def free_m2(self) -> float:
        return self._free_m2

    def get_draw_data(self) -> dict | None:
        """Данные для отрисовки: left_px, top_px, cells, cell_px. None если расчёт не выполнялся."""
        if self._reachable is None:
            return None
        return {
            "left_px": self._left_px,
            "top_px": self._top_px,
            "cells": self._reachable,
            "cell_px": self.CELL_PX,
        }
