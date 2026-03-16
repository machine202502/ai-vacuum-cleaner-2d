"""
Карта посещений: дубликат сетки свободного пространства, у каждой ячейки — счётчик посещений.
Инициализируется по результату FreeSpaceMap; обновляется логикой из agent_controller.visit_logic.
"""
from __future__ import annotations

import random


class VisitMap:
    """Сетка ячеек с счётчиками посещений. Совпадает по размерам и индексам с FreeSpaceMap."""

    def __init__(self) -> None:
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._left_px: int = 0
        self._top_px: int = 0
        self._cell_px: int = 2
        self._reachable: set[tuple[int, int]] = set()
        self._counts: dict[tuple[int, int], int] = {}
        self._visited_count: int = 0
        self._recent_increments: list[tuple[int, int, int]] = []

    def init_from_free_space(self, free_space_map: object) -> None:
        """
        Инициализировать карту по результату FreeSpaceMap после calculate().
        Размеры и список достижимых ячеек копируются, счётчики обнуляются.
        """
        if not getattr(free_space_map, "has_result", False):
            self._reachable = set()
            self._counts = {}
            self._grid_w = self._grid_h = 0
            self._visited_count = 0
            self._recent_increments = []
            return
        self._grid_w = getattr(free_space_map, "_grid_w", 0)
        self._grid_h = getattr(free_space_map, "_grid_h", 0)
        self._left_px = getattr(free_space_map, "_left_px", 0)
        self._top_px = getattr(free_space_map, "_top_px", 0)
        self._cell_px = getattr(free_space_map, "CELL_PX", 2)
        reachable = getattr(free_space_map, "_reachable", None)
        self._reachable = set(reachable) if reachable else set()
        self._counts = {cell: 0 for cell in self._reachable}
        self._visited_count = 0
        self._recent_increments = []

    def increment(self, i: int, j: int) -> None:
        """Увеличить счётчик посещений ячейки (i, j) на 1."""
        if (i, j) not in self._reachable:
            return
        old = self._counts.get((i, j), 0)
        new = old + 1
        self._counts[(i, j)] = new
        if old == 0:
            self._visited_count += 1
        self._recent_increments.append((i, j, new))

    def pop_recent_increments(self) -> list[tuple[int, int, int]]:
        """Вернуть список (i, j, new_count) для всех ячеек, инкрементированных с последнего вызова, и очистить буфер."""
        result = self._recent_increments
        self._recent_increments = []
        return result

    def sample_reachable(self, k: int) -> list[tuple[int, int]]:
        """Вернуть до k случайных достижимых ячеек без полного копирования множества."""
        if not self._reachable:
            return []
        return random.sample(list(self._reachable), min(k, len(self._reachable)))

    def get_count(self, i: int, j: int) -> int:
        """Вернуть число посещений ячейки (i, j)."""
        return self._counts.get((i, j), 0)

    def is_reachable(self, i: int, j: int) -> bool:
        return (i, j) in self._reachable

    @property
    def has_map(self) -> bool:
        return self._grid_w > 0 and self._grid_h > 0 and len(self._reachable) > 0

    @property
    def visited_count(self) -> int:
        """Число ячеек, посещённых хотя бы один раз."""
        return self._visited_count

    @property
    def total_cells(self) -> int:
        """Общее число достижимых ячеек."""
        return len(self._reachable)

    @property
    def grid_w(self) -> int:
        return self._grid_w

    @property
    def grid_h(self) -> int:
        return self._grid_h

    @property
    def left_px(self) -> int:
        return self._left_px

    @property
    def top_px(self) -> int:
        return self._top_px

    @property
    def cell_px(self) -> int:
        return self._cell_px

    @property
    def reachable_cells(self) -> set[tuple[int, int]]:
        return set(self._reachable)

    def get_draw_data(self) -> dict | None:
        """Данные для отрисовки: left_px, top_px, cell_px, counts (dict (i,j)->count), reachable."""
        if not self.has_map:
            return None
        return {
            "left_px": self._left_px,
            "top_px": self._top_px,
            "cell_px": self._cell_px,
            "counts": dict(self._counts),
            "reachable": set(self._reachable),
        }
