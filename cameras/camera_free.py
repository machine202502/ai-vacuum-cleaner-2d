"""
Свободная камера: зум колёсиком (к точке под курсором), панорамирование перетаскиванием мыши.
Камера не привязана к границам мира — можно отдалять и сдвигать за пределы комнаты.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraFree:
    """Состояние камеры: позиция в мировых координатах (левый верхний угла вида), зум."""
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    zoom_min: float = 0.1
    zoom_max: float = 5.0

    _panning: bool = False
    _pan_start: tuple[float, float] = (0.0, 0.0)
    _pan_camera_start: tuple[float, float] = (0.0, 0.0)

    def handle_event(self, event, viewport_size: tuple[int, int]) -> None:
        """Обработать событие pygame (MOUSEWHEEL, MOUSEBUTTONDOWN/UP, MOTION)."""
        import pygame
        vw, vh = viewport_size
        if event.type == pygame.MOUSEWHEEL:
            old_zoom = self.zoom
            self.zoom *= 1.15 if event.y > 0 else 1 / 1.15
            self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom))
            # Точка зума: из события (координаты во viewport) или текущая позиция мыши
            mx, my = getattr(event, "pos", None) or pygame.mouse.get_pos()
            self.x += mx * (1 / old_zoom - 1 / self.zoom)
            self.y += my * (1 / old_zoom - 1 / self.zoom)
        elif event.type == pygame.MOUSEBUTTONDOWN and (event.button == 1 or event.button == 2):
            self._panning = True
            self._pan_start = event.pos
            self._pan_camera_start = (self.x, self.y)
        elif event.type == pygame.MOUSEBUTTONUP and (event.button == 1 or event.button == 2):
            self._panning = False
        elif event.type == pygame.MOUSEMOTION and self._panning:
            dx = (self._pan_start[0] - event.pos[0]) / self.zoom
            dy = (self._pan_start[1] - event.pos[1]) / self.zoom
            self.x = self._pan_camera_start[0] + dx
            self.y = self._pan_camera_start[1] + dy

    def draw(
        self,
        world_surface,
        screen,
        viewport_size: tuple[int, int],
        bg_color: tuple[int, int, int],
    ) -> None:
        """Отрисовать мир на экран с учётом камеры. Мир может выходить за границы — за пределами будет bg_color."""
        import pygame
        vw, vh = viewport_size
        view_w = vw / self.zoom
        view_h = vh / self.zoom
        view_surf = pygame.Surface((max(1, int(view_w) + 2), max(1, int(view_h) + 2)))
        view_surf.fill(bg_color)
        view_surf.blit(world_surface, (-self.x, -self.y))
        scaled = pygame.transform.smoothscale(view_surf, (vw, vh))
        screen.fill(bg_color)
        screen.blit(scaled, (0, 0))
