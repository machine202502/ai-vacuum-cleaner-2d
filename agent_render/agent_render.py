"""
Отрисовка агента (робот-пылесос вид сверху): круг (корпус), колёса (прямоугольники), тонкий прямоугольник (всасывание).
Принимает агента (x, y, angle) и контекст рисования.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass
class RenderContext:
    """Контекст для рисования агента. body_radius в пикселях; origin — сдвиг мировых координат на surface."""
    surface: pygame.surface.Surface
    body_radius: float
    body_color: tuple[int, int, int] = (50, 140, 230)
    wheel_color: tuple[int, int, int] = (40, 40, 40)
    suction_color: tuple[int, int, int] = (45, 120, 200)
    origin_x: int = 0
    origin_y: int = 0


def _make_agent_surface(ctx: RenderContext) -> pygame.Surface:
    """Рисует агента в кадре «нос вправо» (angle=0), центр круга в центре поверхности."""
    r = int(ctx.body_radius)
    suction_width = max(2, int(r * 0.33))
    suction_length = int(r * 1.1)
    wheel_width = max(2, int(r * 0.33))
    wheel_length = int(r * 0.78)
    wheel_offset = int(r * 0.56)
    suction_offset = max(2, int(r * 0.33))

    margin = max(r + suction_length, r + wheel_length // 2) + 4
    size = int(margin * 2)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size / 2, size / 2

    pygame.draw.circle(surf, ctx.body_color, (cx, cy), r)
    pygame.draw.circle(surf, (30, 100, 180), (cx, cy), r, 2)

    suction_rect = pygame.Rect(int(cx - suction_offset - suction_width), int(cy - suction_length / 2), suction_width, suction_length)
    pygame.draw.rect(surf, ctx.suction_color, suction_rect)
    pygame.draw.rect(surf, (30, 100, 180), suction_rect, 1)

    left_rect = pygame.Rect(int(cx - wheel_length / 2), int(cy - wheel_offset - wheel_width / 2), wheel_length, wheel_width)
    right_rect = pygame.Rect(int(cx - wheel_length / 2), int(cy + wheel_offset - wheel_width / 2), wheel_length, wheel_width)
    pygame.draw.rect(surf, ctx.wheel_color, left_rect)
    pygame.draw.rect(surf, (60, 60, 60), left_rect, 1)
    pygame.draw.rect(surf, ctx.wheel_color, right_rect)
    pygame.draw.rect(surf, (60, 60, 60), right_rect, 1)

    return surf


def draw_agent(agent: object, ctx: RenderContext) -> None:
    """Рисует агента на surface. У agent: x, y, angle (градусы)."""
    x = getattr(agent, "x", 0)
    y = getattr(agent, "y", 0)
    angle = getattr(agent, "angle", 0)

    surf = _make_agent_surface(ctx)
    rotated = pygame.transform.rotate(surf, -angle)
    rect = rotated.get_rect(center=(ctx.origin_x + int(x), ctx.origin_y + int(y)))
    ctx.surface.blit(rotated, rect)
