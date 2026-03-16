"""
Отрисовка комнаты: зоны (цветные прямоугольники + подписи), затем стены.
Всё рисуется на surface из контекста. Зоны и текст — приглушённые, под стенами.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from room_loader import Room


@dataclass
class RenderContext:
    surface: pygame.surface.Surface
    wall_color: tuple[int, int, int] = (70, 70, 70)
    zone_text_color: tuple[int, int, int] = (100, 100, 108)
    zone_text_font_size: int = 14
    origin_x: int = 0
    origin_y: int = 0


def draw_room(room: Room, ctx: RenderContext) -> None:
    ox, oy = ctx.origin_x, ctx.origin_y
    for z in room.zones:
        r = z.rect
        pygame.draw.rect(ctx.surface, z.color, (ox + r.x, oy + r.y, r.w, r.h))
    font = pygame.font.SysFont("Arial", ctx.zone_text_font_size)
    for z in room.zones:
        r = z.rect
        cx = ox + r.x + r.w // 2
        cy = oy + r.y + r.h // 2
        text = font.render(z.name, True, ctx.zone_text_color)
        tr = text.get_rect(center=(cx, cy))
        ctx.surface.blit(text, tr)
    for w in room.walls:
        color = getattr(w, "color", ctx.wall_color)
        r = w.rect
        pygame.draw.rect(ctx.surface, color, (ox + r.x, oy + r.y, r.w, r.h))
