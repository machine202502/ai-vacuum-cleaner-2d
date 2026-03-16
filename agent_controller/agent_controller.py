"""
Управление агентом: применение действий к агенту.

apply_flags(fwd, back, left, right) — основной метод: вращение и движение независимы.
  Поворот применяется первым, движение — с уже обновлённым углом.
  Если оба направления активны одновременно (fwd+back или left+right) — взаимно отменяются.

apply(action) — совместимость со старым кодом: принимает строку-действие (одно из 4).
resolve(control) — совместимость: выбирает одно действие из ControlState.

ApplyResult: encoder (м), ir_forward, ir_forward_p_30, ir_forward_m_30 (м, 0..IR_MAX_M).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pygame

from room_loader import Room
from units import meters_to_pixels, pixels_to_meters

from .agent_config import AgentConfig

Action = Literal["turn_left", "turn_right", "forward", "backward"]

IR_MAX_M = 3.0  # макс. дальность ИК-датчика, м


@dataclass(frozen=True)
class ApplyResult:
    """Результат одного шага: энкодер (м) и три ИК-датчика (м, 0..IR_MAX_M)."""
    encoder: float
    ir_forward: float
    ir_forward_p_30: float
    ir_forward_m_30: float


@dataclass(frozen=True)
class ControlState:
    """Входное состояние управления: ровно один True — выполнить это действие."""
    turn_left: bool = False
    turn_right: bool = False
    forward: bool = False
    backward: bool = False


def _circle_hits_rect(cx: float, cy: float, radius: float, r: pygame.Rect) -> bool:
    px = max(r.left, min(cx, r.right))
    py = max(r.top, min(cy, r.bottom))
    return (cx - px) ** 2 + (cy - py) ** 2 <= radius**2


def _segment_intersect(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> tuple[float, float] | None:
    """Пересечение отрезков (x1,y1)-(x2,y2) и (x3,y3)-(x4,y4). Возврат точки на первом отрезке или None."""
    denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(denom) < 1e-10:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
    s = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
    if 0 <= t <= 1 and 0 <= s <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _ray_distance_px(
    ox: float, oy: float, angle_deg: float, max_px: float, wall_rects: list[pygame.Rect]
) -> float:
    """Расстояние от (ox,oy) по лучу angle_deg до ближайшей стены, не более max_px."""
    rad = math.radians(angle_deg)
    ex = ox + max_px * math.cos(rad)
    ey = oy + max_px * math.sin(rad)
    best = max_px
    for r in wall_rects:
        for (ax, ay, bx, by) in [
            (r.left, r.top, r.right, r.top),
            (r.right, r.top, r.right, r.bottom),
            (r.right, r.bottom, r.left, r.bottom),
            (r.left, r.bottom, r.left, r.top),
        ]:
            pt = _segment_intersect(ox, oy, ex, ey, ax, ay, bx, by)
            if pt:
                d = math.hypot(pt[0] - ox, pt[1] - oy)
                if d < best:
                    best = d
    return best


class AgentController:
    """Контроллер агента: принимает комнату и конфиг, применяет действия к агенту."""

    def __init__(self, room: Room, config: AgentConfig, fps: float = 60.0) -> None:
        self.room = room
        self.config = config
        self.fps = fps
        self._wall_rects = [pygame.Rect(w.rect.x, w.rect.y, w.rect.w, w.rect.h) for w in room.walls]

    def resolve(self, control: ControlState) -> Action | None:
        """Если ровно один флаг True — возвращает соответствующее действие, иначе None."""
        flags = [control.turn_left, control.turn_right, control.forward, control.backward]
        if sum(flags) != 1:
            return None
        if control.turn_left:
            return "turn_left"
        if control.turn_right:
            return "turn_right"
        if control.forward:
            return "forward"
        if control.backward:
            return "backward"
        return None

    def get_sensors(self, agent: object) -> ApplyResult:
        """Текущие показания ИК-датчиков (encoder=0). Для отрисовки лучей."""
        return self._sensors(agent)

    def _sensors(self, agent: object) -> ApplyResult:
        """IR-датчики в текущей позиции/угле агента. encoder вызывающий задаёт сам."""
        x = getattr(agent, "x", 0.0)
        y = getattr(agent, "y", 0.0)
        angle = getattr(agent, "angle", 0.0)
        max_px = meters_to_pixels(IR_MAX_M)
        d_fwd = _ray_distance_px(x, y, angle, max_px, self._wall_rects)
        d_p30 = _ray_distance_px(x, y, angle + 30.0, max_px, self._wall_rects)
        d_m30 = _ray_distance_px(x, y, angle - 30.0, max_px, self._wall_rects)
        return ApplyResult(
            encoder=0.0,  # перезапишется при вызове из apply
            ir_forward=min(pixels_to_meters(d_fwd), IR_MAX_M),
            ir_forward_p_30=min(pixels_to_meters(d_p30), IR_MAX_M),
            ir_forward_m_30=min(pixels_to_meters(d_m30), IR_MAX_M),
        )

    def apply_flags(
        self,
        agent: object,
        move_forward: bool = False,
        move_backward: bool = False,
        turn_left: bool = False,
        turn_right: bool = False,
    ) -> ApplyResult:
        """
        Применяет вращение и движение независимо за один шаг.
        Если turn_left и turn_right оба True — взаимно отменяются (нет поворота).
        Если move_forward и move_backward оба True — взаимно отменяются (нет движения).
        Порядок: сначала поворот, затем движение с новым углом.
        """
        x = getattr(agent, "x", 0.0)
        y = getattr(agent, "y", 0.0)
        angle = getattr(agent, "angle", 0.0)
        cfg = self.config
        radius_px = meters_to_pixels(cfg.radius)

        # 1. Поворот (оба активны — отмена)
        if turn_left and not turn_right:
            agent.angle = (angle - cfg.turn_speed) % 360.0
        elif turn_right and not turn_left:
            agent.angle = (angle + cfg.turn_speed) % 360.0

        # 2. Движение с уже обновлённым углом (оба активны — отмена)
        if move_forward == move_backward:
            r = self._sensors(agent)
            return ApplyResult(encoder=0.0, ir_forward=r.ir_forward,
                               ir_forward_p_30=r.ir_forward_p_30, ir_forward_m_30=r.ir_forward_m_30)

        new_angle = agent.angle
        if move_forward:
            sign, move_speed_m_s = 1, cfg.speed
        else:
            sign, move_speed_m_s = -1, cfg.backward_speed

        delta_m = sign * move_speed_m_s / self.fps
        delta_px = meters_to_pixels(delta_m)
        rad = math.radians(new_angle)
        dx = delta_px * math.cos(rad)
        dy = delta_px * math.sin(rad)
        nx, ny = x + dx, y + dy

        def hits(cx: float, cy: float) -> bool:
            for r in self._wall_rects:
                if _circle_hits_rect(cx, cy, radius_px, r):
                    return True
            return False

        encoder_m = 0.0
        if not hits(nx, ny):
            agent.x = nx
            agent.y = ny
            encoder_m = abs(delta_m)
        else:
            mag = math.sqrt(dx * dx + dy * dy)
            if mag >= 1e-6:
                slide_x = abs(dx) / mag
                slide_y = abs(dy) / mag
                sx = x + dx * slide_x
                sy = y + dy * slide_y
                if not hits(sx, y):
                    agent.x = sx
                    agent.y = y
                    encoder_m = pixels_to_meters(abs(sx - x))
                elif not hits(x, sy):
                    agent.x = x
                    agent.y = sy
                    encoder_m = pixels_to_meters(abs(sy - y))

        r = self._sensors(agent)
        return ApplyResult(
            encoder=encoder_m,
            ir_forward=r.ir_forward,
            ir_forward_p_30=r.ir_forward_p_30,
            ir_forward_m_30=r.ir_forward_m_30,
        )

    def apply(self, agent: object, action: Action) -> ApplyResult:
        """Совместимость: применяет одно строковое действие через apply_flags."""
        return self.apply_flags(
            agent,
            move_forward=(action == "forward"),
            move_backward=(action == "backward"),
            turn_left=(action == "turn_left"),
            turn_right=(action == "turn_right"),
        )
