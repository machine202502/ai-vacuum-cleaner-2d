"""
Среда для обучения: reset(), step(action) → obs, reward, done, info.
Без pygame/окна; использует те же комнаты, контроллер и карту посещений.

obs (6 значений):
  [ir_forward_norm, ir_p30_norm, ir_m30_norm, sin_angle, cos_angle, encoder_delta_norm]
  encoder_delta_norm — расстояние за последний шаг / max за шаг (0 = стоим у стены).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from free_space import FreeSpaceMap
from room_loader import load_room, load_room_from_file, ROOMS_DIR
from units import meters_to_pixels
from visit_map import VisitMap

from agent_controller import AgentConfig, AgentController, update_visits

# Совпадает с game.py
WORLD_ORIGIN = 8000
SCALE = meters_to_pixels(1.0)
FPS = 60
EPISODE_DURATION_SEC = 90
MAX_STEPS_DEFAULT = EPISODE_DURATION_SEC * FPS  # 5400
IR_MAX_M = 3.0

# Таблица флагов для 8 дискретных действий
# (move_forward, move_backward, turn_left, turn_right)
ACTION_FLAGS: list[tuple[bool, bool, bool, bool]] = [
    (True,  False, False, False),  # 0: вперёд
    (False, True,  False, False),  # 1: назад
    (False, False, True,  False),  # 2: влево
    (False, False, False, True),   # 3: вправо
    (True,  False, True,  False),  # 4: вперёд + влево
    (True,  False, False, True),   # 5: вперёд + вправо
    (False, True,  True,  False),  # 6: назад + влево
    (False, True,  False, True),   # 7: назад + вправо
]
N_ACTIONS = len(ACTION_FLAGS)  # 8

# Награда за ячейку в зависимости от счётчика посещений после инкремента
REWARD_VISIT_1 = 1.0
REWARD_VISIT_2 = -0.1          # небольшой штраф за повтор
REWARD_VISIT_3 = -0.2
REWARD_VISIT_4 = -0.3
REWARD_VISIT_5_PLUS_DELTA = -0.1

# Сбалансированные штрафы
STEP_PENALTY      = 0.1        # фоновый штраф за время (поощряет быстроту)
COLLISION_PENALTY = 1.0        # упёрся в стену (не должно быть слишком огромным)
IDLE_PENALTY      = 0.2        # поворот на месте без движения вперёд
# Застревание: если encoder==0 дольше STUCK_STEPS шагов подряд — доп. штраф
STUCK_STEPS       = 120        # 2 секунды при 60 FPS
STUCK_PENALTY     = 2.0        # штраф за долгий простой
MAX_STUCK_STEPS   = 300        # терминация эпизода, если застрял на 5 секунд

# Бонусы за поведение (сбор по прямой, сбор рядом с уже убранным; складываются)
STRAIGHT_BONUS_PER_CELL = 0.05   # доп. награда за каждую собранную ячейку при движении строго прямо
NEAR_COLLECTED_BONUS    = 0.05   # доп. награда за сбор в непосредственной близости от уже убранной зоны


def reward_for_visit_count(count: int) -> float:
    """Награда за то, что ячейка имеет данное число посещений (после инкремента)."""
    if count <= 0:
        return 0.0
    if count == 1:
        return REWARD_VISIT_1
    if count == 2:
        return REWARD_VISIT_2
    if count == 3:
        return REWARD_VISIT_3
    if count == 4:
        return REWARD_VISIT_4
    return REWARD_VISIT_4 + (count - 4) * REWARD_VISIT_5_PLUS_DELTA


def _room_data_from_room(room, scale: float) -> dict:
    inv = 1.0 / scale
    walls = []
    for w in room.walls:
        r = w.rect
        walls.append([r.x * inv, r.y * inv, r.w * inv, r.h * inv])
    zones = []
    for z in room.zones:
        r = z.rect
        zones.append({
            "name": z.name,
            "rect": [r.x * inv, r.y * inv, r.w * inv, r.h * inv],
            "color": list(z.color),
        })
    return {
        "agent": [room.agent.x * inv, room.agent.y * inv],
        "walls": walls,
        "zones": zones,
    }


def _room_bounds(room_data: dict) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for wall in room_data.get("walls", []):
        x, y, w, h = wall[0], wall[1], wall[2], wall[3]
        xs.extend((x, x + w))
        ys.extend((y, y + h))
    for z in room_data.get("zones", []):
        r = z["rect"]
        x, y, w, h = r[0], r[1], r[2], r[3]
        xs.extend((x, x + w))
        ys.extend((y, y + h))
    agent = room_data.get("agent")
    if agent:
        xs.append(agent[0])
        ys.append(agent[1])
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _compute_free_space(
    room, agent_x: float, agent_y: float, scale: float, origin: int
) -> FreeSpaceMap:
    """Запустить BFS для комнаты и вернуть заполненный FreeSpaceMap."""
    room_data = _room_data_from_room(room, scale)
    room_data["agent"] = [agent_x / scale, agent_y / scale]
    bounds = _room_bounds(room_data)
    fsm = FreeSpaceMap()
    if bounds:
        fsm.calculate(room_data, bounds, scale, origin, origin)
    return fsm


def _circle_hits_rect(cx: float, cy: float, radius: float, r) -> bool:
    """Круг (cx, cy, radius) пересекает прямоугольник r (x, y, w, h)."""
    left = r.x
    top = r.y
    right = r.x + r.w
    bottom = r.y + r.h
    px = max(left, min(cx, right))
    py = max(top, min(cy, bottom))
    return (cx - px) ** 2 + (cy - py) ** 2 <= radius**2


def _circle_hits_any_wall(cx: float, cy: float, radius_px: float, room) -> bool:
    """Круг в координатах комнаты касается любой стены."""
    for w in room.walls:
        if _circle_hits_rect(cx, cy, radius_px, w.rect):
            return True
    return False


def _random_agent_position_in_free_space(
    room, visit_map: VisitMap, body_radius_px: float, sample_k: int = 200
) -> tuple[float, float]:
    """Случайная достижимая точка, где круг агента не пересекает стены. Иначе — старт комнаты."""
    if not visit_map.has_map:
        return (room.agent.x, room.agent.y)
    cell_px = visit_map.cell_px
    left_px = visit_map.left_px
    top_px = visit_map.top_px
    for (i, j) in visit_map.sample_reachable(sample_k):
        world_x = left_px + (i + 0.5) * cell_px
        world_y = top_px + (j + 0.5) * cell_px
        room_x = world_x - WORLD_ORIGIN
        room_y = world_y - WORLD_ORIGIN
        if not _circle_hits_any_wall(room_x, room_y, body_radius_px, room):
            return (room_x, room_y)
    return (room.agent.x, room.agent.y)


class SimpleAgent:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.angle = 0.0


class VacuumEnv:
    """
    Среда для RL: 8 дискретных действий (движение и вращение независимы).
      0: вперёд          4: вперёд + влево
      1: назад           5: вперёд + вправо
      2: влево           6: назад + влево
      3: вправо          7: назад + вправо
    obs (6 значений):
      [ir_fwd, ir_p30, ir_m30, sin_angle, cos_angle, encoder_delta]
      encoder_delta — расстояние за последний шаг (норм.), 0 = стоим у стены.
    """

    OBS_DIM = 6  # 3 IR + sin/cos угла + encoder_delta

    def __init__(
        self,
        room_name: str = "apartment_1",
        room_path: Path | str | None = None,
        max_steps: int = MAX_STEPS_DEFAULT,
        fps: float = FPS,
    ) -> None:
        if room_path is not None:
            self._room = load_room_from_file(Path(room_path))
        else:
            self._room = load_room(room_name)
        self._max_steps = max_steps
        self._fps = fps
        self._config = AgentConfig()
        self._body_radius_px = meters_to_pixels(self._config.radius)

        # Encoder delta: нормализуем по максимуму за один шаг (скорость вперёд / fps)
        self._max_step_dist = max(1e-9, self._config.speed / fps)

        # BFS вычисляется один раз в __init__ — планировка комнаты не меняется между эпизодами
        self._cached_free_space_map: FreeSpaceMap = _compute_free_space(
            self._room, self._room.agent.x, self._room.agent.y, SCALE, WORLD_ORIGIN
        )
        _tmp_vm = VisitMap()
        _tmp_vm.init_from_free_space(self._cached_free_space_map)
        self._total_cells_cached: int = _tmp_vm.total_cells

        # Runtime-состояние, инициализируется в reset()
        self._controller: AgentController | None = None
        self._visit_map: VisitMap | None = None
        self._agent: SimpleAgent | None = None
        self._encoder_total = 0.0
        self._encoder_delta = 0.0  # расстояние за последний шаг (0 = не двигались)
        self._step_count = 0
        self._stuck_counter = 0    # шагов подряд без движения (для STUCK_PENALTY)

    def _rebuild_cache(self) -> None:
        """Пересчитать BFS после смены комнаты."""
        self._cached_free_space_map = _compute_free_space(
            self._room, self._room.agent.x, self._room.agent.y, SCALE, WORLD_ORIGIN
        )
        _tmp_vm = VisitMap()
        _tmp_vm.init_from_free_space(self._cached_free_space_map)
        self._total_cells_cached = _tmp_vm.total_cells

    def reset(
        self,
        *,
        room_name: str | None = None,
        room_path: Path | str | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        """Сброс среды. Возвращает (obs, info)."""
        if room_path is not None:
            self._room = load_room_from_file(Path(room_path))
            self._rebuild_cache()
        elif room_name is not None:
            self._room = load_room(room_name)
            self._rebuild_cache()

        self._controller = AgentController(self._room, self._config, fps=self._fps)
        self._visit_map = VisitMap()
        self._visit_map.init_from_free_space(self._cached_free_space_map)
        self._agent = SimpleAgent(self._room.agent.x, self._room.agent.y)
        self._encoder_total = 0.0
        self._encoder_delta = 0.0
        self._step_count = 0
        self._stuck_counter = 0

        rx, ry = _random_agent_position_in_free_space(
            self._room, self._visit_map, self._body_radius_px
        )
        self._agent.x = rx
        self._agent.y = ry
        self._agent.angle = random.uniform(0.0, 360.0)

        obs = self._get_obs()
        info = {
            "step": 0,
            "encoder": 0.0,
            "visit_total": self._total_cells_cached,
        }
        return obs, info

    def _get_obs(self) -> list[float]:
        if self._controller is None or self._agent is None:
            return [0.0] * self.OBS_DIM
        s = self._controller.get_sensors(self._agent)
        ir_fwd = min(s.ir_forward / IR_MAX_M, 1.0)
        ir_p30 = min(s.ir_forward_p_30 / IR_MAX_M, 1.0)
        ir_m30 = min(s.ir_forward_m_30 / IR_MAX_M, 1.0)
        angle_rad = math.radians(self._agent.angle)
        sin_a = math.sin(angle_rad)
        cos_a = math.cos(angle_rad)
        enc_delta = min(self._encoder_delta / self._max_step_dist, 1.0)
        return [ir_fwd, ir_p30, ir_m30, sin_a, cos_a, enc_delta]

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        """
        action: 0..7 по таблице ACTION_FLAGS.
        Возвращает (obs, reward, done, info).
        """
        if self._controller is None or self._agent is None or self._visit_map is None:
            raise RuntimeError("Call reset() first")

        fwd, back, left, right = ACTION_FLAGS[action]

        prev_x, prev_y, prev_angle = self._agent.x, self._agent.y, self._agent.angle

        result = self._controller.apply_flags(
            self._agent,
            move_forward=fwd, move_backward=back,
            turn_left=left, turn_right=right,
        )
        self._encoder_delta = result.encoder  # расстояние за этот шаг (0 = не двигались)
        self._encoder_total += result.encoder

        reward = -STEP_PENALTY

        if result.encoder == 0:
            # Поворот на месте или удар в стену
            reward -= IDLE_PENALTY
            # Дополнительный штраф если реально пытались двигаться, но упёрлись
            if fwd or back:
                reward -= COLLISION_PENALTY
            # Счётчик застревания
            self._stuck_counter += 1
            if self._stuck_counter % STUCK_STEPS == 0:
                reward -= STUCK_PENALTY
        else:
            self._stuck_counter = 0  # двинулись — сброс счётчика
            if fwd:
                # Движение вперёд: засчитываем посещения через геометрию щётки
                update_visits(
                    self._agent, prev_x, prev_y, prev_angle,
                    self._visit_map, self._body_radius_px, WORLD_ORIGIN, WORLD_ORIGIN, True,
                )
                moving_straight = not left and not right
                for _i, _j, new_count in self._visit_map.pop_recent_increments():
                    r = reward_for_visit_count(new_count)
                    if moving_straight:
                        r += STRAIGHT_BONUS_PER_CELL
                    # Бонус за сбор рядом с уже убранной зоной (хотя бы один сосед посещён)
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        if self._visit_map.get_count(_i + di, _j + dj) >= 1:
                            r += NEAR_COLLECTED_BONUS
                            break
                    reward += r
            elif back:
                # Движение назад (включая назад+поворот): посещения не засчитываются,
                # штраф как за холостой поворот — агент не убирает двигаясь задним ходом
                reward -= IDLE_PENALTY

        self._step_count += 1
        done = self._step_count >= self._max_steps or self._stuck_counter >= MAX_STUCK_STEPS
        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "encoder": self._encoder_total,
            "visited": self._visit_map.visited_count,
            "total_cells": self._total_cells_cached,
            "reward_step": reward,
        }
        return obs, reward, done, info

    def set_max_steps(self, max_steps: int) -> None:
        """Обновить длину эпизода — для curriculum learning."""
        self._max_steps = max_steps

    @property
    def action_space_n(self) -> int:
        return N_ACTIONS  # 8

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM
