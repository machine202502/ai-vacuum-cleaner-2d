"""
Управление пылесосом обученной моделью (policy_final.pt).
То же, что game.py: комната, камера, карта посещений, ИК, encoder. Управление — по модели.
Поддерживает MLP и GRU-политику — конфиг читается автоматически из чекпоинта.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pygame
import pygame_gui
import torch

import agent_render
import room_render
from agent_controller import AgentConfig, AgentController, update_visits
from cameras import CameraFree
from free_space import FreeSpaceMap
from room_loader import ROOMS_DIR, load_room, load_room_from_file
from units import meters_to_pixels
from visit_map import VisitMap
import visit_map_render
from training.policy_net import PolicyNet

WORLD_SIZE = 16000
WORLD_ORIGIN = WORLD_SIZE // 2
SCALE = meters_to_pixels(1.0)
FPS = 60
INITIAL_W = 1000
INITIAL_H = 700

PANEL_W = 260
PANEL_BG = (38, 38, 44)
BG = (28, 28, 32)
TEXT_CLR = (200, 200, 208)
DEFAULT_ROOM_NAME = "apartment_1"

BTN_MARGIN = 10
BTN_H = 36
UI_GAP = 8

SPEED_OPTIONS = ["1x", "2x", "3x", "4x", "5x", "8x", "10x", "15x", "20x"]
VISIT_PCT_UPDATE_MS = 1000

# Нормализация obs — должна совпадать с vacuum_env.py
IR_MAX_M = 3.0
# Encoder total: теоретический максимум за 90-секундный эпизод (= 27 м).
ENCODER_NORM_SCALE = 90 * 0.3  # 90 сек × 0.3 м/с = 27.0 м
# Encoder delta: максимум за один шаг (forward speed / fps).
MAX_STEP_DIST = 0.3 / FPS       # 0.005 м
# Количество шагов в эпизоде (для нормализации time = step/max_steps).
MAX_STEPS_VIS = 90 * FPS        # 5400

ACTION_NAMES = (
    "forward", "backward", "turn_left", "turn_right",
    "forward+left", "forward+right", "backward+left", "backward+right",
)
ACTION_LABELS = (
    "вперёд", "назад", "влево", "вправо",
    "вперёд+влево", "вперёд+вправо", "назад+влево", "назад+вправо",
)
# (move_forward, move_backward, turn_left, turn_right)
_ACTION_FLAGS: list[tuple[bool, bool, bool, bool]] = [
    (True,  False, False, False),  # 0: вперёд
    (False, True,  False, False),  # 1: назад
    (False, False, True,  False),  # 2: влево
    (False, False, False, True),   # 3: вправо
    (True,  False, True,  False),  # 4: вперёд + влево
    (True,  False, False, True),   # 5: вперёд + вправо
    (False, True,  True,  False),  # 6: назад + влево
    (False, True,  False, True),   # 7: назад + вправо
]

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "training" / "checkpoints" / "last.pt"


def _room_data_from_room(room, scale: float) -> dict:
    inv = 1.0 / scale
    walls = []
    for w in room.walls:
        r = w.rect
        row = [r.x * inv, r.y * inv, r.w * inv, r.h * inv]
        if getattr(w, "color", None):
            row.extend(w.color[:3])
        walls.append(row)
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


def _update_free_space_and_visit_map(
    room, agent, free_space_map: FreeSpaceMap, visit_map: VisitMap
) -> None:
    room_data = _room_data_from_room(room, SCALE)
    room_data["agent"] = [agent.x / SCALE, agent.y / SCALE]
    bounds = _room_bounds(room_data)
    if bounds and free_space_map.calculate(room_data, bounds, SCALE, WORLD_ORIGIN, WORLD_ORIGIN):
        visit_map.init_from_free_space(free_space_map)


def _circle_hits_rect(cx: float, cy: float, radius: float, r) -> bool:
    left   = getattr(r, "left",   r.x)
    top    = getattr(r, "top",    r.y)
    right  = getattr(r, "right",  r.x + r.w)
    bottom = getattr(r, "bottom", r.y + r.h)
    px = max(left, min(cx, right))
    py = max(top,  min(cy, bottom))
    return (cx - px) ** 2 + (cy - py) ** 2 <= radius**2


def _circle_hits_any_wall(cx: float, cy: float, radius_px: float, room) -> bool:
    for w in room.walls:
        if _circle_hits_rect(cx, cy, radius_px, w.rect):
            return True
    return False


def _random_agent_position_in_free_space(room, visit_map: VisitMap, body_radius_px: float) -> tuple[float, float]:
    if not visit_map.has_map:
        return (room.agent.x, room.agent.y)
    cell_px = visit_map.cell_px
    left_px = visit_map.left_px
    top_px  = visit_map.top_px
    for (i, j) in visit_map.sample_reachable(200):
        world_x = left_px + (i + 0.5) * cell_px
        world_y = top_px  + (j + 0.5) * cell_px
        room_x = world_x - WORLD_ORIGIN
        room_y = world_y - WORLD_ORIGIN
        if not _circle_hits_any_wall(room_x, room_y, body_radius_px, room):
            return (room_x, room_y)
    return (room.agent.x, room.agent.y)


def _create_ui(manager: pygame_gui.UIManager, vw: int, vh: int) -> dict:
    bw = PANEL_W - 2 * BTN_MARGIN
    x, y = BTN_MARGIN, BTN_MARGIN

    h_btn  = 36
    h_line = 20
    h_room = 24
    h_drop = 32
    h_chk  = 26
    h_hint = 52

    btn_load = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(x, y, bw, h_btn),
        text="Загрузить комнату",
        manager=manager,
    )
    y += h_btn + UI_GAP

    label_room = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_room),
        text="(комната)",
        manager=manager,
    )
    y += h_room + UI_GAP

    label_speed = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Скорость мира:",
        manager=manager,
    )
    y += h_line + UI_GAP

    dropdown_speed = pygame_gui.elements.UIDropDownMenu(
        options_list=SPEED_OPTIONS,
        starting_option="1x",
        relative_rect=pygame.Rect(x, y, bw, h_drop),
        manager=manager,
    )
    y += h_drop + UI_GAP


    label_in_fwd = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[ir_fwd] — —",    manager=manager)
    y += h_line + UI_GAP
    label_in_p30 = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[ir_p30] — —",    manager=manager)
    y += h_line + UI_GAP
    label_in_m30 = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[ir_m30] — —",    manager=manager)
    y += h_line + UI_GAP
    label_in_sin   = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[sin_угол] — —",  manager=manager)
    y += h_line + UI_GAP
    label_in_cos   = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[cos_угол] — —",  manager=manager)
    y += h_line + UI_GAP
    label_in_enc   = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[enc] — —",        manager=manager)
    y += h_line + UI_GAP
    label_in_delta = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[delta] — —",      manager=manager)
    y += h_line + UI_GAP
    label_in_time  = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Вход[time] — —",       manager=manager)
    y += h_line + UI_GAP
    label_out_fwd      = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[вперёд] — —",       manager=manager)
    y += h_line + UI_GAP
    label_out_back     = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[назад] — —",        manager=manager)
    y += h_line + UI_GAP
    label_out_left     = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[влево] — —",        manager=manager)
    y += h_line + UI_GAP
    label_out_right    = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[вправо] — —",       manager=manager)
    y += h_line + UI_GAP
    label_out_fwd_left = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[вп+влево] — —",     manager=manager)
    y += h_line + UI_GAP
    label_out_fwd_right= pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[вп+вправо] — —",    manager=manager)
    y += h_line + UI_GAP
    label_out_back_left= pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[нз+влево] — —",     manager=manager)
    y += h_line + UI_GAP
    label_out_back_right=pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Выход[нз+вправо] — —",    manager=manager)
    y += h_line + UI_GAP
    label_action       = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(x, y, bw, h_line), text="Действие: —",              manager=manager)
    y += h_line + UI_GAP

    chk_visit_map = pygame_gui.elements.UICheckBox(
        relative_rect=pygame.Rect(x, y, bw, h_chk),
        text="Карта посещений",
        manager=manager,
        initial_state=True,
    )
    y += h_chk + UI_GAP

    label_visit_pct = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Посещено: —",
        manager=manager,
    )
    y += h_line + UI_GAP

    label_encoder = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Encoder: 0.00 м",
        manager=manager,
    )
    y += h_line + UI_GAP

    label_hint = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_hint),
        text="Управление: модель\nСКМ/колёсико — камера\nR — рестарт (новая позиция)",
        manager=manager,
    )

    return {
        "btn_load":         btn_load,
        "label_room":       label_room,
        "label_speed":      label_speed,
        "dropdown_speed":   dropdown_speed,
        "chk_visit_map":    chk_visit_map,
        "label_visit_pct":  label_visit_pct,
        "label_encoder":    label_encoder,
        "label_in_fwd":     label_in_fwd,
        "label_in_p30":     label_in_p30,
        "label_in_m30":     label_in_m30,
        "label_in_sin":     label_in_sin,
        "label_in_cos":     label_in_cos,
        "label_in_enc":     label_in_enc,
        "label_in_delta":   label_in_delta,
        "label_in_time":    label_in_time,
        "label_out_fwd":       label_out_fwd,
        "label_out_back":      label_out_back,
        "label_out_left":      label_out_left,
        "label_out_right":     label_out_right,
        "label_out_fwd_left":  label_out_fwd_left,
        "label_out_fwd_right": label_out_fwd_right,
        "label_out_back_left": label_out_back_left,
        "label_out_back_right":label_out_back_right,
        "label_action":        label_action,
        "label_hint":       label_hint,
    }


class Agent:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.angle = 0.0


def _load_policy(path: Path) -> tuple["PolicyNet", dict]:
    """Загружает модель, автоматически определяя архитектуру из чекпоинта.
    Возвращает (policy, train_config) где train_config содержит max_steps и т.п."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = data.get("config", {})
    policy = PolicyNet(
        obs_dim=cfg.get("obs_dim", 8),
        n_actions=cfg.get("n_actions", 8),
        hidden_size=cfg.get("hidden_size", 256),
        use_gru=cfg.get("use_gru", True),
        encoder_layers=cfg.get("encoder_layers", 6),
    )
    policy.load_state_dict(data["policy"], strict=False)
    policy.eval()
    arch = "GRU" if policy.use_gru else "MLP"
    train_config = data.get("train_config", {})
    max_steps = train_config.get("max_steps", 0)
    max_steps_str = f", max_steps={max_steps}" if max_steps else " (max_steps не сохранён, используется fallback)"
    print(f"Модель загружена: {arch}, hidden={policy.hidden_size}, params={policy.param_count():,}{max_steps_str}")
    return policy, train_config


def _resolve_policy_path(explicit: Path) -> Path:
    """Возвращает путь к модели: явно указанный → last.pt → history/policy_ep_* → старый путь."""
    if explicit.is_file():
        return explicit
    checkpoints_dir = explicit.parent
    # last.pt — всегда самый свежий
    last = checkpoints_dir / "last.pt"
    if last.is_file():
        print(f"Загружаю последний чекпоинт: {last.name}")
        return last
    # Fallback: history/policy_ep_*.pt
    candidates = sorted((checkpoints_dir / "history").glob("policy_ep_*.pt"))
    if candidates:
        latest = candidates[-1]
        print(f"last.pt не найден, загружаю из history: {latest.name}")
        return latest
    return explicit  # вернём оригинальный путь чтобы выдать понятную ошибку


def main() -> None:
    policy_path = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).suffix == ".pt" else DEFAULT_POLICY_PATH
    room_arg = sys.argv[2] if len(sys.argv) > 2 else (sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).suffix != ".pt" else None)
    policy_path = _resolve_policy_path(policy_path)
    if not policy_path.is_file():
        print(f"Модель не найдена: {policy_path}")
        print("Запуск: python agent.py [путь/к/policy_final.pt] [комната]")
        sys.exit(1)

    policy, train_cfg = _load_policy(policy_path)

    if room_arg:
        p = Path(room_arg)
        if p.is_file():
            room = load_room_from_file(p)
            room_label = p.name
        else:
            room = load_room(str(p))
            room_label = p.name if p.suffix else str(p)
    else:
        room = load_room(DEFAULT_ROOM_NAME)
        room_label = DEFAULT_ROOM_NAME

    config = AgentConfig()
    controller = AgentController(room, config, fps=FPS)
    agent = Agent(room.agent.x, room.agent.y)
    body_radius_px = meters_to_pixels(config.radius)

    # Нормализация obs — должна совпадать с тем, что было при обучении.
    # max_steps берём из чекпоинта; если старый чекпоинт без этого поля — fallback на 90 сек.
    _ckpt_max_steps = train_cfg.get("max_steps", 0)
    _obs_max_steps = _ckpt_max_steps if _ckpt_max_steps > 0 else int(90 * FPS)
    _encoder_norm_scale = max(1.0, _obs_max_steps * (config.speed / FPS))
    _max_steps_vis = _obs_max_steps

    pygame.init()
    screen = pygame.display.set_mode((INITIAL_W, INITIAL_H), pygame.RESIZABLE)
    pygame.display.set_caption(f"Vacuum (модель) — {room_label}")
    clock = pygame.time.Clock()
    vw, vh = screen.get_size()
    manager = pygame_gui.UIManager((vw, vh))
    ui = _create_ui(manager, vw, vh)
    ui["label_room"].set_text(room_label[:30] + "..." if len(room_label) > 30 else room_label)

    world_surface = pygame.Surface((WORLD_SIZE, WORLD_SIZE))
    room_ctx = room_render.RenderContext(surface=world_surface, origin_x=WORLD_ORIGIN, origin_y=WORLD_ORIGIN)
    agent_ctx = agent_render.RenderContext(
        surface=world_surface,
        body_radius=meters_to_pixels(config.radius),
        origin_x=WORLD_ORIGIN,
        origin_y=WORLD_ORIGIN,
    )

    camera = CameraFree(zoom_min=0.1, zoom_max=5.0)
    canvas_w = max(1, vw - PANEL_W)
    canvas_h = vh
    camera.x = WORLD_ORIGIN - canvas_w // 2
    camera.y = WORLD_ORIGIN - canvas_h // 2

    speed_multiplier = 1
    free_space_map = FreeSpaceMap()
    visit_map = VisitMap()
    _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
    visit_total_cells = visit_map.total_cells
    # Старт в случайной свободной точке и со случайным углом (как при обучении)
    rx, ry = _random_agent_position_in_free_space(room, visit_map, body_radius_px)
    agent.x, agent.y = rx, ry
    agent.angle = random.uniform(0.0, 360.0)
    last_visit_pct_ticks = 0
    encoder_total = 0.0
    encoder_delta = 0.0   # расстояние за последний шаг (0 = стоим)
    step_count_vis = 0    # счётчик шагов для time = step/max_steps

    # Скрытое состояние GRU: сохраняется между шагами, сбрасывается на рестарте
    hidden = policy.init_hidden() if policy.use_gru else None

    def _get_obs() -> list[float]:
        sens = controller.get_sensors(agent)
        ir_fwd = min(sens.ir_forward       / IR_MAX_M, 1.0)
        ir_p30 = min(sens.ir_forward_p_30  / IR_MAX_M, 1.0)
        ir_m30 = min(sens.ir_forward_m_30  / IR_MAX_M, 1.0)
        angle_rad = math.radians(agent.angle)
        sin_a = math.sin(angle_rad)
        cos_a = math.cos(angle_rad)
        enc       = min(encoder_total / _encoder_norm_scale, 1.0)
        enc_delta = min(encoder_delta / MAX_STEP_DIST, 1.0)
        # Время: 1.0 означает 24 часа (24 * 60 * 60 секунд)
        time_norm = min(step_count_vis / (24 * 60 * 60 * FPS), 1.0)
        return [ir_fwd, ir_p30, ir_m30, sin_a, cos_a, enc, enc_delta, time_norm]

    while True:
        time_delta = clock.tick(FPS) / 1000.0
        vw, vh = screen.get_size()
        canvas_w = max(1, vw - PANEL_W)
        canvas_h = vh
        viewport_size = (canvas_w, canvas_h)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    rx, ry = _random_agent_position_in_free_space(room, visit_map, body_radius_px)
                    agent.x = rx
                    agent.y = ry
                    agent.angle = random.uniform(0.0, 360.0)
                    encoder_total = 0.0
                    encoder_delta = 0.0
                    step_count_vis = 0
                    hidden = policy.init_hidden() if policy.use_gru else None
                    _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
                    visit_total_cells = visit_map.total_cells
                    continue

            mouse_screen = pygame.mouse.get_pos()
            in_canvas = (
                event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL)
                and mouse_screen[0] >= PANEL_W
            )
            canvas_event = event
            if in_canvas and hasattr(event, "pos"):
                pos = (event.pos[0] - PANEL_W, event.pos[1])
                if event.type == pygame.MOUSEWHEEL:
                    canvas_event = pygame.event.Event(
                        pygame.MOUSEWHEEL,
                        pos=pos, x=getattr(event, "x", 0), y=getattr(event, "y", 0),
                    )
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    canvas_event = pygame.event.Event(event.type, pos=pos, button=event.button)
                else:
                    canvas_event = pygame.event.Event(event.type, pos=pos, rel=getattr(event, "rel", (0, 0)))

            if in_canvas and event.type == pygame.MOUSEWHEEL:
                camera.handle_event(canvas_event, viewport_size)

            manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == ui["btn_load"]:
                try:
                    from tkinter import Tk, filedialog
                    root = Tk()
                    root.withdraw()
                    path = filedialog.askopenfilename(initialdir=str(ROOMS_DIR), filetypes=[("JSON", "*.json")])
                    root.destroy()
                    if path:
                        room = load_room_from_file(Path(path))
                        controller = AgentController(room, config, fps=FPS)
                        agent.x = room.agent.x
                        agent.y = room.agent.y
                        agent.angle = 0.0
                        room_label = Path(path).name
                        ui["label_room"].set_text(room_label[:30] + "..." if len(room_label) > 30 else room_label)
                        pygame.display.set_caption(f"Vacuum (модель) — {room_label}")
                        _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
                        rx, ry = _random_agent_position_in_free_space(room, visit_map, body_radius_px)
                        agent.x, agent.y = rx, ry
                        agent.angle = random.uniform(0.0, 360.0)
                        visit_total_cells = visit_map.total_cells
                        encoder_total = 0.0
                        encoder_delta = 0.0
                        step_count_vis = 0
                        hidden = policy.init_hidden() if policy.use_gru else None
                except Exception as e:
                    print(f"Ошибка загрузки: {e}")

            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == ui["dropdown_speed"]:
                text = getattr(event, "text", "1x")
                if text and text.endswith("x") and text[:-1].isdigit():
                    speed_multiplier = int(text[:-1])
            if in_canvas:
                if (
                    canvas_event.type == pygame.MOUSEMOTION
                    or (
                        getattr(canvas_event, "button", None) == 2
                        and canvas_event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)
                    )
                ):
                    camera.handle_event(canvas_event, viewport_size)

        # Управление от модели
        for _ in range(speed_multiplier):
            obs = _get_obs()
            t = torch.tensor([obs], dtype=torch.float32)
            with torch.no_grad():
                # forward_step сохраняет hidden state для GRU между шагами
                logits, probs, hidden = policy.forward_step(t, hidden)
            action = torch.argmax(probs, dim=-1).item()
            action_ru  = ACTION_LABELS[action]
            p = probs.squeeze().tolist()
            # Если модель обучена на 4 действиях — дополняем нулями для отображения
            while len(p) < 8:
                p.append(0.0)
            ui["label_in_fwd"].set_text("Вход[ir_fwd] — {:.4f}".format(obs[0]))
            ui["label_in_p30"].set_text("Вход[ir_p30] — {:.4f}".format(obs[1]))
            ui["label_in_m30"].set_text("Вход[ir_m30] — {:.4f}".format(obs[2]))
            ui["label_in_sin"].set_text("Вход[sin_угол] — {:.4f}".format(obs[3]))
            ui["label_in_cos"].set_text("Вход[cos_угол] — {:.4f}".format(obs[4]))
            ui["label_in_enc"].set_text("Вход[enc] — {:.4f}".format(obs[5]))
            ui["label_in_delta"].set_text("Вход[delta] — {:.4f}".format(obs[6] if len(obs) > 6 else 0.0))
            ui["label_in_time"].set_text("Вход[time] — {:.4f}".format(obs[7] if len(obs) > 7 else 0.0))
            ui["label_out_fwd"].set_text("Выход[вперёд] — {:.4f}".format(p[0]))
            ui["label_out_back"].set_text("Выход[назад] — {:.4f}".format(p[1]))
            ui["label_out_left"].set_text("Выход[влево] — {:.4f}".format(p[2]))
            ui["label_out_right"].set_text("Выход[вправо] — {:.4f}".format(p[3]))
            ui["label_out_fwd_left"].set_text("Выход[вп+влево] — {:.4f}".format(p[4]))
            ui["label_out_fwd_right"].set_text("Выход[вп+вправо] — {:.4f}".format(p[5]))
            ui["label_out_back_left"].set_text("Выход[нз+влево] — {:.4f}".format(p[6]))
            ui["label_out_back_right"].set_text("Выход[нз+вправо] — {:.4f}".format(p[7]))
            ui["label_action"].set_text("Действие: {}".format(action_ru))
            prev_x, prev_y, prev_angle = agent.x, agent.y, agent.angle
            fwd, back, left, right = _ACTION_FLAGS[action] if action < len(_ACTION_FLAGS) else (False, False, False, False)
            result = controller.apply_flags(
                agent,
                move_forward=fwd, move_backward=back,
                turn_left=left,   turn_right=right,
            )
            encoder_delta = result.encoder
            encoder_total += result.encoder
            step_count_vis += 1
            if result.encoder > 0:
                update_visits(
                    agent, prev_x, prev_y, prev_angle,
                    visit_map, body_radius_px, WORLD_ORIGIN, WORLD_ORIGIN, True,
                )

        world_surface.fill(BG)
        room_render.draw_room(room, room_ctx)
        agent_render.draw_agent(agent, agent_ctx)
        sens = controller.get_sensors(agent)
        ox = WORLD_ORIGIN + agent.x
        oy = WORLD_ORIGIN + agent.y
        for angle_off, dist_m in [(0, sens.ir_forward), (30, sens.ir_forward_p_30), (-30, sens.ir_forward_m_30)]:
            rad = math.radians(agent.angle + angle_off)
            end_x = ox + meters_to_pixels(dist_m) * math.cos(rad)
            end_y = oy + meters_to_pixels(dist_m) * math.sin(rad)
            pygame.draw.line(world_surface, (255, 255, 255), (ox, oy), (end_x, end_y), 1)
        ui["label_encoder"].set_text(f"Encoder: {encoder_total:.2f} м")

        ticks_ms = pygame.time.get_ticks()
        if ticks_ms - last_visit_pct_ticks >= VISIT_PCT_UPDATE_MS:
            last_visit_pct_ticks = ticks_ms
            if visit_total_cells > 0:
                visited = visit_map.visited_count
                pct = 100.0 * visited / visit_total_cells
                ui["label_visit_pct"].set_text(f"Посещено: {visited} / {visit_total_cells} ({pct:.1f}%)")
            elif visit_map.has_map:
                ui["label_visit_pct"].set_text("Посещено: 0 / 0 (0%)")
            else:
                ui["label_visit_pct"].set_text("Посещено: —")

        if ui["chk_visit_map"].is_checked and visit_map.has_map:
            draw_data = visit_map.get_draw_data()
            if draw_data:
                visit_map_render.draw_visit_map(world_surface, draw_data)

        canvas_surf = pygame.Surface((canvas_w, canvas_h))
        camera.draw(world_surface, canvas_surf, viewport_size, BG)
        screen.fill(PANEL_BG)
        pygame.draw.rect(screen, BG, (PANEL_W, 0, canvas_w, canvas_h))
        screen.blit(canvas_surf, (PANEL_W, 0))

        if vw != manager.get_root_container().rect.w or vh != manager.get_root_container().rect.h:
            manager.set_window_resolution((vw, vh))
        manager.update(time_delta)
        manager.draw_ui(screen)

        pygame.display.flip()


if __name__ == "__main__":
    main()
