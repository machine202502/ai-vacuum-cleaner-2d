"""
Управление пылесосом обученной моделью (policy_final.pt).
То же, что game.py: комната, камера, карта посещений, ИК, encoder. Управление — по модели.
Поддерживает MLP и GRU-политику — конфиг читается автоматически из чекпоинта.
Кнопка «Скрыть стены, зоны, робота (только ИК)» — остаются лучи дальномеров на тёмном фоне.
Абляция: кнопки «обнулить канал» (текст на кнопке; UICheckBox в pygame_gui рисует подпись справа за rect).
Действие: сетка 3×3 со стрелками; зелёный — argmax, остальные серые.
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


def _install_pygame_gui_translate_fallback() -> None:
    """pygame_gui вызывает i18n.t() из копий translate (from ... import translate).
    Патчим utility и каждый модуль, где имя translate привязано при импорте."""
    import pygame_gui.core.utility as pgu_util
    from pygame_gui.elements import ui_button, ui_label, ui_text_box, ui_text_entry_line
    from pygame_gui.windows import ui_file_dialog

    _real = pgu_util.translate

    def _translate_safe(text_to_translate: str, **keywords) -> str:
        if text_to_translate is None:
            return ""
        try:
            out = _real(text_to_translate, **keywords)
            return text_to_translate if out is None else out
        except Exception:
            return text_to_translate

    safe = _translate_safe
    pgu_util.translate = safe  # type: ignore[assignment, misc]
    ui_label.translate = safe  # type: ignore[misc]
    ui_text_box.translate = safe  # type: ignore[misc]
    ui_button.translate = safe  # type: ignore[misc]
    ui_text_entry_line.translate = safe  # type: ignore[misc]
    ui_file_dialog.translate = safe  # type: ignore[misc]


WORLD_SIZE = 16000
WORLD_ORIGIN = WORLD_SIZE // 2
SCALE = meters_to_pixels(1.0)
FPS = 60
INITIAL_W = 1040
INITIAL_H = 720

PANEL_W = 300
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


def _toggle_selectable_button(btn: pygame_gui.elements.UIButton) -> None:
    """Переключатель на основе UIButton (текст по центру кнопки)."""
    if btn.is_selected:
        btn.unselect()
    else:
        btn.select()


def _obs_net_display_line(obs: list[float]) -> str:
    """Одна строка, без стрелок и спецсимволов (шрифт темы pygame_gui)."""
    if len(obs) < 6:
        return "val: " + " ".join(f"{v:.2f}" for v in obs)
    return (
        f"val: {obs[0]:.2f} {obs[1]:.2f} {obs[2]:.2f} | "
        f"sn {obs[3]:.2f} cs {obs[4]:.2f} | dl {obs[5]:.2f}"
    )


# 3x3: углы = диагонали, стороны = вперёд/назад/влево/вправо, центр = нейтрально
_PAD_CELL_ACTION: tuple[tuple[int | None, ...], ...] = (
    (4, 0, 5),
    (2, None, 3),
    (6, 1, 7),
)
# Индекс действия -> угол стрелки (радианы), кончик смотрит «наружу» от центра поля
_ACTION_ARROW_RAD = (
    -math.pi / 2,  # 0 вперёд
    math.pi / 2,  # 1 назад
    math.pi,  # 2 влево
    0.0,  # 3 вправо
    -3 * math.pi / 4,  # 4 вперёд+влево
    -math.pi / 4,  # 5 вперёд+вправо
    3 * math.pi / 4,  # 6 назад+влево
    math.pi / 4,  # 7 назад+вправо
)

_PAD_BG = (42, 42, 48)
_PAD_BORDER = (68, 68, 76)
_PAD_CELL_LINE = (58, 58, 66)
_PAD_OFF = (100, 100, 108)
_PAD_ON = (60, 210, 95)
_PAD_CENTER_DOT = (72, 72, 80)


def _draw_arrow_head(
    surf: pygame.Surface,
    cx: float,
    cy: float,
    angle_rad: float,
    length: float,
    width: float,
    color: tuple[int, int, int],
) -> None:
    """Треугольник: кончик в направлении angle_rad."""
    tip_x = cx + math.cos(angle_rad) * length
    tip_y = cy + math.sin(angle_rad) * length
    back_x = cx - math.cos(angle_rad) * (length * 0.35)
    back_y = cy - math.sin(angle_rad) * (length * 0.35)
    perp = angle_rad + math.pi / 2
    hw = width * 0.5
    p1 = (int(back_x + math.cos(perp) * hw), int(back_y + math.sin(perp) * hw))
    p2 = (int(back_x - math.cos(perp) * hw), int(back_y - math.sin(perp) * hw))
    pygame.draw.polygon(surf, color, [(int(tip_x), int(tip_y)), p1, p2])


def _draw_action_pad_3x3(
    surf: pygame.Surface,
    rect: pygame.Rect,
    action: int,
    n_act: int,
    caption: str,
) -> None:
    """Рисует сетку 3x3 со стрелками; argmax — зелёный, остальные — серые."""
    caption_h = 22
    inner = pygame.Rect(rect.x + 3, rect.y + 2, rect.w - 6, rect.h - 4 - caption_h)
    pygame.draw.rect(surf, _PAD_BG, inner, border_radius=6)
    pygame.draw.rect(surf, _PAD_BORDER, inner, width=1, border_radius=6)

    cw = inner.w // 3
    ch = inner.h // 3
    ac = min(max(action, 0), max(0, n_act - 1))

    for row in range(3):
        for col in range(3):
            aid = _PAD_CELL_ACTION[row][col]
            cell = pygame.Rect(inner.x + col * cw, inner.y + row * ch, cw, ch)
            pygame.draw.rect(surf, _PAD_CELL_LINE, cell, width=1)

            if aid is None:
                cc = cell.center
                pygame.draw.circle(surf, _PAD_CENTER_DOT, cc, max(2, min(cw, ch) // 10))
                continue
            if aid >= n_act:
                continue

            cx = float(cell.centerx)
            cy = float(cell.centery)
            L = min(cw, ch) * 0.38
            W = min(cw, ch) * 0.42
            col_arrow = _PAD_ON if aid == ac else _PAD_OFF
            _draw_arrow_head(surf, cx, cy, _ACTION_ARROW_RAD[aid], L, W, col_arrow)

    try:
        font = pygame.font.SysFont("segoe ui", 15)
    except Exception:
        font = pygame.font.Font(None, 18)
    line = f"#{action}  {caption}" if caption else f"#{action}"
    txt = font.render(line, True, (210, 210, 218))
    surf.blit(txt, (inner.x + 2, inner.bottom + 3))


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


def _create_ui(
    manager: pygame_gui.UIManager,
    vw: int,
    vh: int,
    *,
    zero_delta_ablate: bool = False,
) -> dict:
    bw = PANEL_W - 2 * BTN_MARGIN
    x, y = BTN_MARGIN, BTN_MARGIN

    h_btn  = 36
    h_line = 20
    h_room = 24
    h_drop = 32
    h_toggle = 36  # как у «Загрузить»: текст внутри кнопки (не UICheckBox — у него подпись справа за rect)
    h_hint = 88
    h_obs  = 26
    h_pad  = 118

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

    label_ablate_title = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line + 6),
        text="Обнулить канал (0 на вход сети):",
        manager=manager,
    )
    y += h_line + 6 + UI_GAP

    # Один столбец: текст целиком в подписи чекбокса (без стрелок и «красивых» символов)
    ablate_labels = (
        "ИК прямо",
        "ИК +30 град",
        "ИК -30 град",
        "синус угла",
        "косинус угла",
        "смещение за шаг (delta)",
    )
    btn_ablate: list[pygame_gui.elements.UIButton] = []
    for idx, lbl in enumerate(ablate_labels):
        b = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, y, bw, h_toggle),
            text=lbl,
            manager=manager,
        )
        if zero_delta_ablate and idx == 5:
            b.select()
        btn_ablate.append(b)
        y += h_toggle + 4
    y += UI_GAP

    label_obs_title = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Числа на вход (после обнулений):",
        manager=manager,
    )
    y += h_line + 2

    label_obs_net = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_obs),
        text="val: --",
        manager=manager,
    )
    y += h_obs + UI_GAP

    label_act_title = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Действие: зелёная стрелка = argmax",
        manager=manager,
    )
    y += h_line + 2

    label_gamepad = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_pad),
        text=" ",
        manager=manager,
    )
    y += h_pad + UI_GAP

    btn_visit_map = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(x, y, bw, h_toggle),
        text="Карта посещений",
        manager=manager,
    )
    btn_visit_map.select()
    y += h_toggle + UI_GAP

    btn_hide_room_robot = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(x, y, bw, h_toggle),
        text="Скрыть стены, зоны, робота (только ИК)",
        manager=manager,
    )
    y += h_toggle + UI_GAP

    label_visit_pct = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, bw, h_line),
        text="Посещено: -",
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
        text=(
            "Управление: модель\n"
            "ПКМ и колесо - камера\n"
            "R - новая позиция\n"
            "Запуск: --zero-delta только delta=0"
        ),
        manager=manager,
    )

    return {
        "btn_load":         btn_load,
        "label_room":       label_room,
        "label_speed":      label_speed,
        "dropdown_speed":   dropdown_speed,
        "btn_visit_map":       btn_visit_map,
        "btn_hide_room_robot": btn_hide_room_robot,
        "btn_ablate":          btn_ablate,
        "label_ablate_title":  label_ablate_title,
        "label_obs_title":     label_obs_title,
        "label_obs_net":       label_obs_net,
        "label_act_title":     label_act_title,
        "label_gamepad":       label_gamepad,
        "label_visit_pct":  label_visit_pct,
        "label_encoder":    label_encoder,
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
        obs_dim=cfg.get("obs_dim", 6),
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
    cli_zero_delta = "--zero-delta" in sys.argv[1:]
    argv = [a for a in sys.argv[1:] if a != "--zero-delta"]
    if len(argv) >= 1 and Path(argv[0]).suffix == ".pt":
        policy_path = Path(argv[0])
        room_arg = argv[1] if len(argv) >= 2 else None
    else:
        policy_path = DEFAULT_POLICY_PATH
        room_arg = argv[0] if len(argv) >= 1 else None
    policy_path = _resolve_policy_path(policy_path)
    if not policy_path.is_file():
        print(f"Модель не найдена: {policy_path}")
        print("Запуск: python agent.py [policy.pt] [комната] [--zero-delta]")
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

    pygame.init()
    _install_pygame_gui_translate_fallback()
    screen = pygame.display.set_mode((INITIAL_W, INITIAL_H), pygame.RESIZABLE)
    pygame.display.set_caption(f"Vacuum (модель) — {room_label}")
    clock = pygame.time.Clock()
    vw, vh = screen.get_size()
    manager = pygame_gui.UIManager((vw, vh))
    ui = _create_ui(manager, vw, vh, zero_delta_ablate=cli_zero_delta)
    ui["label_room"].set_text(room_label[:30] + "..." if len(room_label) > 30 else room_label)
    _toggle_button_targets: frozenset = frozenset(
        (*ui["btn_ablate"], ui["btn_visit_map"], ui["btn_hide_room_robot"])
    )

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

    # Скрытое состояние GRU: сохраняется между шагами, сбрасывается на рестарте
    hidden = policy.init_hidden() if policy.use_gru else None

    last_action = 0
    last_action_name = ""

    def _raw_obs_vector() -> list[float]:
        sens = controller.get_sensors(agent)
        ir_fwd = min(sens.ir_forward       / IR_MAX_M, 1.0)
        ir_p30 = min(sens.ir_forward_p_30  / IR_MAX_M, 1.0)
        ir_m30 = min(sens.ir_forward_m_30  / IR_MAX_M, 1.0)
        angle_rad = math.radians(agent.angle)
        sin_a = math.sin(angle_rad)
        cos_a = math.cos(angle_rad)
        max_step_dist = max(1e-9, config.speed / FPS)
        enc_delta_n = min(encoder_delta / max_step_dist, 1.0)
        return [ir_fwd, ir_p30, ir_m30, sin_a, cos_a, enc_delta_n]

    def _get_obs() -> list[float]:
        v = _raw_obs_vector()
        for i, b in enumerate(ui["btn_ablate"]):
            if b.is_selected:
                v[i] = 0.0
        return v

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

            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element in _toggle_button_targets:
                _toggle_selectable_button(event.ui_element)
            elif event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == ui["btn_load"]:
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
            p = probs.squeeze().tolist()
            # Если модель обучена на 4 действиях — дополняем нулями для отображения
            while len(p) < 8:
                p.append(0.0)
            ui["label_obs_net"].set_text(_obs_net_display_line(obs))
            last_action = action
            last_action_name = (
                ACTION_LABELS[action] if 0 <= action < len(ACTION_LABELS) else "?"
            )
            prev_x, prev_y, prev_angle = agent.x, agent.y, agent.angle
            fwd, back, left, right = _ACTION_FLAGS[action] if action < len(_ACTION_FLAGS) else (False, False, False, False)
            result = controller.apply_flags(
                agent,
                move_forward=fwd, move_backward=back,
                turn_left=left,   turn_right=right,
            )
            encoder_delta = result.encoder
            encoder_total += result.encoder
            if result.encoder > 0:
                update_visits(
                    agent, prev_x, prev_y, prev_angle,
                    visit_map, body_radius_px, WORLD_ORIGIN, WORLD_ORIGIN, True,
                )

        world_surface.fill(BG)
        hide_room_robot = ui["btn_hide_room_robot"].is_selected
        if not hide_room_robot:
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
                ui["label_visit_pct"].set_text("Посещено: -")

        if ui["btn_visit_map"].is_selected and visit_map.has_map:
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
        gp_rect = ui["label_gamepad"].get_abs_rect()
        _draw_action_pad_3x3(
            screen, gp_rect, last_action, policy.n_actions, last_action_name
        )

        pygame.display.flip()


if __name__ == "__main__":
    main()
