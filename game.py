"""
Игра: панель UI, загрузка комнаты, скорость мира, камера, управление роботом, ПКМ — позиция робота.
Карта свободного пространства и карта посещений обновляются при загрузке и при движении вперёд.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pygame
import pygame_gui

import agent_render
import room_render
from agent_controller import AgentConfig, AgentController, update_visits
from cameras import CameraFree
from free_space import FreeSpaceMap
from room_loader import ROOMS_DIR, load_room, load_room_from_file
from units import meters_to_pixels
from visit_map import VisitMap
import visit_map_render

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
SECTION_GAP = 16

SPEED_OPTIONS = ["1x", "2x", "3x", "4x", "5x"]
VISIT_PCT_UPDATE_MS = 1000  # обновлять надпись «Посещено» раз в секунду


def _room_data_from_room(room, scale: float) -> dict:
    """Преобразовать Room (пиксели) в room_data (метры) для FreeSpaceMap."""
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
    """Ограничивающий прямоугольник в метрах (x, y, w, h)."""
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
    """Пересчитать свободное пространство от текущей позиции агента и инициализировать карту посещений."""
    room_data = _room_data_from_room(room, SCALE)
    room_data["agent"] = [agent.x / SCALE, agent.y / SCALE]
    bounds = _room_bounds(room_data)
    if bounds and free_space_map.calculate(room_data, bounds, SCALE, WORLD_ORIGIN, WORLD_ORIGIN):
        visit_map.init_from_free_space(free_space_map)


def _circle_hits_rect(cx: float, cy: float, radius: float, r) -> bool:
    """Круг (cx, cy, radius) пересекает прямоугольник r (x, y, w, h или .left/.right/.top/.bottom)."""
    left = getattr(r, "left", r.x)
    top = getattr(r, "top", r.y)
    right = getattr(r, "right", r.x + r.w)
    bottom = getattr(r, "bottom", r.y + r.h)
    px = max(left, min(cx, right))
    py = max(top, min(cy, bottom))
    return (cx - px) ** 2 + (cy - py) ** 2 <= radius**2


def _circle_hits_any_wall(cx: float, cy: float, radius_px: float, room) -> bool:
    """Круг в координатах комнаты касается любой стены."""
    for w in room.walls:
        if _circle_hits_rect(cx, cy, radius_px, w.rect):
            return True
    return False


def _random_agent_position_in_free_space(room, visit_map: VisitMap, body_radius_px: float) -> tuple[float, float]:
    """
    Случайная точка в свободном пространстве, куда можно поставить агента (круг не касается стен).
    Берём случайную достижимую ячейку, центр ячейки в координатах комнаты — кандидат;
    если круг агента не пересекает стены, возвращаем (x, y), иначе пробуем другую ячейку.
    Если достижимых нет или ни одна не подошла — возвращаем стартовую позицию комнаты.
    """
    if not visit_map.has_map or not visit_map.reachable_cells:
        return (room.agent.x, room.agent.y)
    cell_px = visit_map.cell_px
    left_px = visit_map.left_px
    top_px = visit_map.top_px
    cells = list(visit_map.reachable_cells)
    random.shuffle(cells)
    for (i, j) in cells:
        world_x = left_px + (i + 0.5) * cell_px
        world_y = top_px + (j + 0.5) * cell_px
        room_x = world_x - WORLD_ORIGIN
        room_y = world_y - WORLD_ORIGIN
        if not _circle_hits_any_wall(room_x, room_y, body_radius_px, room):
            return (room_x, room_y)
    return (room.agent.x, room.agent.y)


def _screen_to_world_px(screen_pos: tuple[float, float], camera: CameraFree) -> tuple[float, float]:
    """Экранные координаты (viewport) в мировые пиксели (world_surface)."""
    sx, sy = screen_pos
    wx = camera.x + sx / camera.zoom
    wy = camera.y + sy / camera.zoom
    return (wx, wy)


def _ask_open_room_path() -> Path | None:
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(initialdir=str(ROOMS_DIR), filetypes=[("JSON", "*.json")])
        root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def _create_ui(manager: pygame_gui.UIManager, vw: int, vh: int) -> dict:
    """Создаёт элементы панели: Загрузить, комната, скорость мира."""
    bw = PANEL_W - 2 * BTN_MARGIN
    y = BTN_MARGIN

    btn_load = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Загрузить комнату",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    label_room = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 40),
        text="(комната)",
        manager=manager,
    )
    y += 44

    label_speed = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 22),
        text="Скорость мира:",
        manager=manager,
    )
    y += 26
    dropdown_speed = pygame_gui.elements.UIDropDownMenu(
        options_list=SPEED_OPTIONS,
        starting_option="1x",
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 32),
        manager=manager,
    )
    y += 36

    chk_visit_map = pygame_gui.elements.UICheckBox(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 28),
        text="Карта посещений",
        manager=manager,
        initial_state=True,
    )
    y += 32

    label_visit_pct = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Посещено: —",
        manager=manager,
    )
    y += 28

    label_encoder = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Encoder: 0.00 м",
        manager=manager,
    )
    y += 28

    label_hint = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 80),
        text="WASD — движение\nПКМ — позиция робота\nСКМ/колёсико — камера\nR — рестарт",
        manager=manager,
    )

    return {
        "btn_load": btn_load,
        "label_room": label_room,
        "label_speed": label_speed,
        "dropdown_speed": dropdown_speed,
        "chk_visit_map": chk_visit_map,
        "label_visit_pct": label_visit_pct,
        "label_encoder": label_encoder,
        "label_hint": label_hint,
    }


class Agent:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.angle = 0.0


def main() -> None:
    # Начальная загрузка комнаты
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
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

    pygame.init()
    screen = pygame.display.set_mode((INITIAL_W, INITIAL_H), pygame.RESIZABLE)
    pygame.display.set_caption(f"Vacuum — {room_label}")
    clock = pygame.time.Clock()
    vw, vh = screen.get_size()
    manager = pygame_gui.UIManager((vw, vh))
    ui = _create_ui(manager, vw, vh)
    ui["label_room"].set_text(room_label[: 30] + "..." if len(room_label) > 30 else room_label)

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

    speed_multiplier = 1  # 1, 2, 3, 4, 5
    free_space_map = FreeSpaceMap()
    visit_map = VisitMap()
    _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
    visit_total_cells = len(visit_map.reachable_cells) if visit_map.has_map else 0
    last_visit_pct_ticks = 0
    encoder_total = 0.0  # накопленный путь, м
    body_radius_px = meters_to_pixels(config.radius)

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
                    agent.angle = 0.0
                    encoder_total = 0.0
                    _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
                    visit_total_cells = len(visit_map.reachable_cells) if visit_map.has_map else 0
                    continue

            # События мыши в области холста (справа от панели)
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
                        pos=pos,
                        x=getattr(event, "x", 0),
                        y=getattr(event, "y", 0),
                    )
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    canvas_event = pygame.event.Event(event.type, pos=pos, button=event.button)
                else:
                    canvas_event = pygame.event.Event(event.type, pos=pos, rel=getattr(event, "rel", (0, 0)))

            # Зум камеры — до GUI
            if in_canvas and event.type == pygame.MOUSEWHEEL:
                camera.handle_event(canvas_event, viewport_size)

            manager.process_events(event)

            # UI: загрузка комнаты
            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == ui["btn_load"]:
                path = _ask_open_room_path()
                if path:
                    try:
                        room = load_room_from_file(path)
                        controller = AgentController(room, config, fps=FPS)
                        agent.x = room.agent.x
                        agent.y = room.agent.y
                        agent.angle = 0.0
                        room_label = path.name
                        ui["label_room"].set_text(room_label[:30] + "..." if len(room_label) > 30 else room_label)
                        pygame.display.set_caption(f"Vacuum — {room_label}")
                        _update_free_space_and_visit_map(room, agent, free_space_map, visit_map)
                        visit_total_cells = len(visit_map.reachable_cells) if visit_map.has_map else 0
                        encoder_total = 0.0
                    except Exception as e:
                        print(f"Ошибка загрузки: {e}")

            # UI: скорость мира
            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == ui["dropdown_speed"]:
                text = getattr(event, "text", "1x")
                if text and text.endswith("x") and text[:-1].isdigit():
                    speed_multiplier = int(text[:-1])

            # Камера: СКМ и MOTION
            if in_canvas:
                if (
                    canvas_event.type == pygame.MOUSEMOTION
                    or (
                        getattr(canvas_event, "button", None) == 2
                        and canvas_event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)
                    )
                ):
                    camera.handle_event(canvas_event, viewport_size)
            # ПКМ — установить позицию робота (только по холсту)
            if in_canvas and event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 3:
                pos_canvas = (event.pos[0] - PANEL_W, event.pos[1])
                wx, wy = _screen_to_world_px(pos_canvas, camera)
                agent.x = wx - WORLD_ORIGIN
                agent.y = wy - WORLD_ORIGIN

        # Управление роботом с учётом скорости мира; движение и вращение независимы
        keys = pygame.key.get_pressed()
        ctrl_fwd  = bool(keys[pygame.K_UP]    or keys[pygame.K_w])
        ctrl_back = bool(keys[pygame.K_DOWN]  or keys[pygame.K_s])
        ctrl_left = bool(keys[pygame.K_LEFT]  or keys[pygame.K_a])
        ctrl_right= bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        for _ in range(speed_multiplier):
            prev_x, prev_y, prev_angle = agent.x, agent.y, agent.angle
            result = controller.apply_flags(
                agent,
                move_forward=ctrl_fwd, move_backward=ctrl_back,
                turn_left=ctrl_left,   turn_right=ctrl_right,
            )
            encoder_total += result.encoder
            if result.encoder > 0:
                update_visits(
                    agent, prev_x, prev_y, prev_angle,
                    visit_map, body_radius_px, WORLD_ORIGIN, WORLD_ORIGIN, True,
                )

        world_surface.fill(BG)
        room_render.draw_room(room, room_ctx)
        agent_render.draw_agent(agent, agent_ctx)
        # ИК-датчики: белые лучи от центра агента
        sens = controller.get_sensors(agent)
        ox = WORLD_ORIGIN + agent.x
        oy = WORLD_ORIGIN + agent.y
        for angle_off, dist_m in [(0, sens.ir_forward), (30, sens.ir_forward_p_30), (-30, sens.ir_forward_m_30)]:
            rad = math.radians(agent.angle + angle_off)
            end_x = ox + meters_to_pixels(dist_m) * math.cos(rad)
            end_y = oy + meters_to_pixels(dist_m) * math.sin(rad)
            pygame.draw.line(world_surface, (255, 255, 255), (ox, oy), (end_x, end_y), 1)
        ui["label_encoder"].set_text(f"Encoder: {encoder_total:.2f} м")
        # Процент посещения: обновлять раз в секунду, всего достижимых — один раз (visit_total_cells)
        ticks_ms = pygame.time.get_ticks()
        if ticks_ms - last_visit_pct_ticks >= VISIT_PCT_UPDATE_MS:
            last_visit_pct_ticks = ticks_ms
            if visit_total_cells > 0:
                visited = sum(1 for (i, j) in visit_map.reachable_cells if visit_map.get_count(i, j) > 0)
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

        # Вид камеры только на холст
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
