"""
Редактор планировок с визуальным UI (pygame_gui): кнопки, выбор цвета, режимы.
Холст справа — камера как в игре (зум, панорамирование). ЛКМ/ПКМ по стене — взять/применить цвет.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pygame
import pygame_gui
from pygame_gui.windows import UIColourPickerDialog

from cameras import CameraFree
from free_space import FreeSpaceMap
from free_space_render import draw_free_space
from room_loader import ROOMS_DIR
from room_loader import DEFAULT_WALL_COLOR
from units import meters_to_pixels

INITIAL_W = 1000
INITIAL_H = 700
PANEL_W = 260
FPS = 60
BG = (28, 28, 32)
PANEL_BG = (38, 38, 44)
TEXT_CLR = (200, 200, 208)
ZONE_DEFAULT_COLOR = [42, 42, 48]
SCALE = meters_to_pixels(1.0)
WORLD_SIZE = 16000
WORLD_ORIGIN = WORLD_SIZE // 2
UNDO_LIMIT = 100

BTN_H = 36
BTN_MARGIN = 10
SECTION_GAP = 16


def _default_room():
    return {
        "units": "meters",
        "agent": [1, 1],
        "zones": [],
        "walls": [],
    }


def _screen_to_world(canvas_pos: tuple[float, float], camera: CameraFree, zoom: float) -> tuple[float, float]:
    sx, sy = canvas_pos
    wx_px = camera.x + sx / zoom
    wy_px = camera.y + sy / zoom
    return ((wx_px - WORLD_ORIGIN) / SCALE, (wy_px - WORLD_ORIGIN) / SCALE)


def _snapshot(room_data: dict, zone_counter: int) -> tuple[dict, int]:
    return (copy.deepcopy(room_data), zone_counter)


def _hit_wall(room_data: dict, mx: float, my: float) -> int | None:
    """Индекс стены под точкой (в метрах) или None."""
    for i, wall in enumerate(room_data.get("walls", [])):
        x, y, w, h = wall[0], wall[1], wall[2], wall[3]
        if x <= mx <= x + w and y <= my <= y + h:
            return i
    return None


def _hit_zone(room_data: dict, mx: float, my: float) -> int | None:
    """Индекс зоны под точкой (в метрах) или None. Проверяем с конца — верхние рисуются поверх."""
    zones = room_data.get("zones", [])
    for i in range(len(zones) - 1, -1, -1):
        r = zones[i]["rect"]
        x, y, w, h = r[0], r[1], r[2], r[3]
        if x <= mx <= x + w and y <= my <= y + h:
            return i
    return None


HANDLE_R = 0.15  # зона ручек ресайза; центр прямоугольника — перетаскивание
MIN_RECT = 0.2


def _hit_handle(x: float, y: float, w: float, h: float, mx: float, my: float) -> str | None:
    """Какая ручка ресайза под точкой (mx,my). x,y,w,h — rect в метрах. Возвращает 'n'|'s'|'e'|'w'|'ne'|'nw'|'se'|'sw' или None (центр — для перетаскивания)."""
    if w <= 0 or h <= 0:
        return None
    # Полоса ручек не больше трети меньшей стороны, чтобы всегда был «центр» для движения
    band = min(HANDLE_R, min(w, h) / 3.0)
    left = mx < x + band
    right = mx > x + w - band
    top = my < y + band
    bottom = my > y + h - band
    if top and left:
        return "nw"
    if top and right:
        return "ne"
    if bottom and left:
        return "sw"
    if bottom and right:
        return "se"
    if top:
        return "n"
    if bottom:
        return "s"
    if left:
        return "w"
    if right:
        return "e"
    return None


def _room_bounds(room_data: dict) -> tuple[float, float, float, float] | None:
    """Ограничивающий прямоугольник квартиры в метрах (x, y, w, h). Считается по стенам, зонам и агенту."""
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


def _draw_room_data(surf, room_data: dict, scale: float, origin_x: int, origin_y: int, wall_color_override=None):
    ox, oy = origin_x, origin_y
    for z in room_data.get("zones", []):
        r = z["rect"]
        x = ox + int(r[0] * scale)
        y = oy + int(r[1] * scale)
        w, h = int(r[2] * scale), int(r[3] * scale)
        color = tuple(z["color"][:3])
        pygame.draw.rect(surf, color, (x, y, w, h))
    font = pygame.font.SysFont("Arial", max(10, int(14 * scale / 80)))
    for z in room_data.get("zones", []):
        r = z["rect"]
        cx = ox + (r[0] + r[2] / 2) * scale
        cy = oy + (r[1] + r[3] / 2) * scale
        text = font.render(z["name"], True, (100, 100, 108))
        tr = text.get_rect(center=(cx, cy))
        surf.blit(text, tr)
    for wall in room_data.get("walls", []):
        x = ox + int(wall[0] * scale)
        y = oy + int(wall[1] * scale)
        w, h = int(wall[2] * scale), int(wall[3] * scale)
        color = tuple(wall[4:7]) if len(wall) >= 7 else (wall_color_override or DEFAULT_WALL_COLOR)
        pygame.draw.rect(surf, color, (x, y, w, h))
    ax, ay = room_data.get("agent", [0, 0])
    agent_radius = int(0.2 * scale)
    pygame.draw.circle(surf, (50, 140, 230), (ox + int(ax * scale), oy + int(ay * scale)), agent_radius)
    pygame.draw.circle(surf, (30, 100, 180), (ox + int(ax * scale), oy + int(ay * scale)), agent_radius, 2)


def _create_ui(manager: pygame_gui.UIManager, vw: int, vh: int) -> dict:
    """Создаёт элементы панели. Возвращает dict с ссылками на элементы."""
    bw = PANEL_W - 2 * BTN_MARGIN
    y = BTN_MARGIN

    btn_load = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Загрузить (O)",
        manager=manager,
    )
    y += BTN_H + 4
    btn_save = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Сохранить (S)",
        manager=manager,
    )
    y += BTN_H + 4
    btn_save_as = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Сохранить как...",
        manager=manager,
    )
    y += BTN_H + 4
    btn_new = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Новая комната (N)",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    btn_undo = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, (bw - 4) // 2, BTN_H),
        text="Отмена",
        manager=manager,
    )
    btn_redo = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN + (bw - 4) // 2 + 4, y, (bw - 4) // 2, BTN_H),
        text="Повтор",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    label_mode = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Режим:",
        manager=manager,
    )
    y += 28
    btn_wall = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Стена",
        manager=manager,
    )
    y += BTN_H + 4
    btn_zone = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Зона",
        manager=manager,
    )
    y += BTN_H + 4
    btn_agent = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Робот",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    # Блок «Выбрано»: стена или зона
    label_selection = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Выбрано: —",
        manager=manager,
    )
    y += 28
    btn_selected_wall_color = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, (bw - 4) // 2, BTN_H),
        text="Цвет стены...",
        manager=manager,
    )
    btn_delete_wall = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN + (bw - 4) // 2 + 4, y, (bw - 4) // 2, BTN_H),
        text="Удалить стену",
        manager=manager,
    )
    y += BTN_H + 4
    entry_selected_zone_name = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 28),
        manager=manager,
    )
    y += 32
    btn_selected_zone_color = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, (bw - 4) // 2, BTN_H),
        text="Цвет зоны...",
        manager=manager,
    )
    btn_delete_zone = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN + (bw - 4) // 2 + 4, y, (bw - 4) // 2, BTN_H),
        text="Удалить зону",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    label_color = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Цвет новой стены:",
        manager=manager,
    )
    y += 28
    btn_color = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, (bw - 4) // 2, BTN_H),
        text="Выбрать цвет...",
        manager=manager,
    )
    btn_color_reset = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN + (bw - 4) // 2 + 4, y, (bw - 4) // 2, BTN_H),
        text="По умолчанию",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    label_zone_section = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 24),
        text="Зона (имя и цвет):",
        manager=manager,
    )
    y += 28
    label_zone_name = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, 70, 22),
        text="Имя:",
        manager=manager,
    )
    entry_zone_name = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(BTN_MARGIN + 72, y - 2, bw - 72, 28),
        manager=manager,
        placeholder_text="Зона 1",
    )
    y += 32
    label_zone_color = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 22),
        text="Цвет:",
        manager=manager,
    )
    y += 26
    btn_zone_color = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, BTN_H),
        text="Выбрать цвет зоны...",
        manager=manager,
    )
    y += BTN_H + SECTION_GAP

    label_path = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 60),
        text="(нет файла)",
        manager=manager,
    )
    y += 64

    label_free_section = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 22),
        text="Свободное пространство:",
        manager=manager,
    )
    y += 26
    btn_calc_free = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN, y, (bw - 4) // 2, BTN_H),
        text="Рассчитать",
        manager=manager,
    )
    btn_reset_free = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(BTN_MARGIN + (bw - 4) // 2 + 4, y, (bw - 4) // 2, BTN_H),
        text="Сбросить",
        manager=manager,
    )
    y += BTN_H + 4
    label_free_info = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(BTN_MARGIN, y, bw, 44),
        text="—",
        manager=manager,
    )

    return {
        "btn_load": btn_load,
        "btn_save": btn_save,
        "btn_save_as": btn_save_as,
        "btn_new": btn_new,
        "btn_undo": btn_undo,
        "btn_redo": btn_redo,
        "btn_wall": btn_wall,
        "btn_zone": btn_zone,
        "btn_agent": btn_agent,
        "btn_color": btn_color,
        "btn_color_reset": btn_color_reset,
        "entry_zone_name": entry_zone_name,
        "btn_zone_color": btn_zone_color,
        "label_selection": label_selection,
        "btn_selected_wall_color": btn_selected_wall_color,
        "btn_delete_wall": btn_delete_wall,
        "entry_selected_zone_name": entry_selected_zone_name,
        "btn_selected_zone_color": btn_selected_zone_color,
        "btn_delete_zone": btn_delete_zone,
        "label_mode": label_mode,
        "label_path": label_path,
        "label_free_section": label_free_section,
        "btn_calc_free": btn_calc_free,
        "btn_reset_free": btn_reset_free,
        "label_free_info": label_free_info,
    }


def main():
    pygame.init()
    screen = pygame.display.set_mode((INITIAL_W, INITIAL_H), pygame.RESIZABLE)
    pygame.display.set_caption("Редактор планировок")
    clock = pygame.time.Clock()

    vw, vh = screen.get_size()
    manager = pygame_gui.UIManager((vw, vh))
    ui = _create_ui(manager, vw, vh)

    room_data = _default_room()
    current_path = None
    camera = CameraFree(zoom_min=0.1, zoom_max=5.0)
    mode = "wall"
    drag_start = None
    moving_what = None  # "wall"|"zone"
    moving_index = None
    move_start_world = None
    move_start_rect = None
    resizing_what = None
    resizing_index = None
    resizing_handle = None
    resizing_start_rect = None
    resizing_start_world = None
    current_wall_color = list(DEFAULT_WALL_COLOR)
    zone_counter = 0
    camera_centered = False
    world_surface = None
    undo_stack = []
    redo_stack = []
    colour_picker = None
    zone_colour_picker = None
    colour_picker_for_selected_wall = False
    colour_picker_for_selected_zone = False
    current_zone_color = list(ZONE_DEFAULT_COLOR)
    selected_wall_index = None
    selected_zone_index = None
    SELECT_THRESHOLD = 0.15  # метров — меньше считаем кликом (выбор)
    free_space_map = FreeSpaceMap()

    def push_undo():
        nonlocal undo_stack, redo_stack
        if room_data is None:
            return
        undo_stack.append(_snapshot(room_data, zone_counter))
        if len(undo_stack) > UNDO_LIMIT:
            undo_stack.pop(0)
        redo_stack.clear()

    def do_undo():
        nonlocal room_data, zone_counter, undo_stack, redo_stack
        if not undo_stack or room_data is None:
            return
        redo_stack.append(_snapshot(room_data, zone_counter))
        room_data, zone_counter = undo_stack.pop()
        clear_selection()

    def do_redo():
        nonlocal room_data, zone_counter, undo_stack, redo_stack
        if not redo_stack or room_data is None:
            return
        undo_stack.append(_snapshot(room_data, zone_counter))
        room_data, zone_counter = redo_stack.pop()
        clear_selection()

    def load_path(path: Path):
        nonlocal room_data, current_path, camera_centered, undo_stack, redo_stack, zone_counter
        with path.open(encoding="utf-8") as f:
            room_data = json.load(f)
        if "zones" not in room_data:
            room_data["zones"] = []
        if "walls" not in room_data:
            room_data["walls"] = []
        zone_counter = 0
        for z in room_data.get("zones", []):
            n = z.get("name", "")
            if n.startswith("Зона "):
                try:
                    zone_counter = max(zone_counter, int(n.split()[-1]))
                except (ValueError, IndexError):
                    pass
        current_path = path
        camera_centered = False
        undo_stack.clear()
        redo_stack.clear()
        ui["label_path"].set_text(str(path)[:40] + "..." if len(str(path)) > 40 else str(path))
        ui["entry_zone_name"].set_text(f"Зона {zone_counter + 1}")

    def save_path(path: Path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(room_data, f, ensure_ascii=False, indent=2)
        nonlocal current_path
        current_path = path
        ui["label_path"].set_text(str(path)[:40] + "..." if len(str(path)) > 40 else str(path))

    def set_mode(m: str):
        nonlocal mode
        mode = m
        names = {"wall": "Стена", "zone": "Зона", "agent": "Робот"}
        ui["label_mode"].set_text("Режим: " + names.get(m, m))

    def clear_selection():
        nonlocal selected_wall_index, selected_zone_index, moving_what, moving_index, move_start_world, move_start_rect
        nonlocal resizing_what, resizing_index, resizing_handle, resizing_start_rect, resizing_start_world
        selected_wall_index = None
        selected_zone_index = None
        moving_what = None
        moving_index = None
        move_start_world = None
        move_start_rect = None
        resizing_what = None
        resizing_index = None
        resizing_handle = None
        resizing_start_rect = None
        resizing_start_world = None
        _update_selection_ui()

    def _update_selection_ui():
        if selected_wall_index is not None:
            ui["label_selection"].set_text("Выбрано: стена")
            ui["btn_selected_wall_color"].show()
            ui["btn_delete_wall"].show()
            ui["entry_selected_zone_name"].hide()
            ui["btn_selected_zone_color"].hide()
            ui["btn_delete_zone"].hide()
        elif selected_zone_index is not None and room_data:
            ui["label_selection"].set_text("Выбрано: зона")
            ui["btn_selected_wall_color"].hide()
            ui["btn_delete_wall"].hide()
            ui["entry_selected_zone_name"].show()
            ui["entry_selected_zone_name"].set_text(room_data["zones"][selected_zone_index]["name"])
            ui["btn_selected_zone_color"].show()
            ui["btn_delete_zone"].show()
        else:
            ui["label_selection"].set_text("Выбрано: —")
            ui["btn_selected_wall_color"].hide()
            ui["btn_delete_wall"].hide()
            ui["entry_selected_zone_name"].hide()
            ui["btn_selected_zone_color"].hide()
            ui["btn_delete_zone"].hide()

    def _ask_open_path():
        try:
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            path = filedialog.askopenfilename(initialdir=str(ROOMS_DIR), filetypes=[("JSON", "*.json")])
            root.destroy()
            return Path(path) if path else None
        except Exception:
            return None

    def _ask_save_path():
        try:
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(initialdir=str(ROOMS_DIR), defaultextension=".json", filetypes=[("JSON", "*.json")])
            root.destroy()
            return Path(path) if path else None
        except Exception:
            return None

    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            load_path(p)
    if current_path is None:
        ui["label_path"].set_text("(новая комната)")

    set_mode("wall")
    _update_selection_ui()

    while True:
        time_delta = clock.tick(FPS) / 1000.0
        vw, vh = screen.get_size()
        canvas_w = max(1, vw - PANEL_W)
        canvas_h = vh

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # События мыши в области холста — переводим в координаты холста
            mouse_screen = pygame.mouse.get_pos()
            in_canvas = event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL) and mouse_screen[0] >= PANEL_W
            canvas_event = event
            if in_canvas:
                pos = (mouse_screen[0] - PANEL_W, mouse_screen[1])
                if event.type == pygame.MOUSEWHEEL:
                    # Зум: pos в координатах холста, y — направление прокрутки (1 / -1)
                    canvas_event = pygame.event.Event(
                        pygame.MOUSEWHEEL,
                        pos=pos,
                        x=getattr(event, "x", 0),
                        y=getattr(event, "y", 0),
                    )
                elif hasattr(event, "pos"):
                    pos = (event.pos[0] - PANEL_W, event.pos[1])
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                        canvas_event = pygame.event.Event(event.type, pos=pos, button=event.button)
                    else:
                        canvas_event = pygame.event.Event(event.type, pos=pos, rel=getattr(event, "rel", (0, 0)))

            # Зум камеры — обрабатываем сразу (пока событие не перехватил GUI)
            if in_canvas and event.type == pygame.MOUSEWHEEL:
                camera.handle_event(canvas_event, (max(1, vw - PANEL_W), vh))

            manager.process_events(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    drag_start = None
                    clear_selection()
                if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and room_data:
                    if selected_wall_index is not None and room_data.get("walls"):
                        push_undo()
                        room_data["walls"].pop(selected_wall_index)
                        selected_wall_index = None
                        _update_selection_ui()
                    elif selected_zone_index is not None and room_data.get("zones"):
                        push_undo()
                        room_data["zones"].pop(selected_zone_index)
                        selected_zone_index = None
                        _update_selection_ui()
                if event.mod & pygame.KMOD_CTRL:
                    if event.key == pygame.K_z and not (event.mod & pygame.KMOD_SHIFT):
                        do_undo()
                    elif event.key == pygame.K_y or (event.key == pygame.K_z and event.mod & pygame.KMOD_SHIFT):
                        do_redo()

            # Обработка кнопок GUI
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == ui["btn_load"]:
                    p = _ask_open_path()
                    if p:
                        load_path(p)
                elif event.ui_element == ui["btn_save"]:
                    if room_data and current_path:
                        save_path(current_path)
                    else:
                        p = _ask_save_path()
                        if p:
                            save_path(p)
                elif event.ui_element == ui["btn_save_as"]:
                    if room_data:
                        p = _ask_save_path()
                        if p:
                            save_path(p)
                elif event.ui_element == ui["btn_new"]:
                    room_data = _default_room()
                    current_path = None
                    camera_centered = False
                    undo_stack.clear()
                    redo_stack.clear()
                    ui["label_path"].set_text("(новая комната)")
                elif event.ui_element == ui["btn_undo"]:
                    do_undo()
                elif event.ui_element == ui["btn_redo"]:
                    do_redo()
                elif event.ui_element == ui["btn_wall"]:
                    set_mode("wall")
                elif event.ui_element == ui["btn_zone"]:
                    set_mode("zone")
                elif event.ui_element == ui["btn_agent"]:
                    set_mode("agent")
                elif event.ui_element == ui["btn_color"] and colour_picker is None:
                    colour_picker_for_selected_wall = False
                    colour_picker = UIColourPickerDialog(
                        rect=pygame.Rect(PANEL_W + 30, 30, 340, 400),
                        manager=manager,
                        initial_colour=pygame.Color(*current_wall_color),
                        window_title="Цвет стены",
                    )
                elif event.ui_element == ui["btn_color_reset"]:
                    current_wall_color[:] = list(DEFAULT_WALL_COLOR)
                elif event.ui_element == ui["btn_zone_color"] and zone_colour_picker is None:
                    colour_picker_for_selected_zone = False
                    zone_colour_picker = UIColourPickerDialog(
                        rect=pygame.Rect(PANEL_W + 30, 30, 340, 400),
                        manager=manager,
                        initial_colour=pygame.Color(*current_zone_color),
                        window_title="Цвет зоны",
                    )
                elif event.ui_element == ui["btn_selected_wall_color"] and colour_picker is None and selected_wall_index is not None and room_data:
                    wall = room_data["walls"][selected_wall_index]
                    rgb = tuple(wall[4:7]) if len(wall) >= 7 else DEFAULT_WALL_COLOR
                    colour_picker_for_selected_wall = True
                    colour_picker = UIColourPickerDialog(
                        rect=pygame.Rect(PANEL_W + 30, 30, 340, 400),
                        manager=manager,
                        initial_colour=pygame.Color(*rgb),
                        window_title="Цвет стены",
                    )
                elif event.ui_element == ui["btn_delete_wall"] and selected_wall_index is not None and room_data:
                    push_undo()
                    room_data["walls"].pop(selected_wall_index)
                    clear_selection()
                elif event.ui_element == ui["btn_selected_zone_color"] and zone_colour_picker is None and selected_zone_index is not None and room_data:
                    colour_picker_for_selected_zone = True
                    z = room_data["zones"][selected_zone_index]
                    zone_colour_picker = UIColourPickerDialog(
                        rect=pygame.Rect(PANEL_W + 30, 30, 340, 400),
                        manager=manager,
                        initial_colour=pygame.Color(*z["color"][:3]),
                        window_title="Цвет зоны",
                    )
                elif event.ui_element == ui["btn_delete_zone"] and selected_zone_index is not None and room_data:
                    push_undo()
                    room_data["zones"].pop(selected_zone_index)
                    clear_selection()
                elif event.ui_element == ui["btn_calc_free"] and room_data:
                    bounds = _room_bounds(room_data)
                    if bounds:
                        if free_space_map.calculate(room_data, bounds, SCALE, WORLD_ORIGIN, WORLD_ORIGIN):
                            ui["label_free_info"].set_text(f"{free_space_map.free_pixels} px, {free_space_map.free_m2:.1f} кв.м")
                        else:
                            free_space_map.clear()
                            ui["label_free_info"].set_text("Не удалось (агент вне зоны?)")
                    else:
                        free_space_map.clear()
                        ui["label_free_info"].set_text("Нет границ")
                elif event.ui_element == ui["btn_reset_free"]:
                    free_space_map.clear()
                    ui["label_free_info"].set_text("—")

            # Имя зоны: применяем сразу при вводе (без Enter)
            if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED and event.ui_element == ui["entry_selected_zone_name"]:
                if selected_zone_index is not None and room_data and room_data.get("zones"):
                    text = getattr(event, "text", None) or ui["entry_selected_zone_name"].get_text() or ""
                    room_data["zones"][selected_zone_index]["name"] = text
            if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == ui["entry_selected_zone_name"]:
                if selected_zone_index is not None and room_data and room_data.get("zones"):
                    new_name = ui["entry_selected_zone_name"].get_text().strip()
                    if new_name:
                        room_data["zones"][selected_zone_index]["name"] = new_name

            if event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
                c = event.colour
                rgb = [c.r, c.g, c.b]
                if event.ui_element == colour_picker:
                    if colour_picker_for_selected_wall and selected_wall_index is not None and room_data:
                        push_undo()
                        wall = room_data["walls"][selected_wall_index]
                        while len(wall) < 7:
                            wall.append(DEFAULT_WALL_COLOR[len(wall) - 4])
                        wall[4], wall[5], wall[6] = rgb[0], rgb[1], rgb[2]
                        colour_picker_for_selected_wall = False
                    else:
                        current_wall_color[:] = rgb
                    colour_picker = None
                elif event.ui_element == zone_colour_picker:
                    if colour_picker_for_selected_zone and selected_zone_index is not None and room_data:
                        push_undo()
                        room_data["zones"][selected_zone_index]["color"] = list(rgb)
                        colour_picker_for_selected_zone = False
                    else:
                        current_zone_color[:] = rgb
                    zone_colour_picker = None

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if getattr(event, "ui_element", None) == colour_picker:
                    colour_picker = None
                    colour_picker_for_selected_wall = False
                elif getattr(event, "ui_element", None) == zone_colour_picker:
                    zone_colour_picker = None
                    colour_picker_for_selected_zone = False

            if room_data is None:
                if in_canvas:
                    camera.handle_event(canvas_event, (canvas_w, canvas_h))
                continue

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and not in_canvas:
                drag_start = None

            # Редактирование на холсте
            if in_canvas:
                # Камера: СКМ (панорама) и все MOTION; зум обработан выше до GUI
                if (
                    canvas_event.type == pygame.MOUSEMOTION
                    or (
                        getattr(canvas_event, "button", None) == 2
                        and canvas_event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)
                    )
                ):
                    camera.handle_event(canvas_event, (canvas_w, canvas_h))

                # Пока открыт выбор цвета — не обрабатывать клики по холсту (не сбрасывать выделение, не двигать/ресайзить)
                canvas_editing = not (colour_picker or zone_colour_picker)
                if canvas_editing and canvas_event.type == pygame.MOUSEBUTTONDOWN and canvas_event.button == 1:
                    wm = _screen_to_world(canvas_event.pos, camera, camera.zoom)
                    mx, my = wm[0], wm[1]
                    # Выбранная стена: клик по телу (без ручки) — двигать; по ручке — ресайз
                    if selected_wall_index is not None and room_data.get("walls"):
                        wall = room_data["walls"][selected_wall_index]
                        x, y, w, h = wall[0], wall[1], wall[2], wall[3]
                        on_this_wall = _hit_wall(room_data, mx, my) == selected_wall_index
                        handle = _hit_handle(x, y, w, h, mx, my) if on_this_wall else None
                        if on_this_wall and handle is None:
                            push_undo()
                            moving_what = "wall"
                            moving_index = selected_wall_index
                            move_start_world = (mx, my)
                            move_start_rect = [wall[0], wall[1], wall[2], wall[3]]
                        elif on_this_wall and handle:
                            push_undo()
                            resizing_what = "wall"
                            resizing_index = selected_wall_index
                            resizing_handle = handle
                            resizing_start_rect = [x, y, w, h]
                            resizing_start_world = (mx, my)
                    elif selected_zone_index is not None and room_data.get("zones"):
                        r = room_data["zones"][selected_zone_index]["rect"]
                        x, y, w, h = r[0], r[1], r[2], r[3]
                        on_this_zone = _hit_zone(room_data, mx, my) == selected_zone_index
                        handle = _hit_handle(x, y, w, h, mx, my) if on_this_zone else None
                        if on_this_zone and handle is None:
                            push_undo()
                            moving_what = "zone"
                            moving_index = selected_zone_index
                            move_start_world = (mx, my)
                            move_start_rect = list(r)
                        elif on_this_zone and handle:
                            push_undo()
                            resizing_what = "zone"
                            resizing_index = selected_zone_index
                            resizing_handle = handle
                            resizing_start_rect = [x, y, w, h]
                            resizing_start_world = (mx, my)
                    if not (moving_what or resizing_what):
                        if selected_wall_index is not None or selected_zone_index is not None:
                            # Клик был не по телу/ручке выбранного — снять выделение
                            on_selected = (
                                (selected_wall_index is not None and _hit_wall(room_data, mx, my) == selected_wall_index)
                                or (selected_zone_index is not None and _hit_zone(room_data, mx, my) == selected_zone_index)
                            )
                            if not on_selected:
                                clear_selection()
                                _update_selection_ui()
                            if mode == "wall":
                                drag_start = wm
                            elif mode == "zone":
                                drag_start = wm
                            elif mode == "agent":
                                push_undo()
                                room_data["agent"] = [round(wm[0], 2), round(wm[1], 2)]
                        elif mode == "wall":
                            drag_start = wm
                        elif mode == "zone":
                            drag_start = wm
                        elif mode == "agent":
                            push_undo()
                            room_data["agent"] = [round(wm[0], 2), round(wm[1], 2)]

                elif canvas_editing and canvas_event.type == pygame.MOUSEMOTION and (moving_what or resizing_what):
                    wm = _screen_to_world(canvas_event.pos, camera, camera.zoom)
                    if moving_what == "wall" and moving_index is not None and room_data.get("walls"):
                        dx = wm[0] - move_start_world[0]
                        dy = wm[1] - move_start_world[1]
                        wall = room_data["walls"][moving_index]
                        wall[0] = round(move_start_rect[0] + dx, 2)
                        wall[1] = round(move_start_rect[1] + dy, 2)
                    elif moving_what == "zone" and moving_index is not None and room_data.get("zones"):
                        dx = wm[0] - move_start_world[0]
                        dy = wm[1] - move_start_world[1]
                        r = room_data["zones"][moving_index]["rect"]
                        r[0] = round(move_start_rect[0] + dx, 2)
                        r[1] = round(move_start_rect[1] + dy, 2)
                    elif resizing_what and resizing_handle and resizing_start_rect is not None:
                        x, y, w, h = resizing_start_rect[0], resizing_start_rect[1], resizing_start_rect[2], resizing_start_rect[3]
                        mx, my = wm[0], wm[1]
                        if "e" in resizing_handle:
                            w = max(MIN_RECT, mx - x)
                        if "w" in resizing_handle:
                            nw = max(MIN_RECT, (x + w) - mx)
                            x, w = mx, nw
                        if "s" in resizing_handle:
                            h = max(MIN_RECT, my - y)
                        if "n" in resizing_handle:
                            nh = max(MIN_RECT, (y + h) - my)
                            y, h = my, nh
                        if resizing_what == "wall" and room_data.get("walls"):
                            room_data["walls"][resizing_index][0] = round(x, 2)
                            room_data["walls"][resizing_index][1] = round(y, 2)
                            room_data["walls"][resizing_index][2] = round(w, 2)
                            room_data["walls"][resizing_index][3] = round(h, 2)
                        elif resizing_what == "zone" and room_data.get("zones"):
                            room_data["zones"][resizing_index]["rect"] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]

                elif canvas_editing and canvas_event.type == pygame.MOUSEBUTTONUP and canvas_event.button == 1:
                    if moving_what or resizing_what:
                        moving_what = None
                        moving_index = None
                        move_start_world = None
                        move_start_rect = None
                        resizing_what = None
                        resizing_index = None
                        resizing_handle = None
                        resizing_start_rect = None
                        resizing_start_world = None
                    elif drag_start:
                        wm = _screen_to_world(canvas_event.pos, camera, camera.zoom)
                        x1, y1 = drag_start[0], drag_start[1]
                        x2, y2 = wm[0], wm[1]
                        x, y = min(x1, x2), min(y1, y2)
                        w, h = abs(x2 - x1), abs(y2 - y1)
                        is_click = w < SELECT_THRESHOLD and h < SELECT_THRESHOLD
                        if mode == "wall":
                            if is_click:
                                idx = _hit_wall(room_data, x1, y1)
                                if idx is not None:
                                    selected_wall_index = idx
                                    selected_zone_index = None
                                    wall = room_data["walls"][idx]
                                    if len(wall) >= 7:
                                        current_wall_color[:] = [wall[4], wall[5], wall[6]]
                                    else:
                                        current_wall_color[:] = list(DEFAULT_WALL_COLOR)
                                else:
                                    clear_selection()
                                _update_selection_ui()
                            else:
                                push_undo()
                                room_data["walls"].append([round(x, 2), round(y, 2), round(max(0.2, w), 2), round(max(0.2, h), 2)] + list(current_wall_color))
                                clear_selection()
                                _update_selection_ui()
                        elif mode == "zone":
                            if is_click:
                                idx = _hit_zone(room_data, x1, y1)
                                if idx is not None:
                                    selected_zone_index = idx
                                    selected_wall_index = None
                                    _update_selection_ui()
                                else:
                                    clear_selection()
                                    _update_selection_ui()
                            elif w >= 0.2 and h >= 0.2:
                                push_undo()
                                zone_counter += 1
                                zone_name = ui["entry_zone_name"].get_text().strip() or f"Зона {zone_counter}"
                                room_data["zones"].append({
                                    "name": zone_name,
                                    "rect": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                                    "color": list(current_zone_color),
                                })
                                ui["entry_zone_name"].set_text(f"Зона {zone_counter + 1}")
                                clear_selection()
                                _update_selection_ui()
                        drag_start = None
                elif canvas_editing and canvas_event.type == pygame.MOUSEBUTTONUP and canvas_event.button == 3 and mode == "wall" and room_data.get("walls"):
                    wm = _screen_to_world(canvas_event.pos, camera, camera.zoom)
                    mx, my = wm[0], wm[1]
                    for wall in room_data["walls"]:
                        x, y, w, h = wall[0], wall[1], wall[2], wall[3]
                        if x <= mx <= x + w and y <= my <= y + h:
                            push_undo()
                            while len(wall) < 7:
                                wall.append(DEFAULT_WALL_COLOR[len(wall) - 4])
                            wall[4], wall[5], wall[6] = current_wall_color[0], current_wall_color[1], current_wall_color[2]
                            break

        manager.update(time_delta)

        # Отрисовка
        screen.fill(PANEL_BG)
        pygame.draw.rect(screen, BG, (PANEL_W, 0, canvas_w, canvas_h))

        if room_data is None:
            no_room = pygame.Surface((canvas_w, canvas_h))
            no_room.fill(BG)
            font = pygame.font.SysFont("Arial", 18)
            hint = font.render("Нажмите «Загрузить» или «Новая комната»", True, TEXT_CLR)
            no_room.blit(hint, (canvas_w // 2 - hint.get_width() // 2, canvas_h // 2 - 20))
            screen.blit(no_room, (PANEL_W, 0))
            if vw != manager.get_root_container().rect.w or vh != manager.get_root_container().rect.h:
                manager.set_window_resolution((vw, vh))
            manager.draw_ui(screen)
            pygame.display.flip()
            continue

        if not camera_centered:
            camera.x = WORLD_ORIGIN - canvas_w // 2
            camera.y = WORLD_ORIGIN - canvas_h // 2
            camera_centered = True

        if world_surface is None:
            world_surface = pygame.Surface((WORLD_SIZE, WORLD_SIZE))
        world_surface.fill(BG)
        _draw_room_data(world_surface, room_data, SCALE, WORLD_ORIGIN, WORLD_ORIGIN)
        # Красный прямоугольник — ограничивающая рамка квартиры (в реальном времени)
        bounds = _room_bounds(room_data)
        if bounds is not None:
            bx, by, bw, bh = bounds
            rx = WORLD_ORIGIN + int(bx * SCALE)
            ry = WORLD_ORIGIN + int(by * SCALE)
            rw = max(1, int(bw * SCALE))
            rh = max(1, int(bh * SCALE))
            pygame.draw.rect(world_surface, (255, 60, 60), (rx, ry, rw, rh), 2)
            # Сброс расчёта свободного пространства при изменении рамки
            if free_space_map.has_result and free_space_map.bounds != bounds:
                free_space_map.clear()
                ui["label_free_info"].set_text("—")
        else:
            if free_space_map.has_result:
                free_space_map.clear()
                ui["label_free_info"].set_text("—")
        # Отрисовка достижимого пространства (мигающий зелёный)
        if bounds is not None and free_space_map.has_result and free_space_map.bounds == bounds:
            draw_data = free_space_map.get_draw_data()
            if draw_data:
                draw_free_space(world_surface, draw_data, pygame.time.get_ticks())
        # Подсветка выбранной стены или зоны
        ox, oy = WORLD_ORIGIN, WORLD_ORIGIN
        if selected_wall_index is not None and room_data.get("walls"):
            w = room_data["walls"][selected_wall_index]
            x = ox + int(w[0] * SCALE)
            y = oy + int(w[1] * SCALE)
            ww, hh = int(w[2] * SCALE), int(w[3] * SCALE)
            pygame.draw.rect(world_surface, (255, 220, 80), (x, y, ww, hh), 3)
        if selected_zone_index is not None and room_data.get("zones"):
            z = room_data["zones"][selected_zone_index]["rect"]
            x = ox + int(z[0] * SCALE)
            y = oy + int(z[1] * SCALE)
            ww, hh = int(z[2] * SCALE), int(z[3] * SCALE)
            pygame.draw.rect(world_surface, (255, 220, 80), (x, y, ww, hh), 3)
        # Ручки ресайза (квадраты по углам и серединам сторон)
        def _draw_handles(rect_x: float, rect_y: float, rect_w: float, rect_h: float):
            hr = max(2, int(HANDLE_R * SCALE))
            pts = [
                (rect_x, rect_y), (rect_x + rect_w // 2, rect_y), (rect_x + rect_w, rect_y),
                (rect_x + rect_w, rect_y + rect_h // 2), (rect_x + rect_w, rect_y + rect_h),
                (rect_x + rect_w // 2, rect_y + rect_h), (rect_x, rect_y + rect_h), (rect_x, rect_y + rect_h // 2),
            ]
            for px, py in pts:
                pygame.draw.rect(world_surface, (255, 255, 200), (px - hr // 2, py - hr // 2, hr, hr))
                pygame.draw.rect(world_surface, (180, 160, 60), (px - hr // 2, py - hr // 2, hr, hr), 1)
        if selected_wall_index is not None and room_data.get("walls") and not resizing_what:
            w = room_data["walls"][selected_wall_index]
            _draw_handles(ox + int(w[0] * SCALE), oy + int(w[1] * SCALE), int(w[2] * SCALE), int(w[3] * SCALE))
        if selected_zone_index is not None and room_data.get("zones") and not resizing_what:
            z = room_data["zones"][selected_zone_index]["rect"]
            _draw_handles(ox + int(z[0] * SCALE), oy + int(z[1] * SCALE), int(z[2] * SCALE), int(z[3] * SCALE))
        if drag_start and mode in ("wall", "zone"):
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] >= PANEL_W:
                cx, cy = mouse_pos[0] - PANEL_W, mouse_pos[1]
                wm = _screen_to_world((cx, cy), camera, camera.zoom)
                x = min(drag_start[0], wm[0])
                y = min(drag_start[1], wm[1])
                w = max(0.1, abs(wm[0] - drag_start[0]))
                h = max(0.1, abs(wm[1] - drag_start[1]))
                if mode == "wall":
                    pygame.draw.rect(world_surface, tuple(current_wall_color), (WORLD_ORIGIN + int(x * SCALE), WORLD_ORIGIN + int(y * SCALE), int(w * SCALE), int(h * SCALE)), 2)
                else:
                    pygame.draw.rect(world_surface, (100, 100, 108), (WORLD_ORIGIN + int(x * SCALE), WORLD_ORIGIN + int(y * SCALE), int(w * SCALE), int(h * SCALE)), 2)

        canvas_surf = pygame.Surface((canvas_w, canvas_h))
        camera.draw(world_surface, canvas_surf, (canvas_w, canvas_h), BG)
        screen.blit(canvas_surf, (PANEL_W, 0))

        # Образцы цветов под кнопками
        br = ui["btn_color"].rect
        wall_swatch = pygame.Rect(BTN_MARGIN, br.bottom + 4, 36, 22)
        pygame.draw.rect(screen, tuple(current_wall_color), wall_swatch)
        pygame.draw.rect(screen, (100, 100, 108), wall_swatch, 1)
        zbr = ui["btn_zone_color"].rect
        zone_swatch = pygame.Rect(BTN_MARGIN, zbr.bottom + 4, 36, 22)
        pygame.draw.rect(screen, tuple(current_zone_color), zone_swatch)
        pygame.draw.rect(screen, (100, 100, 108), zone_swatch, 1)

        if vw != manager.get_root_container().rect.w or vh != manager.get_root_container().rect.h:
            manager.set_window_resolution((vw, vh))

        manager.draw_ui(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
