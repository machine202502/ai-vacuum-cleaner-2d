"""
Логика посещения пространства: по геометрии валика (suction), углу ±60° от его концов и движению вперёд.
Засчитываем ячейку, если она была перед валиком (в клине) и стала внутри или позади валика после шага вперёд.
"""
from __future__ import annotations

import math

SUCTION_ANGLE_DEG = 60  # ±60° от направления вперёд, от левого и правого конца валика


def _suction_geometry(body_radius_px: float) -> tuple[float, float, float]:
    """По радиусу корпуса (px) возвращает (suction_width, suction_length, suction_offset) в px."""
    r = body_radius_px
    suction_width = max(2, int(r * 0.33))
    suction_length = int(r * 1.1)
    suction_offset = max(2, int(r * 0.33))
    return (suction_width, suction_length, suction_offset)


def _suction_front_ends_world(
    center_x: float, center_y: float, angle_deg: float,
    suction_width: float, suction_length: float, suction_offset: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    В мировых координатах: левый и правый концы передней грани валика (та, что смотрит по направлению движения).
    При angle=0 «нос» вправо, перед валика — правая грань прямоугольника в локальных координатах.
    """
    rad = math.radians(angle_deg)
    # Локально: центр (0,0), передняя грань на x = + (suction_offset от центра круга до передней грани валика)
    # В agent_render локально: rect (cx - suction_offset - suction_width, cy - suction_length/2, width, length)
    # значит передняя грань (правая) на x = cx - suction_offset = 0 - suction_offset (если центр круга 0,0 то cx=0 не используется, центр агента 0,0)
    # Левый конец передней грани в локальных координатах (относительно центра агента): (suction_offset, -suction_length/2)
    # Правый конец: (suction_offset, suction_length/2)
    # Но в рендере центр круга cx,cy — это центр поверхности, не центр агента. Центр агента = центр круга. Так что перед валика: x = cx - suction_offset, т.е. локально от центра агента передняя грань в (+suction_offset, 0) нет: rect left = cx - suction_offset - width, так что передняя грань (right side of rect) = left + width = cx - suction_offset. При cx=0 это -suction_offset. То есть в локальных координатах «нос» вправо, валик слева от центра, его передняя грань на x = -suction_offset. Левый конец передней грани (верх в локальных координатах): (-suction_offset, -suction_length/2), правый конец: (-suction_offset, suction_length/2). Поворачиваем на angle: cos(angle)*(−offset) + ..., так что при angle=0 точка (-suction_offset, -suction_length/2) остаётся. И forward в локальных = (1, 0), после поворота (cos(angle), sin(angle)). Тогда «перед» валика — это сторона в направлении (cos(angle), sin(angle)). Левый конец передней грани в локальных: (-suction_offset, -suction_length/2), правый: (-suction_offset, suction_length/2).
    left_local_x = -suction_offset
    left_local_y = -suction_length / 2
    right_local_x = -suction_offset
    right_local_y = suction_length / 2
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    left_wx = center_x + left_local_x * cos_a - left_local_y * sin_a
    left_wy = center_y + left_local_x * sin_a + left_local_y * cos_a
    right_wx = center_x + right_local_x * cos_a - right_local_y * sin_a
    right_wy = center_y + right_local_x * sin_a + right_local_y * cos_a
    return ((left_wx, left_wy), (right_wx, right_wy))


def _forward_dir(angle_deg: float) -> tuple[float, float]:
    return (math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)))


def _point_in_wedge(
    px: float, py: float,
    left_end: tuple[float, float], right_end: tuple[float, float],
    angle_deg: float,
) -> bool:
    """Точка (px, py) лежит в клине перед валиком: ±60° от левого и правого конца передней грани."""
    lx, ly = left_end
    rx, ry = right_end
    # Луч от левого конца под углом (angle - 60°), от правого под (angle + 60°)
    left_ray_ang = math.radians(angle_deg - SUCTION_ANGLE_DEG)
    right_ray_ang = math.radians(angle_deg + SUCTION_ANGLE_DEG)
    dx_left = math.cos(left_ray_ang)
    dy_left = math.sin(left_ray_ang)
    dx_right = math.cos(right_ray_ang)
    dy_right = math.sin(right_ray_ang)
    # Точка P должна быть справа от луча из left_end (т.е. «внутри» клина по левому краю)
    # cross(P - left_end, left_ray) >= 0 при одной ориентации
    cross_left = (px - lx) * dy_left - (py - ly) * dx_left
    cross_right = (px - rx) * dy_right - (py - ry) * dx_right
    # Для клина «между» лучами: точка справа от левого луча и слева от правого (или наоборот в зависимости от обхода)
    # Левый луч уходит влево-вперёд, правый вправо-вперёд. «Внутри» = P справа от левого луча и слева от правого.
    return cross_left <= 0 and cross_right >= 0


def _suction_rect_corners_world(
    center_x: float, center_y: float, angle_deg: float,
    suction_width: float, suction_length: float, suction_offset: float,
) -> list[tuple[float, float]]:
    """Четыре угла валика в мировых координатах (для проверки «внутри»)."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    # Локально: левый верх = (-suction_offset - suction_width, -suction_length/2), и т.д.
    corners_local = [
        (-suction_offset - suction_width, -suction_length / 2),
        (-suction_offset, -suction_length / 2),
        (-suction_offset, suction_length / 2),
        (-suction_offset - suction_width, suction_length / 2),
    ]
    out = []
    for lx, ly in corners_local:
        wx = center_x + lx * cos_a - ly * sin_a
        wy = center_y + lx * sin_a + ly * cos_a
        out.append((wx, wy))
    return out


def _point_in_rotated_rect(px: float, py: float, corners: list[tuple[float, float]]) -> bool:
    """Точка внутри четырёхугольника (одна и та же сторона от всех рёбер)."""
    n = len(corners)
    sign = None
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        if abs(cross) < 1e-9:
            continue
        if sign is None:
            sign = 1 if cross > 0 else -1
        elif (1 if cross > 0 else -1) != sign:
            return False
    return True


def _point_behind_suction(
    px: float, py: float,
    center_x: float, center_y: float, angle_deg: float,
    suction_offset: float, suction_width: float,
) -> bool:
    """Точка позади валика (за задней гранью по направлению движения)."""
    fx, fy = _forward_dir(angle_deg)
    # Задняя грань в локальных: x = -suction_offset - suction_width
    back_local_x = -suction_offset - suction_width
    back_center_x = center_x + back_local_x * fx
    back_center_y = center_y + back_local_x * fy
    dot = (px - back_center_x) * fx + (py - back_center_y) * fy
    return dot <= 0


def update_visits(
    agent: object,
    prev_x: float, prev_y: float, prev_angle: float,
    visit_map: object,
    body_radius_px: float,
    origin_x: int,
    origin_y: int,
    moved_forward: bool,
) -> None:
    """
    Обновить карту посещений после шага движения.
    Считаются только ячейки, которые были в клине перед валиком и стали «внутри или позади» валика;
    учитывается только движение вперёд (задним ходом не засчитывается).
    """
    if not moved_forward or not visit_map.has_map:
        return
    x = getattr(agent, "x", 0.0)
    y = getattr(agent, "y", 0.0)
    angle = getattr(agent, "angle", 0.0)
    center_x = origin_x + x
    center_y = origin_y + y
    prev_center_x = origin_x + prev_x
    prev_center_y = origin_y + prev_y

    sw, sl, so = _suction_geometry(body_radius_px)
    corners_now = _suction_rect_corners_world(center_x, center_y, angle, sw, sl, so)
    corners_prev = _suction_rect_corners_world(prev_center_x, prev_center_y, prev_angle, sw, sl, so)
    left_prev, right_prev = _suction_front_ends_world(prev_center_x, prev_center_y, prev_angle, sw, sl, so)

    cell_px = visit_map.cell_px
    left_px = visit_map.left_px
    top_px = visit_map.top_px
    grid_w = visit_map.grid_w
    grid_h = visit_map.grid_h

    # Ограничиваем поиск bounding-box'ом вокруг текущей и предыдущей позиции агента.
    # Все значимые ячейки (клин + валик) находятся в радиусе so+sw+sl от центра.
    margin = so + sw + sl + cell_px
    min_wx = min(center_x, prev_center_x) - margin
    max_wx = max(center_x, prev_center_x) + margin
    min_wy = min(center_y, prev_center_y) - margin
    max_wy = max(center_y, prev_center_y) + margin

    i_min = max(0, int((min_wx - left_px) / cell_px))
    i_max = min(grid_w - 1, int((max_wx - left_px) / cell_px))
    j_min = max(0, int((min_wy - top_px) / cell_px))
    j_max = min(grid_h - 1, int((max_wy - top_px) / cell_px))

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if not visit_map.is_reachable(i, j):
                continue
            cx = left_px + i * cell_px + cell_px / 2.0
            cy = top_px + j * cell_px + cell_px / 2.0

            now_inside = _point_in_rotated_rect(cx, cy, corners_now)
            now_behind = _point_behind_suction(cx, cy, center_x, center_y, angle, so, sw)
            prev_in_wedge = _point_in_wedge(cx, cy, left_prev, right_prev, prev_angle)
            prev_inside = _point_in_rotated_rect(cx, cy, corners_prev)
            prev_behind = _point_behind_suction(cx, cy, prev_center_x, prev_center_y, prev_angle, so, sw)

            if (now_inside or now_behind) and prev_in_wedge and not (prev_inside or prev_behind):
                visit_map.increment(i, j)
