"""
Анимация обучения из checkpoints/training_history.pt → last.gif и delta-last.gif
(те же 6 панелей, что и train_torch._save_training_plot; посещения — ячейки, не %).

Запуск из корня проекта:
  python training/history_to_gif.py
  python training/history_to_gif.py path/to/training_history.pt --out-dir training/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from training.history_plot_common import (
    PLOT_DELTA_LAST_EPISODES,
    draw_training_curves,
    history_plot_rows,
    load_history,
    _window_for_n,
)


def _episode_end_indices(n: int, stride: int) -> list[int]:
    if n <= 0:
        return []
    ends = list(range(stride, n + 1, stride))
    if n not in ends:
        ends.append(n)
    ends = sorted(set(ends))
    if ends[0] != 1:
        ends.insert(0, 1)
    return ends


def render_gif_last(
    hist: dict,
    out_path: Path,
    *,
    target_frames: int,
    stride_override: int | None,
    fps: float,
    smooth_cap: int,
    dpi: int,
) -> None:
    rewards = hist["rewards"]
    n = len(rewards)
    if n == 0:
        print("История пуста, GIF не создан.")
        return

    stride = stride_override if stride_override is not None else max(1, n // max(1, target_frames))
    ends = _episode_end_indices(n, stride)

    ap_all = hist["action_pct"]
    n_rows = history_plot_rows(hist)

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for e in ends:
            w = _window_for_n(e, smooth_cap)
            x = list(range(1, e + 1))
            draw_training_curves(
                axes,
                x,
                rewards[:e],
                hist["visited"][:e],
                hist["losses"][:e],
                hist["value_losses"][:e],
                ap_all,
                w,
                f"PPO Обучение (эпизод {e} / {n})",
                n_rows,
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"Записано: {out_path} ({len(ends)} кадров, stride~{stride})")


def render_gif_delta(
    hist: dict,
    out_path: Path,
    *,
    delta_n: int,
    target_frames: int,
    stride_override: int | None,
    fps: float,
    smooth_cap: int,
    dpi: int,
) -> None:
    rewards = hist["rewards"]
    n = len(rewards)
    if n == 0:
        print("История пуста, delta GIF не создан.")
        return

    stride = stride_override if stride_override is not None else max(1, n // max(1, target_frames))
    ends = _episode_end_indices(n, stride)

    ap_all = hist["action_pct"]
    n_rows = history_plot_rows(hist)

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=dpi):
        for e in ends:
            k = min(delta_n, e)
            start = e - k
            w = _window_for_n(k, smooth_cap)
            x = list(range(start + 1, e + 1))
            tail_title = f"последние {k} эп." if k < e else f"эпизоды 1–{e}"
            ap_seg = ap_all[start:e] if ap_all else None
            draw_training_curves(
                axes,
                x,
                rewards[start:e],
                hist["visited"][start:e],
                hist["losses"][start:e],
                hist["value_losses"][start:e],
                ap_seg,
                w,
                f"PPO Обучение (эпизод {e} / {n}) — {tail_title}",
                n_rows,
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"Записано: {out_path} ({len(ends)} кадров, окно <={delta_n}, stride~{stride})")


def main() -> None:
    p = argparse.ArgumentParser(description="GIF-анимация из training_history.pt")
    p.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=_root / "training" / "checkpoints" / "training_history.pt",
        help="Путь к training_history.pt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_root / "training" / "plots",
        help="Каталог для last.gif и delta-last.gif",
    )
    p.add_argument("--fps", type=float, default=8.0, help="Кадров в секунду в GIF")
    p.add_argument(
        "--target-frames",
        type=int,
        default=180,
        help="Целевое число кадров (stride подбирается как n / target_frames)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Фиксированный шаг по эпизодам между кадрами (перекрывает target-frames)",
    )
    p.add_argument(
        "--smooth-cap",
        type=int,
        default=50,
        help="Максимум окна сглаживания (как в обучении, не больше n//4)",
    )
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument(
        "--delta-n",
        type=int,
        default=PLOT_DELTA_LAST_EPISODES,
        help="Ширина окна для delta-last.gif",
    )
    p.add_argument("--no-delta", action="store_true", help="Не строить delta-last.gif")
    p.add_argument("--no-last", action="store_true", help="Не строить last.gif")
    args = p.parse_args()

    if not args.history.is_file():
        print(f"Файл не найден: {args.history}")
        sys.exit(1)

    hist = load_history(args.history)
    n = len(hist["rewards"])
    print(f"Загружено эпизодов: {n}")

    if not args.no_last:
        render_gif_last(
            hist,
            args.out_dir / "last.gif",
            target_frames=args.target_frames,
            stride_override=args.stride,
            fps=args.fps,
            smooth_cap=args.smooth_cap,
            dpi=args.dpi,
        )
    if not args.no_delta:
        render_gif_delta(
            hist,
            args.out_dir / "delta-last.gif",
            delta_n=args.delta_n,
            target_frames=args.target_frames,
            stride_override=args.stride,
            fps=args.fps,
            smooth_cap=args.smooth_cap,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
