"""
Общая загрузка training_history.pt и отрисовка кривых на matplotlib axes.
Без выбора backend matplotlib — подходит и для TkAgg, и для Agg (GIF).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

# Синхронно с train_torch.py
ACTION_LABELS = (
    "вперёд", "назад", "влево", "вправо",
    "вперёд+влево", "вперёд+вправо", "назад+влево", "назад+вправо",
)
PLOT_DELTA_LAST_EPISODES = 500


def smooth(y: list[float], w: int) -> list[float]:
    """Скользящее среднее; окно не больше длины ряда (иначе короткие delta-окна не сглаживались)."""
    if len(y) == 0:
        return []
    if w <= 1:
        return list(y)
    w = min(w, len(y))
    if w <= 1:
        return list(y)
    out: list[float] = []
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        out.append(sum(y[lo : i + 1]) / (i - lo + 1))
    return out


def _normalize_action_pct(
    action_pct: list | None, m: int
) -> list[tuple[float, ...]] | None:
    if not action_pct or m == 0:
        return None
    if len(action_pct) < m:
        return None
    out: list[tuple[float, ...]] = []
    for row in action_pct[:m]:
        if hasattr(row, "tolist"):
            row = row.tolist()
        out.append(tuple(float(x) for x in row))
    return out


def _window_for_n(n: int, cap: int) -> int:
    return min(cap, max(1, n // 4))


def load_history(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    rewards = [float(x) for x in data.get("rewards", [])]
    visited = [float(x) for x in data.get("visited_pct", [])]
    losses = [float(x) for x in data.get("losses", [])]
    vloss = [float(x) for x in data.get("value_losses", [])]
    ap = data.get("action_pct")
    if ap is not None:
        ap = list(ap)
    return {
        "rewards": rewards,
        "visited_pct": visited,
        "losses": losses,
        "value_losses": vloss,
        "action_pct": ap,
    }


def history_plot_rows(hist: dict) -> int:
    n = len(hist.get("rewards", []))
    return 6 if _normalize_action_pct(hist.get("action_pct"), n) else 4


def draw_training_curves(
    axes: np.ndarray,
    episode_x: list[int],
    r: list[float],
    v: list[float],
    pl: list[float],
    vl: list[float],
    ap_raw: list | None,
    smooth_window: int,
    suptitle: str,
    n_rows: int,
) -> None:
    m = len(r)
    ap = _normalize_action_pct(ap_raw, m)

    ax_reward, ax_visit, ax_ploss, ax_vloss = axes[0], axes[1], axes[2], axes[3]
    for ax in axes[:4]:
        ax.clear()

    ax_reward.plot(episode_x, r, alpha=0.3, color="C0")
    ax_reward.plot(episode_x, smooth(r, smooth_window), color="C0", label="reward (среднее)")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="upper right", fontsize=8)
    ax_reward.grid(True, alpha=0.3)

    ax_visit.plot(episode_x, v, alpha=0.3, color="C1")
    ax_visit.plot(episode_x, smooth(v, smooth_window), color="C1", label="посещено % (среднее)")
    ax_visit.set_ylabel("Посещено %")
    ax_visit.legend(loc="upper right", fontsize=8)
    ax_visit.grid(True, alpha=0.3)

    ax_ploss.plot(episode_x, pl, alpha=0.3, color="C2")
    ax_ploss.plot(episode_x, smooth(pl, smooth_window), color="C2", label="policy loss")
    ax_ploss.set_ylabel("Policy Loss")
    ax_ploss.legend(loc="upper right", fontsize=8)
    ax_ploss.grid(True, alpha=0.3)

    if vl:
        ax_vloss.plot(episode_x, vl, alpha=0.3, color="C3")
        ax_vloss.plot(episode_x, smooth(vl, smooth_window), color="C3", label="value loss")
    ax_vloss.set_ylabel("Value Loss")
    ax_vloss.legend(loc="upper right", fontsize=8)
    ax_vloss.grid(True, alpha=0.3)

    if n_rows == 6 and ap and m > 0:
        ax_act, ax_ent = axes[4], axes[5]
        ax_act.clear()
        ax_ent.clear()
        n_act = len(ap[0])
        max_entropy = math.log(n_act) if n_act > 1 else 1.0

        for i in range(n_act):
            label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else str(i)
            ser = [a[i] for a in ap]
            ax_act.plot(episode_x, ser, alpha=0.25, color=f"C{i}")
            ax_act.plot(
                episode_x,
                smooth(ser, smooth_window),
                color=f"C{i}",
                alpha=0.9,
                label=label,
            )
        ax_act.set_ylabel("Доля (сэмпл.) %")
        ax_act.legend(loc="upper right", fontsize=7, ncol=2)
        ax_act.grid(True, alpha=0.3)
        ax_act.set_ylim(-5, 105)

        def _ep_entropy(counts: tuple[float, ...]) -> float:
            h = 0.0
            for p in counts:
                frac = p / 100.0
                if frac > 1e-9:
                    h -= frac * math.log(frac)
            return h / max_entropy

        entropy_vals = [_ep_entropy(a) for a in ap]
        ax_ent.plot(episode_x, entropy_vals, alpha=0.3, color="C4")
        ax_ent.plot(episode_x, smooth(entropy_vals, smooth_window), color="C4", label="энтропия (норм.)")
        ax_ent.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="макс")
        ax_ent.set_ylabel("Энтропия")
        ax_ent.set_xlabel("Эпизод")
        ax_ent.legend(loc="lower right", fontsize=7)
        ax_ent.grid(True, alpha=0.3)
        ax_ent.set_ylim(-0.05, 1.1)
    else:
        ax_vloss.set_xlabel("Эпизод")

    fig = axes[0].figure
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
