"""
Генерация графиков из checkpoint (по умолчанию training/checkpoints/last.pt).

Пишет:
  training/plots/last.png
  training/plots/delta-last.png
  training/plots/extend/*
  training/plots/extend/architecture.png  — схема PolicyNet (слева направо)

Запуск из корня проекта:
  python training/render_plots_from_checkpoint.py
  python training/render_plots_from_checkpoint.py --ckpt training/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from training.history_plot_common import (
    PLOT_DELTA_LAST_EPISODES,
    draw_training_curves,
    history_plot_rows,
    load_history,
)


def _as_float_list(v) -> list[float]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def _extract_history_from_checkpoint(data: dict) -> dict | None:
    """Пытается вытащить историю прямо из checkpoint."""
    direct = data.get("training_history")
    if isinstance(direct, dict):
        rewards = _as_float_list(direct.get("rewards"))
        visited = _as_float_list(direct.get("visited_cells"))
        losses = _as_float_list(direct.get("losses"))
        value_losses = _as_float_list(direct.get("value_losses"))
        action_pct = direct.get("action_pct")
        if action_pct is not None:
            action_pct = list(action_pct)
        if rewards:
            return {
                "rewards": rewards,
                "visited": visited,
                "losses": losses,
                "value_losses": value_losses,
                "action_pct": action_pct,
            }

    # Иногда история хранится плоско в checkpoint
    rewards = _as_float_list(data.get("rewards"))
    if rewards:
        action_pct = data.get("action_pct")
        if action_pct is not None:
            action_pct = list(action_pct)
        return {
            "rewards": rewards,
            "visited": _as_float_list(data.get("visited_cells")),
            "losses": _as_float_list(data.get("losses")),
            "value_losses": _as_float_list(data.get("value_losses")),
            "action_pct": action_pct,
        }
    return None


def _smooth(y: list[float], w: int) -> list[float]:
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


def _save_main_training_plots(hist: dict, out_dir: Path, *, step: int, delta_n: int, smooth_cap: int, dpi: int) -> None:
    rewards = hist.get("rewards", [])
    n = len(rewards)
    if n == 0:
        print("История пуста: last/delta-last не построены.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = history_plot_rows(hist)
    x_full = list(range(1, n + 1))
    w_full = max(1, min(smooth_cap, max(1, n // 4)))

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes)
    draw_training_curves(
        axes,
        x_full,
        hist["rewards"],
        hist["visited"],
        hist["losses"],
        hist["value_losses"],
        hist.get("action_pct"),
        w_full,
        f"PPO Обучение (шаг {step})",
        n_rows,
    )
    (out_dir / "last.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "last.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    k = min(max(1, delta_n), n)
    start = n - k
    x_tail = list(range(start + 1, n + 1))
    w_tail = max(1, min(smooth_cap, max(1, k // 4)))

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes)
    ap = hist.get("action_pct")
    ap_seg = ap[start:n] if ap else None
    draw_training_curves(
        axes,
        x_tail,
        hist["rewards"][start:n],
        hist["visited"][start:n],
        hist["losses"][start:n],
        hist["value_losses"][start:n],
        ap_seg,
        w_tail,
        f"PPO Обучение (шаг {step}) — последние {k} эпизодов",
        n_rows,
    )
    fig.savefig(out_dir / "delta-last.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Сохранены: {out_dir / 'last.png'}")
    print(f"Сохранены: {out_dir / 'delta-last.png'}")


def _infer_dims_from_state(state: dict) -> tuple[int, int, int]:
    """(obs_dim, hidden_size, n_actions) из весов encoder / head."""
    enc0 = state.get("encoder.0.weight")
    if isinstance(enc0, torch.Tensor) and enc0.dim() == 2:
        h, d = int(enc0.shape[0]), int(enc0.shape[1])
    else:
        d, h = 6, 128
    head_out = state.get("head.2.weight")
    if isinstance(head_out, torch.Tensor) and head_out.dim() == 2:
        n_act = int(head_out.shape[0])
    else:
        n_act = 8
    return d, h, n_act


def _count_encoder_linear_layers(state: dict) -> int:
    n = 0
    for k, v in state.items():
        if k.startswith("encoder.") and k.endswith(".weight") and isinstance(v, torch.Tensor):
            n += 1
    return max(1, n)


def _count_mlp_extra_linear(state: dict) -> int:
    return sum(
        1
        for k, v in state.items()
        if k.startswith("mlp_extra.") and k.endswith(".weight") and isinstance(v, torch.Tensor)
    )


def _build_model_config(data: dict, state: dict) -> dict:
    cfg: dict = dict(data.get("config") or {})
    d, h, n_act = _infer_dims_from_state(state)
    cfg.setdefault("obs_dim", d)
    cfg.setdefault("hidden_size", h)
    cfg.setdefault("n_actions", n_act)
    if "use_gru" not in cfg:
        cfg["use_gru"] = any(k.startswith("rnn.") for k in state)
    if "encoder_layers" not in cfg:
        cfg["encoder_layers"] = _count_encoder_linear_layers(state)
    return cfg


def _obs_channel_caption(obs_dim: int) -> str:
    if obs_dim == 6:
        return (
            "каналы:\n"
            "• IR прямо\n"
            "• IR +30°\n"
            "• IR −30°\n"
            "• sin(θ)\n"
            "• cos(θ)\n"
            "• encoder Δ (норм.)"
        )
    return f"вектор размера {obs_dim}"


def _save_architecture_diagram(cfg: dict, state: dict, out_path: Path, *, dpi: int) -> None:
    """Схема PolicyNet: слева направо (размерности из config + уточнение по state_dict)."""
    obs_d = int(cfg.get("obs_dim", 6))
    h = int(cfg.get("hidden_size", 128))
    n_act = int(cfg.get("n_actions", 8))
    use_gru = bool(cfg.get("use_gru", True))
    enc_l = int(cfg.get("encoder_layers", 2))
    mx = _count_mlp_extra_linear(state)

    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f6f8")
    ax.set_facecolor("#f6f6f8")

    def _box(
        cx: float,
        cy: float,
        w: float,
        hgt: float,
        title: str,
        body: str,
        *,
        fc: str = "#e4e6ef",
        ec: str = "#2a2a32",
    ) -> None:
        patch = FancyBboxPatch(
            (cx - w / 2, cy - hgt / 2),
            w,
            hgt,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.35,
        )
        ax.add_patch(patch)
        ax.text(
            cx,
            cy + hgt * 0.12,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1a1a20",
        )
        ax.text(
            cx,
            cy - hgt * 0.08,
            body,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#333340",
            linespacing=1.35,
        )

    def _arrow(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        arr = FancyArrowPatch(
            p0,
            p1,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            color="#3d3d48",
            linewidth=1.35,
            clip_on=False,
        )
        ax.add_patch(arr)

    w_main = 0.145
    h_main = 0.34
    x_obs, y_obs = 0.09, 0.5
    x_enc, y_enc = 0.28, 0.5
    x_core, y_core = 0.48, 0.5
    x_fork, y_fork = 0.64, 0.5
    x_act, y_act = 0.86, 0.72
    x_val, y_val = 0.86, 0.28

    obs_body = f"размерность {obs_d}\n\n{_obs_channel_caption(obs_d)}"
    _box(x_obs, y_obs, w_main + 0.02, h_main + 0.12, "Вход: наблюдение", obs_body, fc="#d8e8f0")

    enc_body = (
        f"Sequential: {enc_l}× (Linear + ReLU)\n"
        f"тензор на выходе: ({h},)"
    )
    _box(x_enc, y_enc, w_main, h_main, "Encoder", enc_body, fc="#e8e4f4")

    if use_gru:
        core_title = "Память (GRU)"
        core_body = (
            "GRU, 1 слой, batch_first\n"
            f"на каждом шаге: ({h},) → ({h},)\n"
            "скрытое состояние h_t"
        )
    else:
        core_title = "После encoder"
        core_body = "Без GRU: mlp_extra\n" + (
            f"{mx}× (Linear + ReLU)\n{h} → … → {h}"
            if mx > 0
            else "Identity — доп. слоёв нет"
        )
    _box(x_core, y_core, w_main, h_main, core_title, core_body, fc="#f0e8dc")

    _box(
        x_fork,
        y_fork,
        0.07,
        0.14,
        "h",
        "один вектор\nдля двух голов",
        fc="#dfe8df",
    )

    act_body = (
        "2× Linear + ReLU\n"
        f"{h} → {h} → {n_act}\n"
        "→ logits → softmax\n(actor)"
    )
    _box(x_act, y_act, w_main, h_main + 0.06, "Голова: политика", act_body, fc="#dde8f7")

    val_body = (
        "2× Linear + ReLU\n"
        f"{h} → {h} → 1\n"
        "→ value V(s)\n(critic)"
    )
    _box(x_val, y_val, w_main, h_main + 0.06, "Голова: ценность", val_body, fc="#f5e0e0")

    def _right(cx: float, cy: float, w: float) -> tuple[float, float]:
        return (cx + w / 2, cy)

    def _left(cx: float, cy: float, w: float) -> tuple[float, float]:
        return (cx - w / 2, cy)

    wo = w_main + 0.02
    _arrow(_right(x_obs, y_obs, wo), _left(x_enc, y_enc, w_main))
    _arrow(_right(x_enc, y_enc, w_main), _left(x_core, y_core, w_main))
    _arrow(_right(x_core, y_core, w_main), _left(x_fork, y_fork, 0.07))

    wf = 0.07
    _arrow(_right(x_fork, y_fork, wf), _left(x_act, y_act, w_main))
    _arrow(_right(x_fork, y_fork, wf), _left(x_val, y_val, w_main))

    algo = str(cfg.get("algo", "ppo"))
    if use_gru:
        footer = (
            f"PolicyNet ({algo.upper()})  ·  hidden_size={h}  ·  encoder_layers={enc_l}  ·  "
            f"use_gru=True  ·  n_actions={n_act}"
        )
    else:
        footer = (
            f"PolicyNet ({algo.upper()})  ·  hidden_size={h}  ·  encoder_layers={enc_l}  ·  "
            f"use_gru=False  ·  mlp_extra_linear={mx}  ·  n_actions={n_act}"
        )
    ax.text(
        0.5,
        0.06,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555560",
    )
    fig.suptitle("Архитектура (поток данных, только представления)", fontsize=13, y=0.96)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.patch.get_facecolor())
    plt.close(fig)


def _floating_params(state: dict) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
            out.append((name, tensor.detach().cpu()))
    return out


def _save_param_stats_plot(params: list[tuple[str, torch.Tensor]], out_path: Path, *, dpi: int) -> None:
    names = [n for n, _ in params]
    means = [float(t.abs().mean().item()) for _, t in params]
    stds = [float(t.std(unbiased=False).item()) for _, t in params]

    idx = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(idx, means, color="C0")
    ax1.set_ylabel("|w| mean")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Параметры: средний модуль")

    ax2.bar(idx, stds, color="C1")
    ax2.set_ylabel("std")
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Параметры: стандартное отклонение")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_hist_grid(params: list[tuple[str, torch.Tensor]], out_path: Path, *, dpi: int, max_panels: int = 16) -> None:
    chosen = params[:max_panels]
    n = len(chosen)
    if n == 0:
        return
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows))
    axes_arr = np.ravel(axes) if isinstance(axes, np.ndarray) else np.array([axes])

    for ax in axes_arr[n:]:
        ax.axis("off")

    for i, (name, t) in enumerate(chosen):
        ax = axes_arr[i]
        x = t.flatten().numpy()
        ax.hist(x, bins=80, color="C0", alpha=0.85)
        ax.set_title(name, fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Распределение весов по параметрам (hist)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_linear_heatmaps(params: list[tuple[str, torch.Tensor]], out_path: Path, *, dpi: int, max_panels: int = 12) -> None:
    mats = [(n, t) for n, t in params if t.dim() == 2][:max_panels]
    if not mats:
        return
    cols = 3
    rows = math.ceil(len(mats) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows))
    axes_arr = np.ravel(axes) if isinstance(axes, np.ndarray) else np.array([axes])
    for ax in axes_arr[len(mats):]:
        ax.axis("off")

    for i, (name, t) in enumerate(mats):
        ax = axes_arr[i]
        arr = t.numpy()
        vmax = max(1e-9, float(np.quantile(np.abs(arr), 0.99)))
        im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("in")
        ax.set_ylabel("out")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Теплокарты матриц весов", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_gru_gate_plots(state: dict, out_path: Path, *, dpi: int) -> None:
    key = "rnn.weight_ih_l0"
    if key not in state or not isinstance(state[key], torch.Tensor):
        return
    w_ih = state[key].detach().cpu()
    chunks = torch.chunk(w_ih, 3, dim=0)
    gate_names = ["reset", "update", "new"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for i, (gname, g) in enumerate(zip(gate_names, chunks)):
        arr = g.numpy()
        vmax = max(1e-9, float(np.quantile(np.abs(arr), 0.99)))
        im = axes[i, 0].imshow(arr, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        axes[i, 0].set_title(f"{key} [{gname}]")
        axes[i, 0].set_xlabel("in")
        axes[i, 0].set_ylabel("out")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

        x = g.flatten().numpy()
        axes[i, 1].hist(x, bins=80, color=f"C{i}", alpha=0.85)
        axes[i, 1].set_title(f"hist [{gname}]")
        axes[i, 1].grid(True, alpha=0.25)

    fig.suptitle("GRU gates (input weights)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_action_history_plot(hist: dict, out_path: Path, *, dpi: int) -> None:
    ap = hist.get("action_pct")
    n = len(hist.get("rewards", []))
    if not ap or n == 0:
        return
    ap = list(ap)[:n]
    n_act = len(ap[0]) if ap else 0
    if n_act == 0:
        return

    x = list(range(1, len(ap) + 1))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(n_act):
        y = [float(row[i]) for row in ap]
        ax.plot(x, y, alpha=0.15, color=f"C{i}")
        ax.plot(x, _smooth(y, max(1, len(y) // 20)), color=f"C{i}", label=f"act#{i}")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Доля действия, %")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_title("Доли действий (raw + сглажено)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_extend_plots(
    state: dict,
    hist: dict | None,
    out_dir: Path,
    *,
    cfg: dict,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_architecture_diagram(cfg, state, out_dir / "architecture.png", dpi=dpi)

    params = _floating_params(state)
    if not params:
        print("В checkpoint нет float-параметров для extend-графиков.")
        (out_dir / "summary.txt").write_text(
            "Extend plots (только схема)\narchitecture.png — PolicyNet\n",
            encoding="utf-8",
        )
        print(f"Сохранена схема: {out_dir / 'architecture.png'}")
        return

    _save_param_stats_plot(params, out_dir / "weights_stats.png", dpi=dpi)
    _save_hist_grid(params, out_dir / "weights_hist_grid.png", dpi=dpi)
    _save_linear_heatmaps(params, out_dir / "weights_heatmaps.png", dpi=dpi)
    _save_gru_gate_plots(state, out_dir / "gru_gates.png", dpi=dpi)
    if hist:
        _save_action_history_plot(hist, out_dir / "action_pct_history.png", dpi=dpi)

    report_path = out_dir / "summary.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Extend plots generated from checkpoint\n")
        f.write("architecture.png — схема PolicyNet (поток данных)\n")
        f.write(f"parameters: {len(params)}\n")
        for name, t in params:
            arr = t.flatten()
            f.write(
                f"{name}: shape={tuple(t.shape)}, mean={arr.mean().item():.6f}, "
                f"std={arr.std(unbiased=False).item():.6f}, "
                f"min={arr.min().item():.6f}, max={arr.max().item():.6f}\n"
            )

    print(f"Сохранены extend-графики в: {out_dir} (в т.ч. architecture.png)")


def main() -> None:
    p = argparse.ArgumentParser(description="Рендер графиков из checkpoint last.pt")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=_root / "training" / "checkpoints" / "last.pt",
        help="Путь к checkpoint (.pt)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_root / "training" / "plots",
        help="Каталог для last.png/delta-last.png и extend/",
    )
    p.add_argument(
        "--delta-n",
        type=int,
        default=PLOT_DELTA_LAST_EPISODES,
        help="Окно эпизодов для delta-last.png",
    )
    p.add_argument("--smooth-cap", type=int, default=50, help="Макс. окно сглаживания")
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        print(f"Checkpoint не найден: {ckpt_path}")
        sys.exit(1)

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy_state = data.get("policy")
    if not isinstance(policy_state, dict):
        print("В checkpoint не найден state_dict в ключе 'policy'.")
        sys.exit(1)

    step = int(data.get("episode", 0) or data.get("step", 0) or 0)

    hist = _extract_history_from_checkpoint(data)
    if hist is None:
        hist_path = ckpt_path.parent / "training_history.pt"
        if hist_path.is_file():
            hist = load_history(hist_path)
            print(f"История взята из: {hist_path}")
        else:
            print("История не найдена ни в checkpoint, ни рядом как training_history.pt.")

    out_dir = args.out_dir.resolve()
    model_cfg = _build_model_config(data, policy_state)
    if hist:
        _save_main_training_plots(
            hist,
            out_dir,
            step=step,
            delta_n=max(1, args.delta_n),
            smooth_cap=max(1, args.smooth_cap),
            dpi=args.dpi,
        )

    _save_extend_plots(policy_state, hist, out_dir / "extend", cfg=model_cfg, dpi=args.dpi)


if __name__ == "__main__":
    main()

