"""
Интерактивный просмотр динамики весов по чекпоинтам из training/checkpoints/history/.

Режимы (как extend/weights_*.png): stats, GRU gates, heatmaps, hist.

Два этапа при старте (фоновый поток + пул потоков для .pt и Agg; окно Tk не блокируется):
  1) Параллельная загрузка policy_ep_*.pt → снимки весов в numpy.
  2) Параллельный оффлайн-рендер по чекпоинтам (каждый поток — свой Figure, все 4 режима).

Прогресс: строка статуса + полоса ttk.Progressbar; обновление через queue + root.after.

Смена эпизода/режима: только imshow готовой картинки (без пересчёта matplotlib).

Кеш предрендера (по умолчанию вкл.): training/plots/history/cache/*.npz
  — совпадение mtime/size .pt, render_dpi и figsize; иначе перерисовка и запись.

Запуск:
  python training/checkpoint_weights_viewer.py
  python training/checkpoint_weights_viewer.py --render-dpi 80   # меньше RAM
  python training/checkpoint_weights_viewer.py --workers 8       # параллель загрузка+рендер
  python training/checkpoint_weights_viewer.py --no-cache        # без дискового кеша
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

_EP_RE = re.compile(r"policy_ep_(\d+)\.pt$", re.IGNORECASE)

FIGSIZE_INCH = (11.0, 7.5)
MODES = ("stats", "gru", "heatmaps", "hist")

# Версия формата кеша (поля npz); при смене логики рендера — увеличить.
_TILE_CACHE_FORMAT = 1


def _default_tile_cache_dir() -> Path:
    return _root / "training" / "plots" / "history" / "cache"


def _tile_cache_npz_path(cache_dir: Path, pt_path: Path, dpi: int) -> Path:
    return cache_dir / f"{pt_path.stem}_dpi{dpi}.npz"


def _pt_stat_fingerprint(pt_path: Path) -> tuple[int, int]:
    st = pt_path.stat()
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return (int(mtime_ns), int(st.st_size))


def _try_load_tile_cache(cache_path: Path, pt_path: Path, dpi: int) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        mtime_ns, size = _pt_stat_fingerprint(pt_path)
        with np.load(cache_path, allow_pickle=False) as z:
            if int(z["format"][0]) != _TILE_CACHE_FORMAT:
                return None
            if int(z["render_dpi"][0]) != dpi:
                return None
            if abs(float(z["fig_w"][0]) - FIGSIZE_INCH[0]) > 1e-6 or abs(
                float(z["fig_h"][0]) - FIGSIZE_INCH[1]
            ) > 1e-6:
                return None
            if int(z["pt_mtime_ns"][0]) != mtime_ns or int(z["pt_size"][0]) != size:
                return None
            out: dict[str, np.ndarray] = {}
            for m in MODES:
                a = np.asarray(z[m])
                if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
                    return None
                out[m] = a.copy()
            return out
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _save_tile_cache(
    cache_path: Path,
    row: dict[str, np.ndarray],
    pt_path: Path,
    dpi: int,
) -> None:
    """Пишем во временный файл, имя обязано заканчиваться на .npz — иначе numpy
    дописывает ещё один .npz (получится *.npz.tmp.npz), и os.replace не находит файл.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mtime_ns, size = _pt_stat_fingerprint(pt_path)
    tmp = cache_path.parent / f"{cache_path.stem}._tmp_{os.getpid()}.npz"
    arrays = {m: np.asarray(row[m], dtype=np.uint8) for m in MODES}
    try:
        np.savez_compressed(
            tmp,
            format=np.array([_TILE_CACHE_FORMAT], dtype=np.int64),
            render_dpi=np.array([dpi], dtype=np.int64),
            fig_w=np.array([FIGSIZE_INCH[0]], dtype=np.float64),
            fig_h=np.array([FIGSIZE_INCH[1]], dtype=np.float64),
            pt_mtime_ns=np.array([mtime_ns], dtype=np.int64),
            pt_size=np.array([size], dtype=np.int64),
            **arrays,
        )
        os.replace(tmp, cache_path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _floating_params(state: dict) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
            out.append((name, tensor.detach().cpu()))
    return out


def _sort_checkpoint_paths(history_dir: Path) -> list[Path]:
    paths = list(history_dir.glob("policy_ep_*.pt"))

    def key(p: Path) -> tuple[int, str]:
        m = _EP_RE.search(p.name)
        return (int(m.group(1)), p.name) if m else (0, p.name)

    return sorted(paths, key=key)


def _episode_label(path: Path, data: dict) -> str:
    ep = data.get("episode")
    if ep is not None:
        return str(int(ep))
    m = _EP_RE.search(path.name)
    return m.group(1) if m else "?"


@dataclass(frozen=True)
class CheckpointSnapshot:
    path: Path
    episode: str
    param_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    gru_chunks: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    heatmap_names: tuple[str, ...]
    heatmap_arrays: tuple[np.ndarray, ...]
    hist_names: tuple[str, ...]
    hist_arrays: tuple[np.ndarray, ...]


def _snapshot_from_file(path: Path) -> CheckpointSnapshot:
    data = torch.load(path, map_location="cpu", weights_only=False)
    policy = data.get("policy")
    if not isinstance(policy, dict):
        raise TypeError(f"В {path.name} нет dict policy")
    ep = _episode_label(path, data)
    params = _floating_params(policy)
    if not params:
        raise ValueError(f"Нет float-параметров: {path.name}")

    names = tuple(n for n, _ in params)
    means = np.array([float(t.abs().mean().item()) for _, t in params], dtype=np.float32)
    stds = np.array([float(t.std(unbiased=False).item()) for _, t in params], dtype=np.float32)

    key = "rnn.weight_ih_l0"
    gru: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    if key in policy and isinstance(policy[key], torch.Tensor):
        w = policy[key].detach().float().cpu().numpy()
        parts = np.split(w.astype(np.float32), 3, axis=0)
        gru = (parts[0], parts[1], parts[2])
    else:
        gru = None

    hm = [(n, t.detach().float().cpu().numpy().astype(np.float32)) for n, t in params if t.dim() == 2][:12]
    heatmap_names = tuple(n for n, _ in hm)
    heatmap_arrays = tuple(a for _, a in hm)

    hp = params[:16]
    hist_names = tuple(n for n, _ in hp)
    hist_arrays = tuple(
        t.detach().float().cpu().numpy().astype(np.float32).ravel() for _, t in hp
    )

    return CheckpointSnapshot(
        path=path,
        episode=ep,
        param_names=names,
        means=means,
        stds=stds,
        gru_chunks=gru,
        heatmap_names=heatmap_names,
        heatmap_arrays=heatmap_arrays,
        hist_names=hist_names,
        hist_arrays=hist_arrays,
    )


def _snapshots_compatible(a: CheckpointSnapshot, b: CheckpointSnapshot) -> str | None:
    if a.param_names != b.param_names:
        return "разный набор параметров (param_names)"
    if a.heatmap_names != b.heatmap_names:
        return "разный набор heatmap-слоёв"
    if a.hist_names != b.hist_names:
        return "разный набор hist-слоёв (первые 16)"
    if (a.gru_chunks is None) != (b.gru_chunks is None):
        return "в одном чекпоинте есть GRU, в другом нет"
    return None


def _populate_stats(fig: Figure, snap: CheckpointSnapshot) -> None:
    fig.clear()
    names = snap.param_names
    idx = np.arange(len(names))
    ax1 = fig.add_subplot(211)
    ax1.bar(idx, snap.means, color="C0")
    ax1.set_ylabel("|w| mean")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Параметры: средний модуль")
    ax2 = fig.add_subplot(212)
    ax2.bar(idx, snap.stds, color="C1")
    ax2.set_ylabel("std")
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Параметры: стандартное отклонение")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    fig.suptitle(f"weights_stats  ·  эп. {snap.episode}", fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])


def _populate_gru(fig: Figure, snap: CheckpointSnapshot) -> None:
    fig.clear()
    key = "rnn.weight_ih_l0"
    if snap.gru_chunks is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Нет GRU (rnn.weight_ih_l0) в чекпоинтах", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"gru_gates  ·  эп. {snap.episode}", fontsize=11, y=1.02)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        return
    gate_names = ["reset", "update", "new"]
    for i, (gname, g) in enumerate(zip(gate_names, snap.gru_chunks)):
        arr = np.asarray(g, dtype=np.float64)
        vmax = max(1e-9, float(np.quantile(np.abs(arr), 0.99)))
        ax0 = fig.add_subplot(3, 2, 2 * i + 1)
        im = ax0.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax0.set_title(f"{key} [{gname}]")
        ax0.set_xlabel("in")
        ax0.set_ylabel("out")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        ax1 = fig.add_subplot(3, 2, 2 * i + 2)
        ax1.hist(arr.ravel(), bins=80, color=f"C{i}", alpha=0.85)
        ax1.set_title(f"hist [{gname}]")
        ax1.grid(True, alpha=0.25)
    fig.suptitle(f"gru_gates (input weights)  ·  эп. {snap.episode}", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])


def _populate_heatmaps(fig: Figure, snap: CheckpointSnapshot) -> None:
    fig.clear()
    if not snap.heatmap_arrays:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Нет 2D-весов", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"weights_heatmaps  ·  эп. {snap.episode}", fontsize=11, y=1.01)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        return
    cols = 3
    n = len(snap.heatmap_arrays)
    rows = math.ceil(n / cols)
    for i, (name, arr) in enumerate(zip(snap.heatmap_names, snap.heatmap_arrays)):
        a = np.asarray(arr, dtype=np.float64)
        ax = fig.add_subplot(rows, cols, i + 1)
        vmax = max(1e-9, float(np.quantile(np.abs(a), 0.99)))
        im = ax.imshow(a, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("in")
        ax.set_ylabel("out")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"weights_heatmaps  ·  эп. {snap.episode}", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])


def _populate_hist_grid(fig: Figure, snap: CheckpointSnapshot) -> None:
    fig.clear()
    if not snap.hist_arrays:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Нет параметров", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"weights_hist_grid  ·  эп. {snap.episode}", fontsize=11, y=1.01)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        return
    cols = 4
    n = len(snap.hist_arrays)
    rows = math.ceil(n / cols)
    for i, (name, flat) in enumerate(zip(snap.hist_names, snap.hist_arrays)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(np.asarray(flat, dtype=np.float64), bins=80, color="C0", alpha=0.85)
        ax.set_title(name, fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"weights_hist_grid  ·  эп. {snap.episode}", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])


def _populate_figure(fig: Figure, mode: str, snap: CheckpointSnapshot) -> None:
    if mode == "stats":
        _populate_stats(fig, snap)
    elif mode == "gru":
        _populate_gru(fig, snap)
    elif mode == "heatmaps":
        _populate_heatmaps(fig, snap)
    else:
        _populate_hist_grid(fig, snap)


def _figure_to_rgb(fig: Figure) -> np.ndarray:
    """RGBA buffer Agg → RGB uint8, shape (H, W, 3), строка 0 = верх."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _render_checkpoint_tiles(
    snap: CheckpointSnapshot,
    dpi: int,
    *,
    cache_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Один чекпоинт × все режимы; свой Figure — безопасно из разных потоков (Agg).

    Второй элемент: True если строка взята с диска (кеш).
    """
    pt_path = snap.path.resolve()
    if cache_dir is not None:
        cpath = _tile_cache_npz_path(cache_dir.resolve(), pt_path, dpi)
        hit = _try_load_tile_cache(cpath, pt_path, dpi)
        if hit is not None:
            return hit, True

    row: dict[str, np.ndarray] = {}
    fig = Figure(figsize=FIGSIZE_INCH, dpi=dpi)
    try:
        for mode in MODES:
            _populate_figure(fig, mode, snap)
            row[mode] = _figure_to_rgb(fig)
        if cache_dir is not None:
            try:
                _save_tile_cache(
                    _tile_cache_npz_path(cache_dir.resolve(), pt_path, dpi),
                    row,
                    pt_path,
                    dpi,
                )
            except OSError:
                pass
        return row, False
    finally:
        plt.close(fig)


def _default_worker_count() -> int:
    return max(1, min(8, os.cpu_count() or 4))


def _rgb_ram_mib(tiles: list[dict[str, np.ndarray]]) -> float:
    b = 0
    for row in tiles:
        for arr in row.values():
            b += arr.nbytes
    return b / (1024 * 1024)


def _load_and_prerender_worker(
    paths: list[Path],
    render_dpi: int,
    progress_q: queue.Queue,
    *,
    max_workers: int,
    cache_dir: Path | None = None,
) -> None:
    """Фоновый поток: torch + Agg, без tk. Параллельная загрузка и рендер (ThreadPoolExecutor)."""
    try:
        n = len(paths)
        total_steps = n * (1 + len(MODES))
        cur = 0
        workers = max(1, min(max_workers, n))

        snapshots: list[CheckpointSnapshot | None] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_i = {ex.submit(_snapshot_from_file, paths[i]): i for i in range(n)}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                snapshots[i] = fut.result()
                progress_q.put(
                    (
                        "progress",
                        cur,
                        total_steps,
                        f"Загрузка .pt {cur + 1}/{n}: {paths[i].name}",
                    )
                )
                cur += 1

        snaps = [s for s in snapshots if s is not None]
        if len(snaps) != n:
            raise RuntimeError("внутренняя ошибка: не все снимки загружены")
        ref = snaps[0]
        for j in range(1, n):
            err = _snapshots_compatible(ref, snaps[j])
            if err:
                raise ValueError(f"{snaps[j].path.name}: несовместим с первым чекпоинтом ({err})")

        meta = [(s.episode, s.path.name) for s in snaps]
        total_r = n * len(MODES)

        tiles: list[dict[str, np.ndarray] | None] = [None] * n
        cache_hits = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_i = {
                ex.submit(
                    _render_checkpoint_tiles,
                    snaps[i],
                    render_dpi,
                    cache_dir=cache_dir,
                ): i
                for i in range(n)
            }
            done_ckpt = 0
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                row, from_cache = fut.result()
                tiles[i] = row
                if from_cache:
                    cache_hits += 1
                done_ckpt += 1
                tag = "Кеш" if from_cache else "Рендер"
                progress_q.put(
                    (
                        "progress",
                        cur,
                        total_steps,
                        f"{tag} {done_ckpt}/{n}: {snaps[i].path.name} (×{len(MODES)})",
                    )
                )
                cur += len(MODES)

        del snaps
        progress_q.put(
            ("done", [t for t in tiles if t is not None], meta, total_r, cache_hits),
        )
    except Exception as e:
        progress_q.put(("error", str(e)))


class WeightEvolutionViewer:
    def __init__(
        self,
        history_dir: Path,
        *,
        dpi: int = 100,
        render_dpi: int | None = None,
        max_workers: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.history_dir = history_dir.resolve()
        self.dpi = dpi
        self.render_dpi = render_dpi if render_dpi is not None else dpi
        self.max_workers = max_workers if max_workers is not None else _default_worker_count()
        self.cache_dir = cache_dir
        self.paths = _sort_checkpoint_paths(self.history_dir)
        self._tiles: list[dict[str, np.ndarray]] = []
        self._meta: list[tuple[str, str]] = []  # episode, filename
        self._progress_q: queue.Queue = queue.Queue()
        self._worker_done = False

        self.root = tk.Tk()
        self.root.title("Веса политики — предрендер в RAM")
        self.root.minsize(900, 700)

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(self.root)
        btn_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        tk.Label(btn_row, text="График:", font=("", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self._mode_var = tk.StringVar(value="stats")
        self._radios: list[tk.Radiobutton] = []
        labels = [
            ("stats", "|w| / std"),
            ("gru", "GRU gates"),
            ("heatmaps", "Теплокарты"),
            ("hist", "Гистограммы"),
        ]
        for val, text in labels:
            rb = tk.Radiobutton(
                btn_row,
                text=text,
                variable=self._mode_var,
                value=val,
                command=self._on_mode_change,
                state=tk.DISABLED,
            )
            rb.pack(side=tk.LEFT, padx=4)
            self._radios.append(rb)

        slide_frame = tk.Frame(self.root)
        slide_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self._idx_var = tk.IntVar(value=max(0, len(self.paths) - 1))
        self._scale = tk.Scale(
            slide_frame,
            from_=0,
            to=max(0, len(self.paths) - 1),
            orient=tk.HORIZONTAL,
            variable=self._idx_var,
            command=self._on_slider_moved,
            label="Чекпоинт (индекс)",
            showvalue=True,
            length=800,
            state=tk.DISABLED,
        )
        self._scale.pack(fill=tk.X)
        self.status = tk.Label(self.root, text="Инициализация…", anchor="w", justify=tk.LEFT)
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        prog_frame = tk.Frame(self.root)
        prog_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        self._progress_bar = ttk.Progressbar(
            prog_frame,
            mode="determinate",
            maximum=max(1, len(self.paths) * (1 + len(MODES))),
        )
        self._progress_bar.pack(fill=tk.X)

        self.fig = Figure(figsize=FIGSIZE_INCH, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=top)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tb_frame = tk.Frame(top)
        tb_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tb_frame)

        if not self.paths:
            self.status.config(text=f"Нет файлов policy_ep_*.pt в {self.history_dir}")
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Нет чекпоинтов", ha="center", va="center", fontsize=14)
            ax.axis("off")
            self.canvas.draw()
            for rb in self._radios:
                rb.config(state=tk.NORMAL)
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Загрузка и рендер {len(self.paths)} чекпоинтов × 4 графика…",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        self.canvas.draw()
        self.root.after(80, self._start_background_worker)

    def _start_background_worker(self) -> None:
        self._worker_done = False
        self._progress_bar["value"] = 0
        cache_note = f", кеш={self.cache_dir}" if self.cache_dir else ", кеш выкл."
        self.status.config(
            text=(
                f"Фоновая загрузка и рендер ({len(self.paths)} чекпоинтов, "
                f"потоков={self.max_workers}{cache_note})… Окно не блокируется."
            ),
        )
        t = threading.Thread(
            target=_load_and_prerender_worker,
            args=(self.paths, self.render_dpi, self._progress_q),
            kwargs={"max_workers": self.max_workers, "cache_dir": self.cache_dir},
            daemon=True,
            name="checkpoint-prerender",
        )
        t.start()
        self._poll_progress_queue()

    def _poll_progress_queue(self) -> None:
        if self._worker_done:
            return
        try:
            while True:
                msg = self._progress_q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, cur, total, label = msg
                    self._progress_bar["maximum"] = max(1, total)
                    self._progress_bar["value"] = min(cur + 1, total)
                    self.status.config(text=label)
                elif kind == "done":
                    _, tiles, meta, total_r, cache_hits = msg
                    self._worker_done = True
                    self._tiles = tiles
                    self._meta = meta
                    self._finish_preload_ok(total_r, cache_hits)
                    return
                elif kind == "error":
                    _, err = msg
                    self._worker_done = True
                    self._finish_preload_err(err)
                    return
        except queue.Empty:
            pass
        self.root.after(40, self._poll_progress_queue)

    def _finish_preload_ok(self, total_r: int, cache_hits: int = 0) -> None:
        ram_mib = _rgb_ram_mib(self._tiles)
        n = len(self._tiles)
        self._progress_bar["value"] = self._progress_bar["maximum"]
        cache_part = f" Кеш: {cache_hits}/{n} с диска." if self.cache_dir else ""
        self.status.config(
            text=(
                f"Готово: {n} чекпоинтов, {total_r} картинок в RAM (~{ram_mib:.0f} МиБ RGB). "
                f"render_dpi={self.render_dpi}.{cache_part} Смена слайдера — только imshow."
            ),
        )
        self._set_controls_enabled(True)
        self._scale.config(to=max(0, n - 1))
        self._idx_var.set(max(0, n - 1))
        self._redraw()

    def _finish_preload_err(self, err: str) -> None:
        self.status.config(text=f"Ошибка: {err}")
        messagebox.showerror("Фоновая загрузка / рендер", err)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, err, ha="center", va="center", fontsize=10, wrap=True)
        ax.axis("off")
        self.canvas.draw()

    def _set_controls_enabled(self, enabled: bool) -> None:
        st = tk.NORMAL if enabled else tk.DISABLED
        self._scale.config(state=st)
        for rb in self._radios:
            rb.config(state=st)

    def _on_mode_change(self) -> None:
        self._redraw()

    def _on_slider_moved(self, val: str) -> None:
        if not self._tiles:
            return
        try:
            i = int(float(val))
        except ValueError:
            return
        i = max(0, min(i, len(self._tiles) - 1))
        if i != int(self._idx_var.get()):
            self._idx_var.set(i)
        self._redraw()

    def _redraw(self) -> None:
        if not self._tiles:
            return
        idx = max(0, min(int(self._idx_var.get()), len(self._tiles) - 1))
        mode = self._mode_var.get()
        ep, name = self._meta[idx]
        img = self._tiles[idx][mode]

        self.status.config(
            text=(
                f"Эпизод: {ep}  |  {name}  |  [{idx + 1}/{len(self._tiles)}]  "
                f"|  предрендер (мгновенно)"
            ),
        )

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.imshow(img, origin="upper", aspect="auto", interpolation="nearest")
        ax.axis("off")
        self.fig.tight_layout(pad=0)
        self.canvas.draw()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Просмотр весов: предрендер всех графиков в RAM",
    )
    ap.add_argument(
        "--history-dir",
        type=Path,
        default=_root / "training" / "checkpoints" / "history",
        help="Каталог с policy_ep_*.pt",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI окна отображения (холст Tk)",
    )
    ap.add_argument(
        "--render-dpi",
        type=int,
        default=None,
        help="DPI при предрендере (по умолчанию = --dpi; меньше — меньше RAM)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Число потоков для параллельной загрузки .pt и рендера Agg "
            "(по умолчанию: min(8, CPU), минимум 1)"
        ),
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Не читать и не писать кеш предрендера на диск",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Каталог кеша .npz (по умолчанию: {_default_tile_cache_dir()})",
    )
    args = ap.parse_args()
    d = args.history_dir.resolve()
    if not d.is_dir():
        print(f"Каталог не найден: {d}")
        sys.exit(1)
    workers = args.workers if args.workers is not None else _default_worker_count()
    if workers < 1:
        print("--workers должен быть >= 1")
        sys.exit(1)
    tile_cache: Path | None = None
    if not args.no_cache:
        tile_cache = (
            args.cache_dir.resolve()
            if args.cache_dir is not None
            else _default_tile_cache_dir()
        )

    app = WeightEvolutionViewer(
        d,
        dpi=args.dpi,
        render_dpi=args.render_dpi,
        max_workers=workers,
        cache_dir=tile_cache,
    )
    app.run()


if __name__ == "__main__":
    main()
