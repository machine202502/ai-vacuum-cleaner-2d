"""
Окно: слева delta (окно из N эпизодов до X, N — слайдер), справа last (1..X).
Слайдер X — конечный эпизод; отдельный слайдер — окно сглаживания (больше — виднее тренд).
Панели под графиками: зум/панорама; ось X синхронизируется между всеми графиками слева и справа,
ось Y — между парой строк (одинаковая метрика).
Автообновление training_history.pt.
Кнопка Render сохраняет правый график в training/plots/manual-last.png.

Запуск из корня проекта:
  python plot-render.py
  python plot-render.py --history training/checkpoints/training_history.pt --delta 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from training.history_plot_common import (
    PLOT_DELTA_LAST_EPISODES,
    draw_training_curves,
    history_plot_rows,
    load_history,
)

# Верх слайдера сглаживания (эпизодов в скользящем среднем)
SMOOTH_SLIDER_MAX = 400


class PlotRenderApp:
    def __init__(
        self,
        history_path: Path,
        reload_ms: int,
        delta_span: int,
        smooth_cap: int,
    ) -> None:
        self.history_path = history_path
        self.reload_ms = reload_ms
        self.delta_span = delta_span

        self.hist: dict = {}
        self.n = 0
        self._n_rows: int | None = None
        self.fig_delta = self.fig_last = None
        self.canvas_delta = self.canvas_last = None
        self.axes_delta = self.axes_last = None
        self._redraw_after_id: str | None = None
        self._delta_redraw_after_id: str | None = None
        self.manual_last_path = _root / "training" / "plots" / "manual-last.png"

        self._zoom_sync_x = False
        self._zoom_sync_y = False
        self._xlim_cids: list[tuple[object, int]] = []
        self._ylim_cids: list[tuple[object, int]] = []
        self._tb_frame_delta: tk.Frame | None = None
        self._tb_frame_last: tk.Frame | None = None

        self.root = tk.Tk()
        self.root.title("PPO — delta (слева) / last (справа)")
        self.root.minsize(900, 600)

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)
        top.grid_rowconfigure(1, weight=1)

        hdr_d = tk.Frame(top)
        hdr_d.grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.lbl_delta_title = tk.Label(hdr_d, text="", font=("", 10, "bold"))
        self.lbl_delta_title.pack(side=tk.LEFT)
        tk.Label(top, text="Last (эп. 1 … X)", font=("", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=6, pady=2
        )

        self.frame_delta = tk.Frame(top)
        self.frame_delta.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.frame_last = tk.Frame(top)
        self.frame_last.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        self.ep_var = tk.IntVar(value=1)
        self.ep_scale = tk.Scale(
            bottom,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.ep_var,
            command=self._on_scale_moved,
            label="Конечный эпизод X",
        )
        self.ep_scale.pack(fill=tk.X)

        _d0 = max(1, delta_span)
        self.delta_var = tk.IntVar(value=_d0)
        self.delta_scale = tk.Scale(
            bottom,
            from_=1,
            to=max(500, _d0),
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.delta_var,
            command=self._on_delta_slider_moved,
            label="Окно delta (число эпизодов в окне)",
            showvalue=True,
        )
        self.delta_scale.pack(fill=tk.X, pady=(0, 2))

        _s0 = max(1, min(smooth_cap, SMOOTH_SLIDER_MAX))
        self.smooth_var = tk.IntVar(value=_s0)
        self.smooth_scale = tk.Scale(
            bottom,
            from_=1,
            to=SMOOTH_SLIDER_MAX,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.smooth_var,
            command=self._on_scale_moved,
            label="Сглаживание (окно эпизодов; больше — плавнее, лучше виден рост)",
            showvalue=True,
        )
        self.smooth_scale.pack(fill=tk.X, pady=(0, 2))

        row = tk.Frame(bottom)
        row.pack(fill=tk.X, pady=4)
        tk.Button(row, text="Обновить из файла", command=self.refresh_manual).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(row, text="Render → manual-last.png", command=self.save_manual_last).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.status = tk.Label(row, text="Загрузка…", anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._update_delta_caption()
        self.refresh_from_disk(is_auto=False, is_initial=True)
        self._schedule_auto_reload()

    def _schedule_auto_reload(self) -> None:
        self.root.after(self.reload_ms, self._auto_reload_tick)

    def _auto_reload_tick(self) -> None:
        if self.root.winfo_exists():
            self.refresh_from_disk(is_auto=True, is_initial=False)
            self._schedule_auto_reload()

    def refresh_manual(self) -> None:
        self.refresh_from_disk(is_auto=False, is_initial=False)

    def _update_delta_caption(self) -> None:
        d = self.delta_span
        self.lbl_delta_title.config(
            text=f"Delta (окно {d} эп.: max(1, X−{d}+1) … X)",
        )

    def _on_delta_slider_moved(self, _val: str) -> None:
        if self._delta_redraw_after_id is not None:
            self.root.after_cancel(self._delta_redraw_after_id)
        self._delta_redraw_after_id = self.root.after(40, self._apply_delta_slider)

    def _apply_delta_slider(self) -> None:
        self._delta_redraw_after_id = None
        if self.n <= 0 or self.axes_delta is None:
            return
        try:
            v = int(self.delta_var.get())
        except tk.TclError:
            return
        upper = max(1, self.n)
        v = max(1, min(v, upper))
        if v != int(self.delta_var.get()):
            self.delta_var.set(v)
        if v == self.delta_span:
            return
        self.delta_span = v
        self._update_delta_caption()
        self._redraw_plots()

    def _sync_delta_scale_range(self) -> None:
        upper = max(1, self.n)
        self.delta_span = min(max(1, self.delta_span), upper)
        self.delta_scale.configure(from_=1, to=upper)
        self.delta_var.set(self.delta_span)
        self._update_delta_caption()

    def _smooth_window(self, series_len: int) -> int:
        """Окно скользящего среднего, не больше длины ряда."""
        if series_len <= 0:
            return 1
        try:
            w = int(self.smooth_var.get())
        except tk.TclError:
            w = 1
        return max(1, min(w, series_len))

    def save_manual_last(self) -> None:
        if self.fig_last is None or self.n == 0:
            self.status.config(text="Нет данных для Render")
            return
        self._redraw_plots()
        self.manual_last_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig_last.savefig(self.manual_last_path, dpi=100, bbox_inches="tight")
        self.status.config(text=f"Сохранено: {self.manual_last_path}")

    def _on_scale_moved(self, _val: str) -> None:
        if self._redraw_after_id is not None:
            self.root.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.root.after(40, self._redraw_plots)

    def _disconnect_zoom_sync(self) -> None:
        for ax, cid in self._xlim_cids:
            try:
                ax.callbacks.disconnect(cid)
            except (ValueError, AttributeError):
                pass
        self._xlim_cids.clear()
        for ax, cid in self._ylim_cids:
            try:
                ax.callbacks.disconnect(cid)
            except (ValueError, AttributeError):
                pass
        self._ylim_cids.clear()

    def _all_axes_flat(self) -> list:
        if self.axes_delta is None or self.axes_last is None:
            return []
        return list(np.ravel(self.axes_delta)) + list(np.ravel(self.axes_last))

    def _ax_row_index(self, ax) -> tuple[str, int] | tuple[None, int]:
        for i, a in enumerate(np.ravel(self.axes_delta)):
            if a is ax:
                return ("d", i)
        for i, a in enumerate(np.ravel(self.axes_last)):
            if a is ax:
                return ("l", i)
        return (None, -1)

    def _partner_ax(self, side: str, idx: int):
        if side == "d":
            return np.ravel(self.axes_last)[idx]
        return np.ravel(self.axes_delta)[idx]

    def _on_xlim_changed(self, src_ax) -> None:
        if self._zoom_sync_x:
            return
        self._zoom_sync_x = True
        try:
            lo, hi = src_ax.get_xlim()
            for ax in self._all_axes_flat():
                if ax is src_ax:
                    continue
                olo, ohi = ax.get_xlim()
                if abs(olo - lo) < 1e-9 and abs(ohi - hi) < 1e-9:
                    continue
                ax.set_xlim(lo, hi)
            self.canvas_delta.draw_idle()
            self.canvas_last.draw_idle()
        finally:
            self._zoom_sync_x = False

    def _on_ylim_changed(self, src_ax) -> None:
        if self._zoom_sync_y:
            return
        side, idx = self._ax_row_index(src_ax)
        if idx < 0:
            return
        self._zoom_sync_y = True
        try:
            lo, hi = src_ax.get_ylim()
            partner = self._partner_ax(side, idx)
            plo, phi = partner.get_ylim()
            if abs(plo - lo) < 1e-9 and abs(phi - hi) < 1e-9:
                return
            partner.set_ylim(lo, hi)
            self.canvas_delta.draw_idle()
            self.canvas_last.draw_idle()
        finally:
            self._zoom_sync_y = False

    def _attach_zoom_sync(self) -> None:
        self._disconnect_zoom_sync()
        for ax in self._all_axes_flat():
            cid_x = ax.callbacks.connect(
                "xlim_changed", lambda _e, a=ax: self._on_xlim_changed(a)
            )
            self._xlim_cids.append((ax, cid_x))
            cid_y = ax.callbacks.connect(
                "ylim_changed", lambda _e, a=ax: self._on_ylim_changed(a)
            )
            self._ylim_cids.append((ax, cid_y))

    def _destroy_canvases(self) -> None:
        self._disconnect_zoom_sync()
        if self._tb_frame_delta is not None:
            self._tb_frame_delta.destroy()
            self._tb_frame_delta = None
        if self._tb_frame_last is not None:
            self._tb_frame_last.destroy()
            self._tb_frame_last = None
        for c in (self.canvas_delta, self.canvas_last):
            if c is not None:
                c.get_tk_widget().destroy()
        self.canvas_delta = self.canvas_last = None
        self.fig_delta = self.fig_last = None
        self.axes_delta = self.axes_last = None

    def _build_figures(self, n_rows: int) -> None:
        self._destroy_canvases()
        h = 2.0 * n_rows + 0.5
        self.fig_delta, ax_d = plt.subplots(n_rows, 1, figsize=(5.0, h), dpi=96, sharex=True)
        self.fig_last, ax_l = plt.subplots(n_rows, 1, figsize=(5.0, h), dpi=96, sharex=True)
        self.axes_delta = np.array([ax_d]) if n_rows == 1 else np.asarray(ax_d)
        self.axes_last = np.array([ax_l]) if n_rows == 1 else np.asarray(ax_l)

        self.canvas_delta = FigureCanvasTkAgg(self.fig_delta, master=self.frame_delta)
        self.canvas_delta.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._tb_frame_delta = tk.Frame(self.frame_delta)
        self._tb_frame_delta.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_delta, self._tb_frame_delta)

        self.canvas_last = FigureCanvasTkAgg(self.fig_last, master=self.frame_last)
        self.canvas_last.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._tb_frame_last = tk.Frame(self.frame_last)
        self._tb_frame_last.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas_last, self._tb_frame_last)

        self._attach_zoom_sync()

    def refresh_from_disk(self, *, is_auto: bool, is_initial: bool) -> None:
        old_n = self.n
        cur_ep = int(self.ep_var.get()) if old_n > 0 else 1
        was_at_end = old_n > 0 and cur_ep >= old_n

        if not self.history_path.is_file():
            self.status.config(text=f"Нет файла: {self.history_path}")
            return

        try:
            hist = load_history(self.history_path)
        except Exception as exc:
            self.status.config(text=f"Ошибка чтения: {exc}")
            return

        self.hist = hist
        self.n = len(hist.get("rewards", []))
        n_rows = history_plot_rows(hist)

        if self.n == 0:
            self.status.config(text="История пуста")
            return

        if n_rows != self._n_rows:
            self._n_rows = n_rows
            self._build_figures(n_rows)

        self.ep_scale.configure(to=self.n)

        if is_initial or old_n == 0:
            self.ep_var.set(self.n)
        elif was_at_end:
            self.ep_var.set(self.n)
        else:
            self.ep_var.set(max(1, min(cur_ep, self.n)))

        self._sync_delta_scale_range()

        tag = "авто" if is_auto else "ручное"
        self.status.config(
            text=f"{tag}: {self.n} эп. | файл: {self.history_path.name}"
        )
        self._redraw_plots()

    def _redraw_plots(self) -> None:
        self._redraw_after_id = None
        if self.n == 0 or self.axes_delta is None or self.axes_last is None:
            return

        X = int(self.ep_var.get())
        X = max(1, min(X, self.n))
        if X != int(self.ep_var.get()):
            self.ep_var.set(X)

        rewards = self.hist["rewards"]
        visited = self.hist["visited"]
        losses = self.hist["losses"]
        vloss = self.hist["value_losses"]
        ap = self.hist.get("action_pct")

        w_last = self._smooth_window(X)
        x_last = list(range(1, X + 1))
        draw_training_curves(
            self.axes_last,
            x_last,
            rewards[:X],
            visited[:X],
            losses[:X],
            vloss[:X],
            ap,
            w_last,
            f"Last: эпизоды 1 … {X} (всего {self.n})",
            self._n_rows or 6,
        )

        start = max(0, X - self.delta_span)
        m = X - start
        w_d = self._smooth_window(m) if m > 0 else 1
        x_delta = list(range(start + 1, X + 1))
        k = X - start
        ap_delta = ap[start:X] if ap else None
        draw_training_curves(
            self.axes_delta,
            x_delta,
            rewards[start:X],
            visited[start:X],
            losses[start:X],
            vloss[start:X],
            ap_delta,
            w_d,
            f"Delta: {k} эп. ({start + 1} … {X})",
            self._n_rows or 6,
        )

        self.canvas_delta.draw_idle()
        self.canvas_last.draw_idle()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Просмотр training_history.pt: delta | last")
    ap.add_argument(
        "--history",
        type=Path,
        default=_root / "training" / "checkpoints" / "training_history.pt",
        help="Путь к training_history.pt",
    )
    ap.add_argument(
        "--reload-sec",
        type=int,
        default=60,
        help="Интервал автообновления из файла (сек)",
    )
    ap.add_argument(
        "--delta",
        type=int,
        default=PLOT_DELTA_LAST_EPISODES,
        help="Ширина окна delta (эпизодов)",
    )
    ap.add_argument(
        "--smooth-cap",
        "--smooth",
        type=int,
        default=50,
        dest="smooth_cap",
        help="Начальное окно сглаживания (1…%d; в окне — слайдер)" % SMOOTH_SLIDER_MAX,
    )
    args = ap.parse_args()

    app = PlotRenderApp(
        history_path=args.history.resolve(),
        reload_ms=max(1_000, args.reload_sec * 1000),
        delta_span=max(1, args.delta),
        smooth_cap=max(1, min(args.smooth_cap, SMOOTH_SLIDER_MAX)),
    )
    app.run()


if __name__ == "__main__":
    main()
