"""
Обучение PolicyNet (actor-critic) методом PPO с GRU и Truncated BPTT.

Улучшения по сравнению с REINFORCE:
  - PPO clip loss (ε=0.2) — защита от слишком крупных обновлений
  - Critic (value head) + GAE (λ=0.95) — резкое снижение дисперсии
  - Truncated BPTT (TBPTT_LEN=256) — стабильные градиенты через GRU
  - Нормализация advantages и returns per-trajectory
  - Cosine LR schedule (lr → lr*0.01)
  - Убывающий entropy coefficient (0.01 → 0.001)
  - Curriculum: episode_time растёт от --curriculum-start до --episode-time-sec

Запуск:
  python training/train_torch.py apartment_1 --episodes 5000 --episode-time-sec 90 --gru --workers 0 --plot-every 100 --save-every 100
  # С curriculum (рекомендуется для fresh start):
  python training/train_torch.py apartment_1 --episodes 5000 --episode-time-sec 90 --gru --workers 0 --curriculum --plot-every 100 --save-every 100
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F

from training.policy_net import PolicyNet
from training.vacuum_env import FPS, VacuumEnv

# ─────────────────────────────────────────────────────────────────────────────
# PPO гиперпараметры
# ─────────────────────────────────────────────────────────────────────────────

CLIP_EPS          = 0.2     # PPO clip ratio
VALUE_COEFF       = 0.5     # вес value loss в суммарном loss
ENTROPY_COEFF_START = 0.05   # entropy bonus на старте (замедлено падение)
ENTROPY_COEFF_END   = 0.02   # в конце не даём схлопнуться — сохраняем исследование
PPO_EPOCHS        = 4       # проходов по данным одного rollout
TBPTT_LEN         = 256     # длина сегмента для Truncated BPTT
GAE_LAMBDA        = 0.95    # λ для GAE
GRAD_CLIP         = 0.5     # max norm для clip_grad_norm_
REWARD_SCALE      = 0.01    # Масштабирование наград для стабилизации Value Loss

ACTION_LABELS = ("вперёд", "назад", "влево", "вправо", "вп+влево", "вп+вправо", "нз+влево", "нз+вправо")

# ─────────────────────────────────────────────────────────────────────────────
# Параллельные воркеры
# ─────────────────────────────────────────────────────────────────────────────

_worker_env: "VacuumEnv | None" = None
_worker_policy: "PolicyNet | None" = None


def _init_worker(
    room_name: str,
    max_steps: int,
    fps: float,
    obs_dim: int,
    n_actions: int,
    hidden_size: int,
    use_gru: bool,
    encoder_layers: int,
) -> None:
    """Вызывается один раз при старте каждого воркер-процесса."""
    global _worker_env, _worker_policy
    import random
    _r = Path(__file__).resolve().parent.parent
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))
    seed = random.randint(0, 99999) ^ os.getpid()
    random.seed(seed)
    torch.manual_seed(seed)

    from training.vacuum_env import VacuumEnv as _VacuumEnv
    from training.policy_net import PolicyNet as _PolicyNet
    
    init_room = room_name
    if room_name == "random":
        from room_loader.room_loader import ROOMS_DIR
        random_rooms = [p.stem for p in ROOMS_DIR.glob("random*.json")]
        if random_rooms:
            init_room = random_rooms[0]
        else:
            init_room = "apartment_1"
            
    _worker_env = _VacuumEnv(room_name=init_room, max_steps=max_steps, fps=fps)
    _worker_policy = _PolicyNet(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_size=hidden_size, use_gru=use_gru,
        encoder_layers=encoder_layers,
    )


def _collect_episode_worker(policy_state_dict: dict, max_steps: int, run_room_name: str | None = None) -> tuple:
    """
    Воркер: один полный PPO rollout.
    Возвращает (traj_obs, traj_act, traj_rew, traj_logp, traj_val,
                traj_hidden_starts, last_value, ep_info).
    """
    global _worker_env, _worker_policy
    _worker_policy.load_state_dict(policy_state_dict)
    _worker_policy.eval()
    _worker_env.set_max_steps(max_steps)

    if run_room_name is not None:
        obs, _ = _worker_env.reset(room_name=run_room_name)
    else:
        obs, _ = _worker_env.reset()
    traj_obs:            list = []
    traj_act:            list = []
    traj_rew:            list = []
    traj_logp:           list = []
    traj_val:            list = []
    traj_hidden_starts:  list = []  # hidden state в начале каждого TBPTT сегмента

    hidden = _worker_policy.init_hidden() if _worker_policy.use_gru else None
    step = 0

    while True:
        if step % TBPTT_LEN == 0:
            traj_hidden_starts.append(hidden.detach().cpu() if hidden is not None else None)

        t = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            logits, value, hidden = _worker_policy.forward_step_ac(t, hidden)
            probs    = torch.softmax(logits, dim=-1)
            dist     = torch.distributions.Categorical(probs=probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

        traj_obs.append(obs)
        traj_act.append(action.item())
        traj_val.append(value.item())
        traj_logp.append(log_prob.item())

        obs, reward, done, info = _worker_env.step(action.item())
        traj_rew.append(reward)
        step += 1

        if done:
            break

    with torch.no_grad():
        t = torch.tensor([obs], dtype=torch.float32)
        _, last_val, _ = _worker_policy.forward_step_ac(t, hidden)
        last_value = last_val.item()

    return traj_obs, traj_act, traj_rew, traj_logp, traj_val, traj_hidden_starts, last_value, info


# ─────────────────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gae(
    rewards: list[float],
    values: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """
    Generalized Advantage Estimation.
    Возвращает (advantages, returns).
    Эпизод с фиксированной длиной — последний шаг bootstrap'ится через last_value.
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# График обучения
# ─────────────────────────────────────────────────────────────────────────────

def _save_training_plot(
    plot_dir: Path,
    step: int,
    rewards: list[float],
    visited_pct: list[float],
    losses: list[float],
    value_losses: list[float],
    action_pct: list[tuple[float, ...]] | None = None,
    window: int = 50,
) -> None:
    """6 панелей: reward, посещено%, policy loss, value loss, доли действий, энтропия."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        print(f"График не сохранён: нет модуля {e.name}. pip install matplotlib")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    n = len(rewards)
    x = range(1, n + 1)

    def smooth(y: list[float], w: int) -> list[float]:
        if w <= 1 or len(y) < w:
            return y
        out = []
        for i in range(len(y)):
            lo = max(0, i - w + 1)
            out.append(sum(y[lo: i + 1]) / (i - lo + 1))
        return out

    n_rows = 6 if action_pct else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.2 * n_rows), sharex=True)
    fig.suptitle(f"PPO Обучение (шаг {step})", fontsize=10)

    ax_reward, ax_visit, ax_ploss, ax_vloss = axes[:4]
    if n_rows == 6:
        ax_act, ax_ent = axes[4], axes[5]

    ax_reward.plot(x, rewards, alpha=0.3, color="C0")
    ax_reward.plot(x, smooth(rewards, window), color="C0", label="reward (среднее)")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="upper right", fontsize=8)
    ax_reward.grid(True, alpha=0.3)

    ax_visit.plot(x, visited_pct, alpha=0.3, color="C1")
    ax_visit.plot(x, smooth(visited_pct, window), color="C1", label="посещено % (среднее)")
    ax_visit.set_ylabel("Посещено %")
    ax_visit.legend(loc="upper right", fontsize=8)
    ax_visit.grid(True, alpha=0.3)

    ax_ploss.plot(x, losses, alpha=0.3, color="C2")
    ax_ploss.plot(x, smooth(losses, window), color="C2", label="policy loss")
    ax_ploss.set_ylabel("Policy Loss")
    ax_ploss.legend(loc="upper right", fontsize=8)
    ax_ploss.grid(True, alpha=0.3)

    if value_losses:
        ax_vloss.plot(x, value_losses, alpha=0.3, color="C3")
        ax_vloss.plot(x, smooth(value_losses, window), color="C3", label="value loss")
    ax_vloss.set_ylabel("Value Loss")
    ax_vloss.legend(loc="upper right", fontsize=8)
    ax_vloss.grid(True, alpha=0.3)

    if action_pct and len(action_pct) == n:
        import math as _math
        n_act = len(action_pct[0])
        max_entropy = _math.log(n_act) if n_act > 1 else 1.0

        for i in range(n_act):
            label = ACTION_LABELS[i] if i < len(ACTION_LABELS) else str(i)
            ax_act.plot(x, [a[i] for a in action_pct], alpha=0.8, color=f"C{i}", label=label)
        ax_act.set_ylabel("Доля (сэмпл.) %")
        ax_act.legend(loc="upper right", fontsize=7, ncol=2)
        ax_act.grid(True, alpha=0.3)
        ax_act.set_ylim(-5, 105)

        def _ep_entropy(counts: tuple[float, ...]) -> float:
            h = 0.0
            for p in counts:
                frac = p / 100.0
                if frac > 1e-9:
                    h -= frac * _math.log(frac)
            return h / max_entropy

        entropy_vals = [_ep_entropy(a) for a in action_pct]
        ax_ent.plot(x, entropy_vals, alpha=0.3, color="C4")
        ax_ent.plot(x, smooth(entropy_vals, window), color="C4", label="энтропия (норм.)")
        ax_ent.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="макс")
        ax_ent.set_ylabel("Энтропия")
        ax_ent.set_xlabel("Эпизод")
        ax_ent.legend(loc="lower right", fontsize=7)
        ax_ent.grid(True, alpha=0.3)
        ax_ent.set_ylim(-0.05, 1.1)
    else:
        ax_vloss.set_xlabel("Эпизод")

    plt.tight_layout()
    history_dir = plot_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"step_{step:06d}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    last_path = plot_dir / "last.png"
    plt.savefig(last_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"График сохранён: {path}  |  last: {last_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Основной цикл PPO
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    env: VacuumEnv,
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    episodes: int,
    gamma: float = 0.99,
    device: torch.device | None = None,
    plot_every: int = 0,
    plot_dir: Path | str | None = None,
    plot_smooth_window: int = 50,
    save_every: int = 0,
    save_dir: Path | str | None = None,
    n_workers: int = 1,
    room_name: str = "apartment_1",
    ep_start: int = 0,
    initial_history: dict | None = None,
    # Curriculum
    curriculum: bool = False,
    curriculum_start_sec: float = 20.0,
    curriculum_step_sec: float = 10.0,
    curriculum_every: int = 200,
    max_episode_sec: float = 90.0,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    out_dir  = Path(plot_dir) if plot_dir else Path(__file__).resolve().parent / "plots"
    ckpt_dir = Path(save_dir) if save_dir else Path(__file__).resolve().parent / "checkpoints"

    if initial_history:
        history_rewards:      list[float]             = list(initial_history.get("rewards", []))
        history_visited_pct:  list[float]             = list(initial_history.get("visited_pct", []))
        history_losses:       list[float]             = list(initial_history.get("losses", []))
        history_value_losses: list[float]             = list(initial_history.get("value_losses", []))
        history_action_pct:   list[tuple[float, ...]] = list(initial_history.get("action_pct", []))
    else:
        history_rewards      = []
        history_visited_pct  = []
        history_losses       = []
        history_value_losses = []
        history_action_pct   = []

    model_config = {
        "obs_dim":        policy.obs_dim,
        "n_actions":      policy.n_actions,
        "hidden_size":    policy.hidden_size,
        "use_gru":        policy.use_gru,
        "encoder_layers": policy.encoder_layers,
        "algo":           "ppo",
    }

    all_rooms = []
    if room_name == "random":
        from room_loader.room_loader import ROOMS_DIR
        # Ищем только комнаты, которые начинаются на 'random'
        all_rooms = [p.stem for p in ROOMS_DIR.glob("random*.json")]
        if not all_rooms:
            # Если таких файлов нет, падаем с понятной ошибкой или берем дефолт
            print("Внимание: файлов random*.json не найдено, использую apartment_1")
            all_rooms = ["apartment_1"]
        print(f"Ресурс: {len(all_rooms)} комнат (random*.json). Смена случайной комнаты каждые 20 эпизодов.")

    import random as _random
    current_room = _random.choice(all_rooms) if room_name == "random" and all_rooms else room_name

    def _get_current_max_steps(ep: int) -> int:
        if not curriculum:
            return env._max_steps
        tier = ep // curriculum_every
        cur_sec = min(curriculum_start_sec + tier * curriculum_step_sec, max_episode_sec)
        return int(cur_sec * FPS)

    executor: ProcessPoolExecutor | None = None
    if n_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(
                room_name, env._max_steps, env._fps,
                env.obs_dim, env.action_space_n,
                policy.hidden_size, policy.use_gru, policy.encoder_layers,
            ),
        )
        print(f"Параллельные воркеры: {n_workers}")

    ep_done = ep_start

    try:
        last_switched_ep = ep_done
        while ep_done < episodes:
            # ── Смена комнаты каждые 20 эпизодов (если random) ───────────────
            if room_name == "random" and ep_done - last_switched_ep >= 20:
                import random as _random
                current_room = _random.choice(all_rooms)
                last_switched_ep = ep_done
                print(f"--- Смена комнаты на: {current_room} ---")

            # ── Curriculum: обновляем max_steps ─────────────────────────────
            current_max_steps = _get_current_max_steps(ep_done)
            env.set_max_steps(current_max_steps)

            # ── Entropy coefficient schedule (линейный decay) ────────────────
            progress = ep_done / max(1, episodes - 1)
            entropy_coeff = ENTROPY_COEFF_START + (ENTROPY_COEFF_END - ENTROPY_COEFF_START) * progress

            # ── Сбор траекторий ──────────────────────────────────────────────
            pass_room = current_room if room_name == "random" else None

            if executor is not None:
                n_batch = min(n_workers, episodes - ep_done)
                cpu_sd  = {k: v.cpu() for k, v in policy.state_dict().items()}
                futures = [executor.submit(_collect_episode_worker, cpu_sd, current_max_steps, pass_room)
                           for _ in range(n_batch)]
                try:
                    batch = [f.result() for f in futures]
                except Exception as exc:
                    print(f"Ошибка в воркере: {exc}")
                    raise
            else:
                # Однопоточный rollout
                if pass_room:
                    obs, _ = env.reset(room_name=pass_room)
                else:
                    obs, _ = env.reset()
                traj_obs:           list = []
                traj_act:           list = []
                traj_rew:           list = []
                traj_logp:          list = []
                traj_val:           list = []
                traj_hidden_starts: list = []
                hidden = policy.init_hidden(device=device) if policy.use_gru else None
                step = 0
                ep_info = {}
                while True:
                    if step % TBPTT_LEN == 0:
                        traj_hidden_starts.append(hidden.detach().cpu() if hidden is not None else None)
                    t = torch.tensor([obs], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits, value, hidden = policy.forward_step_ac(t, hidden)
                        probs    = torch.softmax(logits, dim=-1)
                        dist     = torch.distributions.Categorical(probs=probs)
                        action   = dist.sample()
                        log_prob = dist.log_prob(action)
                    traj_obs.append(obs)
                    traj_act.append(action.item())
                    traj_val.append(value.item())
                    traj_logp.append(log_prob.item())
                    obs, reward, done, ep_info = env.step(action.item())
                    traj_rew.append(reward)
                    step += 1
                    if done:
                        break
                with torch.no_grad():
                    t = torch.tensor([obs], dtype=torch.float32, device=device)
                    _, last_val_t, _ = policy.forward_step_ac(t, hidden)
                    last_value = last_val_t.item()
                batch = [(traj_obs, traj_act, traj_rew, traj_logp, traj_val,
                          traj_hidden_starts, last_value, ep_info)]

            # ── PPO обновление по всем траекториям батча ─────────────────────
            valid = [(o, a, r, lp, v, hs, lv, i)
                     for o, a, r, lp, v, hs, lv, i in batch if o]
            if not valid:
                ep_done += len(batch)
                continue

            ep_policy_losses: list[float] = []
            ep_value_losses:  list[float] = []
            
            all_advs_flat = []
            traj_data = []

            # 1. Вычисляем GAE для всех траекторий батча и собираем advantages
            for (t_obs, t_act, t_rew, t_logp_old, t_val_old,
                 t_hidden_starts, last_val, ep_info) in valid:
                
                # Масштабируем награды для стабилизации Value Loss
                scaled_rew = [r * REWARD_SCALE for r in t_rew]
                
                advs, returns = _compute_gae(scaled_rew, t_val_old, last_val, gamma, GAE_LAMBDA)
                all_advs_flat.extend(advs)
                traj_data.append({
                    "obs": t_obs,
                    "act": t_act,
                    "logp_old": t_logp_old,
                    "advs": advs,
                    "returns": returns,
                    "hidden_starts": t_hidden_starts
                })

            # 2. Глобальная нормализация advantages по всему батчу
            adv_tensor_flat = torch.tensor(all_advs_flat, dtype=torch.float32, device=device)
            adv_mean = adv_tensor_flat.mean()
            adv_std  = adv_tensor_flat.std() + 1e-8

            # 3. Нарезаем все траектории на TBPTT сегменты
            all_segments = []
            for t_data in traj_data:
                T = len(t_data["obs"])
                obs_t      = torch.tensor(t_data["obs"],      dtype=torch.float32, device=device)
                act_t      = torch.tensor(t_data["act"],      dtype=torch.long,    device=device)
                logp_old_t = torch.tensor(t_data["logp_old"], dtype=torch.float32, device=device)
                adv_t      = torch.tensor(t_data["advs"],     dtype=torch.float32, device=device)
                ret_t      = torch.tensor(t_data["returns"],  dtype=torch.float32, device=device)
                
                # Нормализуем advantages глобальными статистиками
                adv_t = (adv_t - adv_mean) / adv_std
                
                # ВАЖНО: Мы больше не нормализуем returns (ret_t), так как предсказываем масштабированные награды!

                t_hidden_starts = t_data["hidden_starts"]
                for seg_idx, seg_start in enumerate(range(0, T, TBPTT_LEN)):
                    seg_end = min(seg_start + TBPTT_LEN, T)
                    
                    h_init = t_hidden_starts[seg_idx] if (t_hidden_starts and seg_idx < len(t_hidden_starts)) else None
                    
                    burn_obs = None
                    h_burn = None
                    if policy.use_gru and seg_start > 0:
                        burn_start = seg_start - TBPTT_LEN
                        burn_obs = obs_t[burn_start:seg_start]
                        prev_seg_idx = seg_idx - 1
                        if t_hidden_starts and prev_seg_idx < len(t_hidden_starts):
                            h_burn = t_hidden_starts[prev_seg_idx]

                    all_segments.append({
                        "obs": obs_t[seg_start:seg_end],
                        "act": act_t[seg_start:seg_end],
                        "adv": adv_t[seg_start:seg_end],
                        "logp": logp_old_t[seg_start:seg_end],
                        "ret": ret_t[seg_start:seg_end],
                        "h_init": h_init,
                        "burn_obs": burn_obs,
                        "h_burn": h_burn,
                    })

            # 4. PPO обновление (проходы по всем сегментам)
            for _epoch in range(PPO_EPOCHS):
                # Перемешиваем сегменты между эпохами для стабильности
                import random as _random
                _random.shuffle(all_segments)
                
                traj_ploss = 0.0
                traj_vloss = 0.0
                n_updates  = 0
                
                for seg in all_segments:
                    h_init = seg["h_init"].to(device) if seg["h_init"] is not None else None
                    
                    # Burn-in: восстанавливаем актуальный hidden state обновлённой политики
                    if _epoch > 0 and seg["burn_obs"] is not None and policy.use_gru:
                        hb = seg["h_burn"].to(device) if seg["h_burn"] is not None else None
                        with torch.no_grad():
                            _, _, h_init = policy.forward_sequence_tbptt(seg["burn_obs"], hb)
                        h_init = h_init.detach()

                    logits_seg, values_seg, _ = policy.forward_sequence_tbptt(seg["obs"], h_init)
                    probs_seg = torch.softmax(logits_seg, dim=-1)
                    dist_seg  = torch.distributions.Categorical(probs=probs_seg)
                    new_logp  = dist_seg.log_prob(seg["act"])
                    entropy   = dist_seg.entropy().mean()

                    # PPO clip loss
                    ratio = torch.exp(new_logp - seg["logp"])
                    pg1   = -ratio * seg["adv"]
                    pg2   = -torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * seg["adv"]
                    policy_loss = torch.max(pg1, pg2).mean()

                    # Value loss: критик предсказывает возвраты от масштабированных наград
                    value_loss = 0.5 * F.mse_loss(values_seg.squeeze(-1), seg["ret"])

                    loss = policy_loss + VALUE_COEFF * value_loss - entropy_coeff * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
                    optimizer.step()

                    traj_ploss += policy_loss.item()
                    traj_vloss += value_loss.item()
                    n_updates  += 1
                    
                if n_updates > 0:
                    ep_policy_losses.append(traj_ploss / n_updates)
                    ep_value_losses.append(traj_vloss / n_updates)

            # LR scheduler: шаг один раз за "раунд" (per batch)
            scheduler.step()

            # ── Логирование ─────────────────────────────────────────────────
            mean_ploss = sum(ep_policy_losses) / max(1, len(ep_policy_losses))
            mean_vloss = sum(ep_value_losses)  / max(1, len(ep_value_losses))

            for (t_obs, t_act, t_rew, t_logp_old, t_val_old,
                 t_hidden_starts, last_val, ep_info) in valid:
                ep_done += 1
                total_reward = sum(t_rew)
                visited      = ep_info.get("visited", 0)
                total_cells  = ep_info.get("total_cells", 0)
                pct          = 100.0 * visited / total_cells if total_cells else 0.0
                Tp           = len(t_act)
                n_act = env.action_space_n
                action_pct_ep = tuple(
                    100.0 * t_act.count(a) / Tp for a in range(n_act)
                ) if Tp > 0 else tuple(0.0 for _ in range(n_act))

                history_rewards.append(total_reward)
                history_visited_pct.append(pct)
                history_losses.append(mean_ploss)
                history_value_losses.append(mean_vloss)
                history_action_pct.append(action_pct_ep)

                cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                cur_sec = current_max_steps / FPS
                print(
                    f"Эпизод {ep_done}/{episodes}: "
                    f"reward={total_reward:.1f}, "
                    f"посещено={visited}/{total_cells} ({pct:.1f}%), "
                    f"p_loss={mean_ploss:.4f}, v_loss={mean_vloss:.4f}, "
                    f"ent={entropy_coeff:.4f}, lr={cur_lr:.5f}"
                    + (f", t={cur_sec:.0f}s" if curriculum else "")
                )

                if plot_every > 0 and ep_done % plot_every == 0:
                    _save_training_plot(
                        out_dir,
                        step=ep_done,
                        rewards=history_rewards,
                        visited_pct=history_visited_pct,
                        losses=history_losses,
                        value_losses=history_value_losses,
                        action_pct=history_action_pct,
                        window=min(plot_smooth_window, max(1, len(history_rewards) // 4)),
                    )

                if save_every > 0 and ep_done % save_every == 0:
                    _save_checkpoint(
                        ckpt_dir, policy, optimizer, scheduler,
                        ep_done, model_config,
                        history_rewards, history_visited_pct,
                        history_losses, history_value_losses, history_action_pct,
                        max_steps=current_max_steps,
                    )

            ep_done += len(batch) - len(valid)

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    # Финальное сохранение
    if episodes > 0 and history_rewards:
        _save_checkpoint(
            ckpt_dir, policy, optimizer, scheduler,
            episodes, model_config,
            history_rewards, history_visited_pct,
            history_losses, history_value_losses, history_action_pct,
            final=True,
            max_steps=current_max_steps,
        )
        _save_training_plot(
            out_dir,
            step=episodes,
            rewards=history_rewards,
            visited_pct=history_visited_pct,
            losses=history_losses,
            value_losses=history_value_losses,
            action_pct=history_action_pct,
            window=min(plot_smooth_window, max(1, len(history_rewards) // 4)),
        )


def _save_checkpoint(
    ckpt_dir: Path,
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    episode: int,
    model_config: dict,
    rewards: list,
    visited_pct: list,
    losses: list,
    value_losses: list,
    action_pct: list,
    final: bool = False,
    max_steps: int = 0,
) -> None:
    ckpt_data = {
        "policy":       policy.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "episode":      episode,
        "config":       model_config,
        "train_config": {"max_steps": max_steps},
    }
    hist_data = {
        "rewards":      rewards,
        "visited_pct":  visited_pct,
        "losses":       losses,
        "value_losses": value_losses,
        "action_pct":   action_pct,
    }
    hist_ckpt_dir = ckpt_dir / "history"
    hist_ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = "policy_final.pt" if final else f"policy_ep_{episode:06d}.pt"
    path = hist_ckpt_dir / name
    torch.save(ckpt_data, path)
    torch.save(ckpt_data, ckpt_dir / "last.pt")
    torch.save(hist_data, ckpt_dir / "training_history.pt")
    print(f"Чекпоинт сохранён: {path}  |  last: {ckpt_dir / 'last.pt'}")


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение PolicyNet (PPO) в VacuumEnv")
    parser.add_argument("room", nargs="?", default="apartment_1")
    parser.add_argument("--episode-time-sec", type=float, default=90.0)
    parser.add_argument("--episodes",    type=int,   default=300)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--gru",         action="store_true")
    parser.add_argument("--workers",     type=int,   default=1,
                        help="0 = авто (все ядра), 1 = без параллельности")
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--plot-every",  type=int,   default=0)
    parser.add_argument("--plot-dir",    type=str,   default=None)
    parser.add_argument("--plot-window", type=int,   default=50)
    parser.add_argument("--save-every",  type=int,   default=0)
    parser.add_argument("--save-dir",    type=str,   default=None)
    # Curriculum
    parser.add_argument("--curriculum",            action="store_true",
                        help="Постепенно увеличивать длину эпизода")
    parser.add_argument("--curriculum-start-sec",  type=float, default=20.0)
    parser.add_argument("--curriculum-step-sec",   type=float, default=10.0)
    parser.add_argument("--curriculum-every",      type=int,   default=200,
                        help="Добавлять --curriculum-step-sec каждые N эпизодов")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        import random
        random.seed(args.seed)

    max_steps = int(args.episode_time_sec * FPS)
    
    all_rooms_for_init = []
    if args.room == "random":
        from room_loader.room_loader import ROOMS_DIR
        all_rooms_for_init = [p.stem for p in ROOMS_DIR.glob("random*.json")]
        
    init_room = all_rooms_for_init[0] if (args.room == "random" and all_rooms_for_init) else args.room
    env = VacuumEnv(room_name=init_room, max_steps=max_steps)
    obs_dim   = env.obs_dim
    n_actions = env.action_space_n

    policy = PolicyNet(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=args.hidden,
        use_gru=args.gru,
        encoder_layers=args.encoder_layers,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    # n_workers нужен до создания scheduler, чтобы правильно задать T_max
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    # scheduler.step() вызывается 1 раз за батч (= n_workers эпизодов),
    # поэтому T_max = ceil(episodes / n_workers), а не episodes
    _lr_t_max = max(1, math.ceil(args.episodes / n_workers))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_lr_t_max, eta_min=args.lr * 0.01
    )

    # ── Автозагрузка чекпоинта ───────────────────────────────────────────────
    ckpt_dir = Path(args.save_dir) if args.save_dir else Path(__file__).resolve().parent / "checkpoints"
    ep_start = 0
    initial_history: dict | None = None

    _last = ckpt_dir / "last.pt"
    _hist_candidates = sorted((ckpt_dir / "history").glob("policy_ep_*.pt")) if (ckpt_dir / "history").exists() else []
    resume_path = _last if _last.exists() else (_hist_candidates[-1] if _hist_candidates else None)

    if resume_path is not None:
        try:
            data = torch.load(resume_path, map_location="cpu", weights_only=False)
            saved_cfg = data.get("config", {})
            compat = (
                saved_cfg.get("obs_dim")        == obs_dim
                and saved_cfg.get("n_actions")   == n_actions
                and saved_cfg.get("hidden_size") == args.hidden
                and saved_cfg.get("use_gru")     == args.gru
                and saved_cfg.get("encoder_layers", 6) == args.encoder_layers
                and saved_cfg.get("algo")        == "ppo"
            )
            if compat:
                policy.load_state_dict(data["policy"])
                ep_start = int(data.get("episode", 0))
                if "optimizer" in data:
                    optimizer.load_state_dict(data["optimizer"])
                if "scheduler" in data:
                    scheduler.load_state_dict(data["scheduler"])
                print(f"Продолжаю с PPO чекпоинта: {resume_path.name}  (эпизод {ep_start})")
                hist_path = ckpt_dir / "training_history.pt"
                if hist_path.exists():
                    try:
                        initial_history = torch.load(hist_path, map_location="cpu", weights_only=False)
                        print(f"История загружена: {len(initial_history.get('rewards', []))} эпизодов")
                    except Exception as exc:
                        print(f"Не удалось загрузить историю: {exc}")
            else:
                algo = saved_cfg.get("algo", "reinforce")
                print(f"Чекпоинт {resume_path.name} несовместим (algo={algo}) — старт с нуля")
        except Exception as exc:
            print(f"Не удалось загрузить чекпоинт {resume_path.name}: {exc} — старт с нуля")

    arch = "GRU" if args.gru else "MLP"
    print(
        f"Архитектура: {arch}  |  "
        f"{obs_dim} → {args.hidden} → {n_actions}  |  "
        f"Параметры: {policy.param_count():,}  |  "
        f"Воркеры: {n_workers}"
        + (" (авто)" if args.workers == 0 else "")
        + (f"  |  Curriculum: {args.curriculum_start_sec:.0f}→{args.episode_time_sec:.0f}s" if args.curriculum else "")
    )

    train_ppo(
        env, policy, optimizer, scheduler, args.episodes,
        gamma=args.gamma,
        plot_every=args.plot_every,
        plot_dir=args.plot_dir,
        plot_smooth_window=args.plot_window,
        save_every=args.save_every,
        save_dir=args.save_dir,
        n_workers=n_workers,
        room_name=args.room,
        ep_start=ep_start,
        initial_history=initial_history,
        curriculum=args.curriculum,
        curriculum_start_sec=args.curriculum_start_sec,
        curriculum_step_sec=args.curriculum_step_sec,
        curriculum_every=args.curriculum_every,
        max_episode_sec=args.episode_time_sec,
    )


if __name__ == "__main__":
    main()
