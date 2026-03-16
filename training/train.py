"""
Минимальный цикл обучения: проверка среды и случайная/цикличная политика.
Запуск из корня проекта: python training/train.py [room_name] [--max-steps N] [--episodes M]
Можно заменить политику на свою модель (DQN, PPO и т.д.).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Добавить корень проекта в path при запуске из любой папки
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from training.vacuum_env import VacuumEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка среды VacuumEnv и случайная политика")
    parser.add_argument("room", nargs="?", default="apartment_1", help="Имя комнаты (например apartment_1)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Макс. шагов за эпизод")
    parser.add_argument("--episodes", type=int, default=2, help="Число эпизодов")
    parser.add_argument("--random", action="store_true", help="Случайная политика (иначе цикличная 0,1,2,3)")
    parser.add_argument("--seed", type=int, default=None, help="Seed для random")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    env = VacuumEnv(room_name=args.room, max_steps=args.max_steps)
    obs, info = env.reset()
    print(f"obs_dim={env.obs_dim}, action_space_n={env.action_space_n}, total_cells={info.get('visit_total', 0)}")

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            if args.random:
                action = random.randint(0, env.action_space_n - 1)
            else:
                action = env._step_count % env.action_space_n
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        visited = info.get("visited", 0)
        total_cells = info.get("total_cells", 0)
        pct = 100.0 * visited / total_cells if total_cells else 0
        print(f"Эпизод {ep + 1}: шагов={steps}, reward={total_reward:.2f}, посещено={visited}/{total_cells} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
