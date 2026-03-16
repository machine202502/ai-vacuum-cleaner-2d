# Обучение агента-пылесоса

- **vacuum_env.py** — среда с интерфейсом `reset()` → (obs, info), `step(action)` → (obs, reward, done, info).
  - Действия: 0=вперёд, 1=назад, 2=влево, 3=вправо.
  - Наблюдение: 4 float (IR вперёд, +30°, −30°, нормализованный encoder).
  - Награда за шаг: сумма по ячейкам, чей счётчик посещений вырос на этом шаге:
    - 1 посещение: +1.0
    - 2 посещения: +0.2
    - 3 посещения: 0
    - 4 посещения: −0.1
    - 5 и больше: −0.1×(N−3) (растущий штраф)

- **policy_net.py** — политика на PyTorch (MLP).
  - **Вход**: 4 числа (IR вперёд, IR +30°, IR −30°, encoder, все норм. 0…1).
  - **Скрытые слои**: два полносвязных по 64 нейрона с ReLU: `Linear(4→64) → ReLU → Linear(64→64) → ReLU`.
  - **Выход**: 4 логита действий → softmax → вероятности; действие семплируется из Categorical.
  - Итого архитектура: **4 → 64 → 64 → 4** (~4.7k параметров).

- **train_torch.py** — обучение PolicyNet методом REINFORCE (policy gradient). Запуск из **корня проекта**:
  ```bash
  python training/train_torch.py apartment_1 --episodes 300 --lr 1e-3
  python training/train_torch.py apartment_1 --episodes 500 --hidden 128 --seed 42
  python training/train_torch.py apartment_1 --plot-every 50 --plot-dir training/plots
  ```
  С `--plot-every N` каждые N эпизодов и в конце сохраняется PNG с тремя графиками: reward, % посещённых ячеек, loss (сырые значения + сглаженные кривые). Папка по умолчанию: `training/plots`.
  С `--save-every N` каждые N эпизодов сохраняется чекпоинт (`policy_ep_XXXXXX.pt`), в конце — `policy_final.pt` в `training/checkpoints`. В файле: `{"policy": state_dict, "episode": N}`. Загрузка: `policy.load_state_dict(torch.load("policy_final.pt")["policy"])`.

- **train.py** — минимальный цикл без нейросети: проверка среды, цикличная или случайная политика:
  ```bash
  python training/train.py apartment_1 --max-steps 2000 --episodes 5
  python training/train.py apartment_1 --random --seed 42
  ```
