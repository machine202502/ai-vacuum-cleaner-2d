"""
Политика на PyTorch для среды пылесоса: Actor-Critic с MLP или GRU.

Архитектура (use_gru=True, рекомендуется):
  obs → Encoder → GRU(H,H) → ┬→ actor_head  → logits (n_actions)
                               └→ value_head  → value  (1)

Архитектура (use_gru=False):
  obs → Encoder → mlp_extra → ┬→ actor_head  → logits
                               └→ value_head  → value

Методы:
  forward_step(obs, hidden)          → (logits, probs, new_hidden)         — один шаг (обратная совместимость)
  forward_step_ac(obs, hidden)       → (logits, value, new_hidden)         — один шаг actor-critic для PPO rollout
  forward_sequence(obs_seq, hidden)  → (logits, probs)                     — вся траектория (обратная совместимость)
  forward_sequence_tbptt(obs, hidden)→ (logits, values, last_hidden)       — TBPTT сегмент для PPO update
  sample(obs, hidden)                → (action, log_prob, new_hidden)      — сэмплирование действия
  init_hidden(device)                → Tensor(1, 1, H)                     — нулевое состояние GRU
  param_count()                      → int
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PolicyNet(nn.Module):

    def __init__(
        self,
        obs_dim: int = 6,
        n_actions: int = 4,
        hidden_size: int = 128,
        use_gru: bool = True,
        n_hidden_layers: int = 2,
        encoder_layers: int = 2,  # Уменьшили по умолчанию до 2
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.use_gru = use_gru
        self.encoder_layers = encoder_layers

        # Входной энкодер: 2 слоя H->H по умолчанию (баланс между выразительностью и скоростью прохождения градиента)
        enc: list[nn.Module] = [nn.Linear(obs_dim, hidden_size), nn.ReLU(inplace=True)]
        for _ in range(max(0, encoder_layers - 1)):
            enc.append(nn.Linear(hidden_size, hidden_size))
            enc.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc)

        if use_gru:
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            extra: list[nn.Module] = []
            for _ in range(n_hidden_layers - 1):
                extra.append(nn.Linear(hidden_size, hidden_size))
                extra.append(nn.ReLU(inplace=True))
            self.mlp_extra: nn.Module = nn.Sequential(*extra) if extra else nn.Identity()

        # Actor head: 2 слоя (один нелинейный + выходной)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions)
        )

        # Critic head: 2 слоя (один нелинейный + выходной)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Ортогональная инициализация весов (стандарт для PPO)."""
        if isinstance(module, nn.Linear):
            # Для Actor'а (последнего слоя) делаем веса меньше, чтобы стартовые логиты были ~0
            # Это дает равную вероятность всем действиям на старте (энтропия максимальна)
            if module == self.head[-1]:
                nn.init.orthogonal_(module.weight, gain=0.01)
            # Для Critic'а (последнего слоя) делаем предсказания близкими к 0
            elif module == self.value_head[-1]:
                nn.init.orthogonal_(module.weight, gain=1.0)
            # Для остальных слоев gain = sqrt(2) по стандарту для ReLU
            else:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param, gain=math.sqrt(2))
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=math.sqrt(2))
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Внутренний encoder+RNN проход
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encoder + RNN для одного шага. obs: (batch, obs_dim)."""
        x = self.encoder(obs)
        if self.use_gru:
            x_rnn = x.unsqueeze(1)          # (batch, 1, H)
            out, new_hidden = self.rnn(x_rnn, hidden)
            x = out.squeeze(1)              # (batch, H)
        else:
            x = self.mlp_extra(x)
            new_hidden = None
        return x, new_hidden

    def _encode_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encoder + RNN для последовательности. obs_seq: (T, obs_dim), batch=1."""
        x = self.encoder(obs_seq)
        if self.use_gru:
            x_rnn = x.unsqueeze(0)          # (1, T, H)
            out, new_hidden = self.rnn(x_rnn, hidden)
            x = out.squeeze(0)              # (T, H)
        else:
            x = self.mlp_extra(x)
            new_hidden = None
        return x, new_hidden

    # ──────────────────────────────────────────────────────────────────────────
    # Основные прямые проходы
    # ──────────────────────────────────────────────────────────────────────────

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Один шаг вывода. obs: (batch, obs_dim). Обратная совместимость."""
        x, new_hidden = self._encode_step(obs, hidden)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs, new_hidden

    def forward_step_ac(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Один шаг actor-critic для PPO rollout.
        obs: (batch, obs_dim). Возвращает (logits, value, new_hidden).
        """
        x, new_hidden = self._encode_step(obs, hidden)
        logits = self.head(x)
        value = self.value_head(x)
        return logits, value, new_hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Вся траектория. obs_seq: (T, obs_dim). Обратная совместимость с REINFORCE.
        Возвращает (logits, probs) размера (T, n_actions).
        """
        x, _ = self._encode_sequence(obs_seq, hidden)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def forward_sequence_tbptt(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        TBPTT сегмент для PPO update. obs_seq: (T, obs_dim).
        Возвращает (logits, values, last_hidden).
          logits:      (T, n_actions)
          values:      (T, 1)
          last_hidden: скрытое состояние после последнего шага сегмента
        """
        x, last_hidden = self._encode_sequence(obs_seq, hidden)
        logits = self.head(x)
        values = self.value_head(x)
        return logits, values, last_hidden

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Батчевый прямой проход. Обратная совместимость."""
        logits, probs, _ = self.forward_step(obs, hidden)
        return logits, probs

    # ──────────────────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ──────────────────────────────────────────────────────────────────────────

    def sample(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Сэмплировать действие. Возвращает (action, log_prob, new_hidden)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits, probs, new_hidden = self.forward_step(obs, hidden)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, new_hidden

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Батчевый log π(a|s). obs: (batch, obs_dim), action: (batch,)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        _, probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.log_prob(action)

    def init_hidden(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Нулевое скрытое состояние GRU: (num_layers=1, batch=1, H)."""
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
