"""LRURec: complex diagonal recurrence, residual blocks and tied item embeddings.

Independent implementation following https://arxiv.org/abs/2310.02367 and the
architecture in https://github.com/yueqirex/LRURec. Parallel prefix doubling
evaluates the recurrence without a Python loop over individual timestamps.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.sequences import pad_history


def parallel_recurrence(updates, eigenvalues):
    """h[t] = eigenvalues * h[t-1] + updates[t], with h[-1] = 0."""
    state, power, stride = updates, eigenvalues, 1
    while stride < updates.shape[1]:
        state = torch.cat((state[:, :stride],
                           state[:, stride:] + power * state[:, :-stride]), dim=1)
        power = power * power
        stride *= 2
    return state


class LinearRecurrentLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        state_size = 2 * hidden_size
        radius_squared = torch.empty(state_size).uniform_(0.8 ** 2, 0.99 ** 2)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(radius_squared)))
        self.theta_log = nn.Parameter(torch.log(2 * math.pi * torch.rand(state_size).clamp_min(1e-6)))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - radius_squared)))
        self.in_proj = nn.Linear(hidden_size, state_size, dtype=torch.complex64)
        self.out_proj = nn.Linear(state_size, hidden_size, dtype=torch.complex64)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        eigenvalues = torch.exp(torch.complex(-self.nu_log.exp(), self.theta_log.exp()))
        updates = self.in_proj(x.to(torch.complex64)) * self.gamma_log.exp()
        states = parallel_recurrence(updates * mask, eigenvalues)
        return self.norm(x + self.dropout(self.out_proj(states).real)) * mask


class LRUBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.recurrent = LinearRecurrentLayer(hidden_size, dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * hidden_size, hidden_size),
                                 nn.Dropout(dropout))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        x = self.recurrent(x, mask)
        return self.norm(x + self.ffn(x)) * mask


class LRURec(nn.Module):
    def __init__(self, n_items, hidden_size=32, num_blocks=2, max_length=50, dropout=0.2):
        super().__init__()
        if min(n_items, hidden_size, num_blocks, max_length) < 1 or not 0 <= dropout < 1:
            raise ValueError("Use positive model dimensions and dropout in [0, 1).")
        self.config = dict(n_items=n_items, hidden_size=hidden_size, num_blocks=num_blocks,
                           max_length=max_length, dropout=dropout)
        self.n_items_, self.max_length = n_items, max_length
        self.embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([LRUBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            for parameter in module.parameters(recurse=False):
                if parameter.is_complex():
                    nn.init.trunc_normal_(parameter.real, std=0.02, a=-0.04, b=0.04)
                    nn.init.trunc_normal_(parameter.imag, std=0.02, a=-0.04, b=0.04)
                else:
                    nn.init.trunc_normal_(parameter, std=0.02, a=-0.04, b=0.04)

    def encode(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        x = self.embedding_norm(self.embedding_dropout(self.embedding(tokens))) * mask
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, tokens, last_only=False):
        hidden = self.encode(tokens)
        if last_only:
            hidden = hidden[:, -1]
        # Padding is excluded from the classifier. Class index = movie token - 1.
        return hidden @ self.embedding.weight[1:].T + self.item_bias

    @torch.inference_mode()
    def score_histories(self, histories):
        self.eval()
        tokens = np.stack([pad_history(h, self.max_length, self.n_items_) for h in histories])
        device = next(self.parameters()).device
        return self(torch.as_tensor(tokens, device=device), last_only=True).cpu().numpy()

    def score_tokens(self, user_id, positive_movie_tokens):
        return self.score_histories([positive_movie_tokens])[0]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / "weights.pt")

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        model = cls(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
        return model.to(device).eval()
