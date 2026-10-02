"""Late-fusion multimodal network for CardioTwin.

Architecture:
  static (8)  -> 2-layer MLP  -> 32-dim embedding
  dynamic (T,5) -> 2-layer BiGRU -> 64-dim temporal embedding
  concat (96) -> MLP head (96->48->1) + sigmoid -> P(decompensation|24h)

Fusion mathematics:
  h_s = MLP_s(x_static)
  h_t = BiGRU(x_dynamic)[final]
  h   = [h_s ; h_t]
  p   = sigmoid(W2 * ReLU(W1 * h + b1) + b2)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    """Two-layer MLP for static EHR baselines.

    Uses LayerNorm (rather than BatchNorm) so single-patient inference
    (batch size 1, as in the live dashboard) is exact in any mode.
    """

    def __init__(self, in_features: int = 8, embed_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicEncoder(nn.Module):
    """Two-layer bidirectional GRU for wearable telemetry."""

    def __init__(
        self,
        in_features: int = 5,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * 2  # 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, h_n = self.gru(x)  # h_n: (2*num_layers, B, hidden)
        # Take last layer forward + backward states.
        fwd = h_n[-2]
        bwd = h_n[-1]
        return torch.cat([fwd, bwd], dim=-1)  # (B, 64)


class CardioTwinNet(nn.Module):
    """Late-fusion CardioTwin classifier."""

    def __init__(
        self,
        n_static: int = 8,
        n_dynamic: int = 5,
        static_dim: int = 32,
        gru_hidden: int = 32,
        gru_layers: int = 2,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.static_encoder = StaticEncoder(in_features=n_static, embed_dim=static_dim)
        self.dynamic_encoder = DynamicEncoder(
            in_features=n_dynamic, hidden_size=gru_hidden, num_layers=gru_layers
        )
        fused_dim = static_dim + self.dynamic_encoder.out_dim  # 96
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 48),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor) -> torch.Tensor:
        h_s = self.static_encoder(x_static)
        h_t = self.dynamic_encoder(x_dynamic)
        h = torch.cat([h_s, h_t], dim=-1)
        return torch.sigmoid(self.head(h))

    def predict_proba(
        self, x_static: torch.Tensor, x_dynamic: torch.Tensor
    ) -> torch.Tensor:
        """Inference helper (no-grad sigmoid probabilities)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x_static, x_dynamic)


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
