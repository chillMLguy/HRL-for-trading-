"""
  Input: (batch, w_cnn) vol-normalized log returns
  Conv1D(1→16, k=5, s=1, pad=2) + ReLU
  Conv1D(16→32, k=5, s=2, pad=2) + ReLU
  Conv1D(32→32, k=3, s=2, pad=1) + ReLU
  AdaptiveAvgPool1D(1) → flatten → (32,)
  Linear(32→8) + ReLU
  Linear(8→latent_dim)             ← feature vector (used by RL)
  Linear(latent_dim→1) + Sigmoid   ← prediction head (training only)
"""
import torch
import torch.nn as nn


class PricePatternCNN(nn.Module):
    def __init__(self, input_length: int, latent_dim: int = 3):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.encoder = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, input_length)
        x = x.unsqueeze(1)            # (batch, 1, input_length)
        x = self.conv(x).squeeze(-1)  # (batch, 32)
        latent = self.encoder(x)       # (batch, latent_dim)
        pred = self.head(latent)       # (batch, 1)
        return pred.squeeze(-1)

    def extract_features(self, x):
        with torch.no_grad():
            x = x.unsqueeze(1)
            x = self.conv(x).squeeze(-1)
            latent = self.encoder(x)
        return latent  # (batch, latent_dim)
