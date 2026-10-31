"""
Model and dataset definitions for wine sensor time-series classification.

Tensor layout: ``(N, C, T)`` — batch **C** channels, **T** time steps.

Provenance (full references and code — usable without the PDF)
================================================================

**CNNLSTM4** — CNN–LSTM architecture **proposed in this work**

**EmissionCNN1DLSTM** — Lighter CNN–LSTM variant used in this repo; same article
and **code availability** as the Sung *et al.* study:
https://www.nature.com/articles/s41467-024-45430-9#code-availability
Associated data/code deposit (Figshare): https://doi.org/10.6084/m9.figshare.24312670

**WineTransformer**, **WineLSTMNet**, **WineCNN1DClassifier** — Adapted from SmellNet
(``Transformer``, ``LSTMNet``, ``CNN1DClassifier`` in ``create_model``):
https://github.com/MIT-MI/SmellNet/blob/main/models/models.py

**WineMLSTMfcn** — Adapted from the MLSTM-FCN PyTorch port (transpose inputs so
``(B, C, T)`` uses **time** as the LSTM sequence axis):
https://github.com/alexmelekhin/MLSTM-FCN-Pytorch/blob/main/src/model.py

"""
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def create_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    model_params: dict = None,
) -> nn.Module:
    """Create a model by registry name. See module docstring for URLs and article links."""
    params = model_params or {}
    if model_name == "CNNLSTM4":
        return CNNLSTM4(in_channels=in_channels, num_classes=num_classes, **params)
    if model_name == "EmissionCNN1DLSTM":
        return EmissionCNN1DLSTM(
            in_channels=in_channels, num_classes=num_classes, **params
        )
    if model_name == "Transformer":
        return WineTransformer(
            in_channels=in_channels, num_classes=num_classes, **params
        )
    if model_name == "LSTMNet":
        return WineLSTMNet(
            in_channels=in_channels, num_classes=num_classes, **params
        )
    if model_name == "CNN1DClassifier":
        return WineCNN1DClassifier(
            in_channels=in_channels, num_classes=num_classes, **params
        )
    if model_name == "MLSTMfcn":
        return WineMLSTMfcn(
            in_channels=in_channels, num_classes=num_classes, **params
        )
    raise ValueError(
        f"Unknown model: {model_name}. Use CNNLSTM4, EmissionCNN1DLSTM, "
        "Transformer, LSTMNet, CNN1DClassifier, or MLSTMfcn."
    )


class SensorDataset(Dataset):
    """Dataset for sensor time-series: X (N,C,T), y (N,C) one-hot for classification/multilabel."""

    def __init__(self, X, y, metadata=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y.ndim == 2:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(
                y, dtype=torch.long if y.dtype.kind in ("i", "u") else torch.float32
            )
        self.metadata = metadata

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# CNNLSTM4 — CNN–LSTM baseline proposed in this manuscript (see paper/paper.tex).
# create_model registry: "CNNLSTM4"  ·  tensor layout: (B, C, T)
# ---------------------------------------------------------------------------
class CNNLSTM4(nn.Module):
    """CNN–LSTM proposed in this work; see ``paper/paper.tex``. Input ``(B, C, T)``."""

    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, T, C)
        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# EmissionCNN1DLSTM — lighter CNN–LSTM variant (Sung et al., Nat. Commun. 2024;
# code availability / eigengraph pipeline). Not the same architecture as CNNLSTM4.
# create_model registry: "EmissionCNN1DLSTM"  ·  layout: (B, C, T)
# ---------------------------------------------------------------------------
class EmissionCNN1DLSTM(nn.Module):
    """Lighter CNN–LSTM; ``(B, C, T)``. Code availability: https://www.nature.com/articles/s41467-024-45430-9#code-availability"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        lstm_units: int = 64,
        dense_units: int = 100,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding — fixed sinusoidal PE along the time axis for
# WineTransformer (applied after projecting channels -> model_dim).
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------------------------------------------------------------------------
# WineTransformer — TransformerEncoder classifier; adapted from SmellNet.
# create_model registry: "Transformer"  ·  upstream:
# https://github.com/MIT-MI/SmellNet/blob/main/models/models.py
# ---------------------------------------------------------------------------
class WineTransformer(nn.Module):
    """Transformer; adapted from SmellNet ``models/models.py``. Input ``(B, C, T)``. https://github.com/MIT-MI/SmellNet/blob/main/models/models.py"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.pos = SinusoidalPositionalEncoding(model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.input_proj(x)
        x = self.pos(x)
        h = self.transformer(x)
        feat = h.mean(dim=1)
        feat = self.dropout(feat)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# WineLSTMNet — bidirectional LSTM + linear embedding + logits; adapted from SmellNet.
# create_model registry: "LSTMNet"  ·  (B, C, T) -> (B, T, C) before LSTM
# https://github.com/MIT-MI/SmellNet/blob/main/models/models.py
# ---------------------------------------------------------------------------
class WineLSTMNet(nn.Module):
    """LSTMNet; adapted from SmellNet ``models/models.py``. ``(B, C, T)`` -> ``(B, T, C)``. https://github.com/MIT-MI/SmellNet/blob/main/models/models.py"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        feat_dim = hidden_dim * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.embedding_layer = nn.Linear(feat_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        out, _ = self.lstm(x)
        if self.pool == "mean":
            feat = out.mean(dim=1)
        elif self.pool == "max":
            feat = out.max(dim=1).values
        else:
            feat = out[:, -1, :]
        emb = self.embedding_layer(feat)
        return self.classifier(emb)


# ---------------------------------------------------------------------------
# WineCNN1DClassifier — Conv1D stack + global average pool + linear head; SmellNet.
# create_model registry: "CNN1DClassifier"  ·  layout: (B, C, T) Conv1d-native
# https://github.com/MIT-MI/SmellNet/blob/main/models/models.py
# ---------------------------------------------------------------------------
class WineCNN1DClassifier(nn.Module):
    """1D-CNN; adapted from SmellNet ``models/models.py``. ``(B, C, T)``. https://github.com/MIT-MI/SmellNet/blob/main/models/models.py"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        channels: tuple = (64, 128, 256),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        convs = []
        c_in = in_channels
        for c_out in channels:
            pad = kernel_size // 2
            convs.extend([
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            c_in = c_out

        self.conv = nn.Sequential(*convs)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.gap(h).squeeze(-1)
        return self.head(h)


# ---------------------------------------------------------------------------
# SELayer — channel squeeze-excitation on conv features (MLSTM-FCN component).
# ---------------------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# WineMLSTMfcn — LSTM branch + parallel multi-scale Conv1D + SE blocks + concat FC.
# create_model registry: "MLSTMfcn"  ·  LSTM sees (B, T, C) via transpose(1,2)
# https://github.com/alexmelekhin/MLSTM-FCN-Pytorch/blob/main/src/model.py
# ---------------------------------------------------------------------------
class WineMLSTMfcn(nn.Module):
    """MLSTM-FCN; adapted from https://github.com/alexmelekhin/MLSTM-FCN-Pytorch/blob/main/src/model.py — LSTM branch uses ``x.transpose(1,2)`` so time is sequence. Logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        num_lstm_out: int = 128,
        num_lstm_layers: int = 1,
        conv1_nf: int = 128,
        conv2_nf: int = 256,
        conv3_nf: int = 128,
        lstm_drop_p: float = 0.3,
        fc_drop_p: float = 0.3,
    ):
        super().__init__()
        self.num_lstm_out = num_lstm_out

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=num_lstm_out,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.conv1 = nn.Conv1d(in_channels, conv1_nf, 8, padding=4)
        self.conv2 = nn.Conv1d(conv1_nf, conv2_nf, 5, padding=2)
        self.conv3 = nn.Conv1d(conv2_nf, conv3_nf, 3, padding=1)

        self.bn1 = nn.BatchNorm1d(conv1_nf)
        self.bn2 = nn.BatchNorm1d(conv2_nf)
        self.bn3 = nn.BatchNorm1d(conv3_nf)

        self.se1 = SELayer(conv1_nf, reduction=min(16, conv1_nf))
        self.se2 = SELayer(conv2_nf, reduction=min(16, conv2_nf))

        self.relu = nn.ReLU()
        self.lstm_drop = nn.Dropout(lstm_drop_p)
        self.conv_drop = nn.Dropout(fc_drop_p)

        self.fc = nn.Linear(conv3_nf + num_lstm_out, num_classes)

    def forward(self, x: torch.Tensor, seq_lens=None) -> torch.Tensor:
        x1 = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if seq_lens is not None:
            x1 = nn.utils.rnn.pack_padded_sequence(
                x1, seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            x1, (ht, ct) = self.lstm(x1)
            x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, padding_value=0.0)
        else:
            x1, (ht, ct) = self.lstm(x1)
        x1 = x1[:, -1, :]
        x1 = self.lstm_drop(x1)

        x2 = self.conv_drop(self.relu(self.bn1(self.conv1(x))))
        x2 = self.se1(x2)
        x2 = self.conv_drop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.conv_drop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, dim=2)

        x_all = torch.cat((x1, x2), dim=1)
        return self.fc(x_all)
