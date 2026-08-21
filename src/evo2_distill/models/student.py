from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ArchitectureV1:
    embedding_dim: int = 8
    channels: int = 16
    kernel_size: int = 3
    stem_kernel_size: int = 7
    stem_stride: int = 2
    dilation_schedule: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    group_norm_groups: int = 4
    dropout: float = 0.10
    head_hidden: int = 32

    @property
    def receptive_field_bp(self) -> int:
        return self.stem_kernel_size + sum(
            2 * (self.kernel_size - 1) * dilation * self.stem_stride
            for dilation in self.dilation_schedule
        )

    @property
    def convolutional_layers(self) -> int:
        return 1 + 2 * len(self.dilation_schedule)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(self.activation(self.norm1(self.conv1(values))))
        values = self.dropout(self.norm2(self.conv2(values)))
        return self.activation(values + residual)


class DNAEncoderV1(nn.Module):
    def __init__(self, architecture: ArchitectureV1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, architecture.embedding_dim)
        self.stem = nn.Conv1d(
            architecture.embedding_dim,
            architecture.channels,
            architecture.stem_kernel_size,
            stride=architecture.stem_stride,
            padding=architecture.stem_kernel_size // 2,
        )
        self.norm = nn.GroupNorm(architecture.group_norm_groups, architecture.channels)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(
                architecture.channels,
                dilation,
                architecture.group_norm_groups,
                architecture.dropout,
            )
            for dilation in architecture.dilation_schedule
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.embedding(tokens).transpose(1, 2)
        values = self.activation(self.norm(self.stem(values)))
        for block in self.blocks:
            values = block(values)
        return values


class ScalarStudentV1(nn.Module):
    def __init__(self, architecture: ArchitectureV1) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = DNAEncoderV1(architecture)
        self.head = nn.Sequential(
            nn.Linear(2 * architecture.channels, architecture.head_hidden),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.head_hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.encoder(tokens)
        pooled = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)
        return self.head(pooled).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

