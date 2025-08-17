import torch.nn as nn
import torch


class FixedMLP(nn.Module):
    def __init__(self, input_dim: int, pooling_fn):
        super().__init__()
        self.pooling_fn = pooling_fn
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        pooled = self.pooling_fn(x)
        return self.model(pooled).view(-1)


class FixedCombinedMLP(nn.Module):
    def __init__(self, input_dim: int, pooling_fns: list):
        super().__init__()
        assert len(pooling_fns) == 2
        self.pooling_fns = pooling_fns

        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        pooled1 = self.pooling_fns[0](x)
        pooled2 = self.pooling_fns[1](x)
        feat1 = self.branch1(pooled1)
        feat2 = self.branch2(pooled2)

        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(0)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(0)

        combined = torch.cat([feat1, feat2], dim=1)
        return self.combined(combined).view(-1)


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, pooling_fn):
        super().__init__()
        self.pooling_fn = pooling_fn
        layer_dims = self._get_hidden_dims(input_dim)

        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def _get_hidden_dims(self, input_dim: int):
        if input_dim <= 640:
            return [256, 128]
        elif input_dim <= 1152:
            return [512, 256, 128]
        elif input_dim >= 2560:
            return [1024, 512, 256, 128]
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}")

    def forward(self, x):
        pooled = self.pooling_fn(x)
        return self.model(pooled).view(-1)


class DynamicCombinedMLP(nn.Module):
    def __init__(self, input_dim: int, pooling_fns: list):
        super().__init__()
        assert len(pooling_fns) == 2
        self.pooling_fns = pooling_fns

        # Create branches
        self.branch1, out_dim1 = self._make_branch(input_dim)
        self.branch2, out_dim2 = self._make_branch(input_dim)

        # Combined layer expects concatenated features
        self.combined = nn.Sequential(
            nn.Linear(out_dim1 + out_dim2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _make_branch(self, input_dim: int):
        layer_dims = self._get_hidden_dims(input_dim)
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        return nn.Sequential(*layers), prev_dim  # Return the final output dimension

    def _get_hidden_dims(self, input_dim: int):
        if input_dim <= 640:
            return [256, 128]
        elif input_dim <= 1152:
            return [512, 256, 128]
        elif input_dim >= 2560:
            return [1024, 512, 256, 128]

        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}")

    def forward(self, x):
        pooled1 = self.pooling_fns[0](x)
        pooled2 = self.pooling_fns[1](x)

        feat1 = self.branch1(pooled1)
        feat2 = self.branch2(pooled2)

        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(0)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(0)

        combined = torch.cat([feat1, feat2], dim=1)
        return self.combined(combined).view(-1)
