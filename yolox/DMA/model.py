import torch
import torch.nn as nn
import numpy as np


class DynamicWeightNet(nn.Module):
    """
    Per-pair reliability weight estimator for motion-appearance cost fusion.

    Input:  feature vector describing one (track, detection) candidate pair.
    Output: [w_motion, w_reid] that sum to 1 via softmax.
    """

    def __init__(self, input_dim: int = 6, hidden_dims: tuple = (64), dropout: float = 0.2):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)
        Returns:
            weights: (N, 2)  [w_motion, w_reid], each row sums to 1
        """
        return torch.softmax(self.net(x), dim=-1)

    @torch.no_grad()
    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """Convenience wrapper for numpy input during tracker inference."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(x_np).float().to(device)
        return self(x).cpu().numpy()

    def save(self, path: str, stats: dict = None):
        arch = {"hidden_dims": list(self.hidden_dims), "dropout": self.dropout}
        torch.save({"state_dict": self.state_dict(), "stats": stats, "arch": arch}, path)

    @classmethod
    def load(cls, path: str, input_dim: int = None, hidden_dims: tuple = None):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["state_dict"]
        arch = ckpt.get("arch")
        if arch is not None:
            # Checkpoints saved after the MLP hyperparameter search (tune_mlp.py)
            # record their own architecture, so odd depths/widths/dropout load
            # correctly without guesswork.
            if input_dim is None:
                input_dim = sd["net.0.weight"].shape[1]
            if hidden_dims is None:
                hidden_dims = tuple(arch["hidden_dims"])
            dropout = arch.get("dropout", 0.0)
        else:
            # Older checkpoints (no "arch" key, always dropout=0): infer the
            # architecture from the state_dict itself. LayerNorm also has a
            # "weight" entry but it's 1D, so filtering on dim() == 2 keeps
            # only the Linear layers' output widths.
            import re
            layer_idxs = sorted(
                int(m.group(1)) for k in sd
                if (m := re.match(r"net\.(\d+)\.weight$", k))
            )
            linear_out_dims = [
                sd[f"net.{idx}.weight"].shape[0]
                for idx in layer_idxs
                if sd[f"net.{idx}.weight"].dim() == 2
            ]
            if input_dim is None:
                input_dim = sd["net.0.weight"].shape[1]
            if hidden_dims is None:
                hidden_dims = tuple(linear_out_dims[:-1])  # drop the final 2-class output layer
            dropout = 0.0
        model = cls(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        model.load_state_dict(sd)
        model.eval()
        stats = ckpt.get("stats", None)
        return model, stats
