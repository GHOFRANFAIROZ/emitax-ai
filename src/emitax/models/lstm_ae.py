from __future__ import annotations
import numpy as np

# LSTM-Autoencoder (PyTorch) — الفحص الزمني. torch يُستورَد داخل الدوال (ثقيل، على Colab).

def build_model(n_features: int, seq_len: int, latent: int = 64, hidden: int = 128):
    import torch
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self, F, T, latent, hidden):
            super().__init__()
            self.T = T
            self.enc1 = nn.LSTM(F, hidden, batch_first=True)
            self.enc2 = nn.LSTM(hidden, latent, batch_first=True)
            self.dec1 = nn.LSTM(latent, latent, batch_first=True)
            self.dec2 = nn.LSTM(latent, hidden, batch_first=True)
            self.out = nn.Linear(hidden, F)

        def forward(self, x):                       # x: (B, T, F)
            h, _ = self.enc1(x)
            _, (hn, _) = self.enc2(h)
            z = hn[-1]                              # (B, latent)
            z = z.unsqueeze(1).expand(-1, x.size(1), -1)
            d, _ = self.dec1(z)
            d, _ = self.dec2(d)
            return self.out(d)                      # (B, T, F)

    return LSTMAutoencoder(n_features, seq_len, latent, hidden)


def train(model, X, epochs: int = 30, lr: float = 1e-3, batch: int = 64, device: str = "cpu"):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for _ in range(epochs):
        model.train(); tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        history.append(tot / len(ds))
    return history


def reconstruction_error(model, X, device: str = "cpu") -> np.ndarray:
    import torch
    model = model.to(device).eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        err = ((model(xb) - xb) ** 2).mean(dim=(1, 2))
    return err.cpu().numpy()


def export_onnx(model, seq_len: int, n_features: int, path: str, device: str = "cpu"):
    """batch وحده ديناميكي (seq_len ثابت)؛ dummy بـ batch=2 حتى يميّز الـ tracer الـ batch."""
    import torch
    model = model.to(device).eval()
    dummy = torch.randn(2, seq_len, n_features, device=device)
    torch.onnx.export(model, dummy, path,
                      input_names=["window"], output_names=["reconstruction"],
                      dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
                      opset_version=17, do_constant_folding=True)
    return path