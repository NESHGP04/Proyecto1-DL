"""
MLP en PyTorch para regresión sobre log1p(SalePrice) + utilidades de entrenamiento.

El MLP es totalmente parametrizable (capas, neuronas, activación, dropout,
batchnorm) para poder documentar las iteraciones que pide el informe (sección 2.3).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """MLP configurable para regresión (1 salida)."""

    def __init__(self, in_features: int, hidden=(256, 128), activation="relu",
                 dropout=0.0, batchnorm=False):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, prev = [], in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_mlp(
    X_train, y_train, X_val, y_val, *,
    hidden=(256, 128), activation="relu", dropout=0.0, batchnorm=False,
    lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=500, patience=30,
    loss="huber", scheduler=True, seed=42, device=None, verbose=False,
):
    """Entrena un MLP con early stopping. Devuelve (modelo, historial, best_val_rmse).

    X/y en ESPACIO LOG del target. El RMSE reportado aquí es RMSE en log.

    Parámetros clave para las iteraciones del informe:
      loss      : 'huber' (robusto a outliers) o 'mse'.
      scheduler : ReduceLROnPlateau (baja lr x0.5 al estancarse la val).
    """
    set_seed(seed)
    device = device or get_device()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], hidden, activation, dropout, batchnorm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15, min_lr=1e-5)
             if scheduler else None)
    loss_fn = nn.SmoothL1Loss() if loss == "huber" else nn.MSELoss()

    hist = {"train_rmse": [], "val_rmse": []}
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_v = loss_fn(model(xb), yb)
            loss_v.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_pred = model(Xtr.to(device)).cpu().numpy()
            va_pred = model(Xva).cpu().numpy()
        tr_rmse = rmse(y_train, tr_pred)
        va_rmse = rmse(y_val, va_pred)
        hist["train_rmse"].append(tr_rmse)
        hist["val_rmse"].append(va_rmse)
        if sched is not None:
            sched.step(va_rmse)

        if va_rmse < best_val - 1e-5:
            best_val, best_state, wait = va_rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch} (best val RMSE={best_val:.4f})")
                break
        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:3d} | train {tr_rmse:.4f} | val {va_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


@torch.no_grad()
def predict_log(model: nn.Module, X, device=None) -> np.ndarray:
    """Predicción en espacio log."""
    device = device or get_device()
    model.eval().to(device)
    xb = torch.tensor(X, dtype=torch.float32).to(device)
    return model(xb).cpu().numpy()
