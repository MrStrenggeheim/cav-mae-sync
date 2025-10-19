# Cross-modal translators for anomaly detection:
# - Trains two MLP regressors: video->audio and audio->video
# - Trains on real-only pairs; evaluates on held-out reals + all fakes
# - Anomaly score = prediction error (MSE + optional cosine residual)

import ast
import math
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from sklearn.metrics import roc_auc_score, average_precision_score

# --------------------------
# Utils
# --------------------------
def to_array(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception:
            return np.fromstring(x.strip("[]"), sep=",", dtype=np.float32)
    raise TypeError(f"Unsupported emb type: {type(x)}")

def build_xy_from_df(df: pd.DataFrame,
                     v_col="video_emb", a_col="audio_emb", y_col="target") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    V = np.stack([to_array(v) for v in df[v_col].values], axis=0)
    A = np.stack([to_array(a) for a in df[a_col].values], axis=0)
    y = df[y_col].values.astype(np.int64)
    return torch.from_numpy(V), torch.from_numpy(A), torch.from_numpy(y)

def fit_std(X: torch.Tensor):
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, std

def apply_std(X: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    return (X - mu) / std

# --------------------------
# Models
# --------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=(512, 256), d_out=768, dropout=0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in d_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CrossModalTranslator(L.LightningModule):
    """
    Two heads:
      f_va: video -> audio
      f_av: audio -> video
    Loss (on reals):
      L = MSE(f_va(V), A) + MSE(f_av(A), V)
        + λ_cyc [ MSE(f_va(f_av(A)), A) + MSE(f_av(f_va(V)), V) ] (optional)
        + λ_cos [ 1 - cos(f_va(V), A) + 1 - cos(f_av(A), V) ] (optional)
    """
    def __init__(self,
                 d=768,
                 hidden=(512, 256),
                 lr=1e-3,
                 lambda_cycle=0.1,
                 lambda_cos=0.1,
                 dropout=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.f_va = MLP(d, hidden, d_out=d, dropout=dropout)
        self.f_av = MLP(d, hidden, d_out=d, dropout=dropout)
        self.mse = nn.MSELoss(reduction="mean")
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, V, A):
        A_hat = self.f_va(V)
        V_hat = self.f_av(A)
        return A_hat, V_hat

    def _loss(self, V, A):
        A_hat = self.f_va(V)
        V_hat = self.f_av(A)
        # base reconstruction losses
        loss_va = self.mse(A_hat, A)
        loss_av = self.mse(V_hat, V)
        loss = loss_va + loss_av

        # cycle consistency
        if self.hparams.lambda_cycle > 0:
            A_cyc = self.f_va(V_hat)
            V_cyc = self.f_av(A_hat)
            loss += self.hparams.lambda_cycle * (self.mse(A_cyc, A) + self.mse(V_cyc, V))

        # cosine residuals
        if self.hparams.lambda_cos > 0:
            # 1 - cosine(pred, target) averaged
            cos_va = 1.0 - self.cos(A_hat, A).mean()
            cos_av = 1.0 - self.cos(V_hat, V).mean()
            loss += self.hparams.lambda_cos * (cos_va + cos_av)

        logs = {"loss": loss, "loss_va": loss_va, "loss_av": loss_av}
        return loss, logs

    def training_step(self, batch, batch_idx):
        V, A = batch
        loss, logs = self._loss(V, A)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        V, A = batch
        loss, logs = self._loss(V, A)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        V, A = batch
        with torch.no_grad():
            A_hat = self.f_va(V)
            V_hat = self.f_av(A)
            # per-sample MSE
            mse_va = ((A_hat - A) ** 2).mean(dim=1)
            mse_av = ((V_hat - V) ** 2).mean(dim=1)
            # cosine residuals: 1 - cos(pred, target)
            cos = nn.CosineSimilarity(dim=1, eps=1e-8)
            cos_va = 1.0 - cos(A_hat, A)
            cos_av = 1.0 - cos(V_hat, V)
            # combine
            score = 0.5 * (mse_va + mse_av) + 0.5 * (cos_va + cos_av)
        return {
            "mse_va": mse_va.cpu().numpy(),
            "mse_av": mse_av.cpu().numpy(),
            "cos_va": cos_va.cpu().numpy(),
            "cos_av": cos_av.cpu().numpy(),
            "score":  score.cpu().numpy(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# --------------------------
# End-to-end routine
# --------------------------
def run_cross_modal(df: pd.DataFrame,
                    v_col="video_emb", a_col="audio_emb", y_col="target",
                    max_epochs=40,
                    batch_size=256,
                    val_frac=0.15,
                    test_frac=0.15,
                    lr=1e-3,
                    lambda_cycle=0.1,
                    lambda_cos=0.1,
                    hidden=(512, 256),
                    dropout=0.0,
                    seed=42,
                    log_dir="lightning_logs"):
    L.seed_everything(seed, workers=True)

    V_all, A_all, y_all = build_xy_from_df(df, v_col=v_col, a_col=a_col, y_col=y_col)
    d = V_all.shape[1]  # 768

    # Split only on REALS
    real_idx = (y_all == 0).nonzero(as_tuple=True)[0]
    fake_idx = (y_all == 1).nonzero(as_tuple=True)[0]
    real_idx = real_idx[torch.randperm(len(real_idx), generator=torch.Generator().manual_seed(seed))]

    n_real = len(real_idx)
    n_val = max(1, int(math.floor(val_frac * n_real)))
    n_test_real = max(1, int(math.floor(test_frac * n_real)))
    n_train = max(1, n_real - n_val - n_test_real)

    idx_train_real = real_idx[:n_train]
    idx_val_real   = real_idx[n_train:n_train + n_val]
    idx_test_real  = real_idx[n_train + n_val:n_train + n_val + n_test_real]
    idx_test = torch.cat([idx_test_real, fake_idx], dim=0)

    # Standardize per modality using TRAIN-REAL stats only
    V_mu, V_std = fit_std(V_all[idx_train_real])
    A_mu, A_std = fit_std(A_all[idx_train_real])

    def stdV(idxs): return apply_std(V_all[idxs], V_mu, V_std)
    def stdA(idxs): return apply_std(A_all[idxs], A_mu, A_std)

    V_tr, A_tr = stdV(idx_train_real), stdA(idx_train_real)
    V_va, A_va = stdV(idx_val_real),   stdA(idx_val_real)
    V_te, A_te = stdV(idx_test),       stdA(idx_test)

    train_loader = DataLoader(TensorDataset(V_tr, A_tr), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader   = DataLoader(TensorDataset(V_va, A_va), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(TensorDataset(V_te, A_te), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = CrossModalTranslator(
        d=d, hidden=hidden, lr=lr,
        lambda_cycle=lambda_cycle, lambda_cos=lambda_cos,
        dropout=dropout
    )

    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="xmodal-{epoch:02d}-{val_loss:.4f}")
    es = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    logger = CSVLogger(save_dir=log_dir, name="cross_modal")

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[ckpt, es],
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # Best checkpoint (optional)
    best = ckpt.best_model_path or None
    if best:
        model = CrossModalTranslator.load_from_checkpoint(best)

    # ---- Evaluation on held-out reals + all fakes ----
    preds = trainer.predict(model, test_loader)
    mse_va = np.concatenate([p["mse_va"] for p in preds])
    mse_av = np.concatenate([p["mse_av"] for p in preds])
    cos_va = np.concatenate([p["cos_va"] for p in preds])
    cos_av = np.concatenate([p["cos_av"] for p in preds])
    score  = np.concatenate([p["score"]  for p in preds])

    y_test = y_all[idx_test].cpu().numpy()

    def safe_auc(y, s): return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
    def safe_ap(y, s):  return average_precision_score(y, s) if len(np.unique(y)) > 1 else np.nan

    metrics = {
        "ROC-AUC (score)": safe_auc(y_test, score),
        "AP (score)":      safe_ap(y_test, score),
        "ROC-AUC (mse_va)": safe_auc(y_test, mse_va),
        "AP (mse_va)":      safe_ap(y_test, mse_va),
        "ROC-AUC (mse_av)": safe_auc(y_test, mse_av),
        "AP (mse_av)":      safe_ap(y_test, mse_av),
        "n_test": len(y_test),
        "n_fakes": int((y_test == 1).sum()),
        "n_reals": int((y_test == 0).sum()),
    }

    print("\n=== Cross-Modal Anomaly Detection (held-out reals + all fakes) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    return {
        "model": model,
        "V_mu": V_mu, "V_std": V_std,
        "A_mu": A_mu, "A_std": A_std,
        "idx_splits": {
            "train_real": idx_train_real, "val_real": idx_val_real,
            "test_real": idx_test_real, "test_all": idx_test
        },
        "scores": {
            "mse_va": mse_va, "mse_av": mse_av,
            "cos_va": cos_va, "cos_av": cos_av,
            "score": score
        },
        "y_test": y_test,
        "metrics": metrics,
        "best_ckpt": best,
    }

# --------------------------
# Example usage
# --------------------------
# df = ...  # DataFrame with columns: 'video_emb', 'audio_emb', 'target'
# results_xm = run_cross_modal(df,
#                              v_col="video_emb", a_col="audio_emb", y_col="target",
#                              max_epochs=40, batch_size=256,
#                              lambda_cycle=0.1, lambda_cos=0.1)
