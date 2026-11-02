#!/usr/bin/env python
"""
Ablação da via topológica isolada (grupo 2 de "Comparações e ablações").

Estrutura esperada (a partir da raiz do projeto, um nível acima de src/):
    data/UCSB/topological.npz     (idem para CR, LA, LG, NHL)
com
    X : (N, 15, 18) float  -> medidas topológicas ACUMULADAS por raio (sem padronizar)
    y : (N,) int           -> rótulos 0..C-1
Resultados em results/ (criado automaticamente).
(use --dummy para testar o pipeline com dados sintéticos)

Protocolo: holdout 20% estratificado (seed fixa) NUNCA é tocado aqui.
Todas as decisões usam apenas os 80% restantes, por CV estratificada 5-fold
(opcionalmente repetida), com as mesmas partições para todas as configurações.

Estágios:
  enc  : {bilstm, lstm, gru, mlp, cnn1d} x {cum, diff}          (r_max = 15)
  rmax : r_max in {4, 7, 10, 15} para --enc e --rep escolhidos
  hp   : grade pequena de hiperparâmetros para --enc e --rep escolhidos

Exemplos:
  python src/ablacao_topologica.py --stage enc --dummy
  python src/ablacao_topologica.py --stage rmax --enc bilstm --rep diff
  python src/ablacao_topologica.py --stage hp --enc bilstm --rep cum --repeats 3
"""
import argparse
import copy
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (pasta com data/ e src/)
FEATURES_FILE = "topological.npz"
SEED = 42  # reutilize ESTA divisão de holdout no pipeline completo
DATASETS = ["UCSB", "CR", "LA", "LG", "NHL"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FEAT = 18


# ----------------------------------------------------------------- dados
def load(ds, data_dir, dummy):
    p = Path(data_dir) / ds / FEATURES_FILE
    if p.exists():
        d = np.load(p)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)
    if not dummy:
        raise FileNotFoundError(p)
    n = {"UCSB": 58, "CR": 165, "LA": 528, "LG": 265, "NHL": 374}[ds]
    c = {"UCSB": 2, "CR": 2, "LA": 4, "LG": 2, "NHL": 3}[ds]
    rng = np.random.default_rng(0)
    y = rng.integers(0, c, n)
    X = np.cumsum(rng.random((n, 15, N_FEAT)) + y[:, None, None] * 0.05, axis=1)
    return X.astype(np.float32), y.astype(np.int64)


def dev_split(y):
    """Separa o holdout (nunca usado aqui) e devolve os índices de desenvolvimento."""
    idx_dev, _ = train_test_split(np.arange(len(y)), test_size=0.2,
                                  stratify=y, random_state=SEED)
    return idx_dev


def prep(X, rep, rmax):
    """Recorta em r_max (medidas acumuladas: os r primeiros passos bastam) e
    opcionalmente converte em diferenças entre raios consecutivos."""
    X = X[:, :rmax]
    if rep == "diff":
        X = np.concatenate([X[:, :1], np.diff(X, axis=1)], axis=1)
    return X


# ----------------------------------------------------------------- modelos
class Encoder(nn.Module):
    """Todos os codificadores produzem embedding de 128 dims (comparação justa)."""

    def __init__(self, kind, T, hidden=64, emb=128):
        super().__init__()
        self.kind = kind
        if kind in ("bilstm", "lstm", "gru"):
            bi = kind == "bilstm"
            h = hidden if bi else hidden * 2  # uni: dobra h p/ manter 128 dims
            cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = cls(N_FEAT, h, batch_first=True, bidirectional=bi)
            self.out_dim = h * (2 if bi else 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(T * N_FEAT, emb), nn.ReLU())
            self.out_dim = emb
        elif kind == "cnn1d":
            self.net = nn.Sequential(
                nn.Conv1d(N_FEAT, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, emb, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.out_dim = emb
        else:
            raise ValueError(kind)

    def forward(self, x):  # x: (B, T, 18)
        if self.kind in ("bilstm", "lstm", "gru"):
            _, h = self.rnn(x)
            if self.kind != "gru":
                h = h[0]  # LSTM devolve (h, c)
            # h: (dirs, B, H) -> [h_fwd_final || h_bwd_final] (= eq. do embedding topológico)
            return torch.cat(list(h), dim=-1)
        if self.kind == "cnn1d":
            x = x.transpose(1, 2)
        return self.net(x)


class Net(nn.Module):
    def __init__(self, kind, T, n_classes, hidden, dropout):
        super().__init__()
        self.enc = Encoder(kind, T, hidden)
        self.ln = nn.LayerNorm(self.enc.out_dim)
        self.head = nn.Sequential(nn.Linear(self.enc.out_dim, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.ln(self.enc(x)))


# ----------------------------------------------------------------- treino
def metrics(y, proba):
    pred = proba.argmax(1)
    try:
        auc = (roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2
               else roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = np.nan
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"), auc=auc)


def fit_eval(Xtr, ytr, Xva, yva, n_classes, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # padronização por coordenada (T x 18), ajustada SÓ no treino da rodada
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xtr = torch.tensor((Xtr - mu) / sd, device=DEVICE)
    xva = torch.tensor((Xva - mu) / sd, device=DEVICE)
    ytr_t = torch.tensor(ytr, device=DEVICE)
    yva_t = torch.tensor(yva, device=DEVICE)

    model = Net(cfg["enc"], Xtr.shape[1], n_classes, cfg["hidden"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bs = 8 if len(ytr) < 100 else 16  # UCSB: treino ~29-37 imagens

    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(ytr), device=DEVICE)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(xtr[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(xva), yva_t).item()
        if vl < best - 1e-4:
            best, best_state, best_ep, wait = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(xva), dim=1).cpu().numpy()
    return {**metrics(yva, proba), "best_ep": best_ep + 1, "val_loss": best}


# ----------------------------------------------------------------- experimentos
def make_configs(args):
    base = dict(enc="bilstm", rep="cum", rmax=15, hidden=64, lr=1e-3, wd=1e-4,
                dropout=0.5, max_epochs=args.max_epochs, patience=args.patience)
    if args.stage == "enc":
        return [{**base, "enc": e, "rep": r}
                for e in ["bilstm", "lstm", "gru", "mlp", "cnn1d"] for r in ["cum", "diff"]]
    base.update(enc=args.enc, rep=args.rep)
    if args.stage == "rmax":
        return [{**base, "rmax": r} for r in [4, 7, 10, 15]]
    if args.stage == "hp":
        return [{**base, "lr": lr, "hidden": h, "dropout": d, "wd": wd}
                for lr, h, d, wd in itertools.product([1e-3, 3e-4], [32, 64, 128],
                                                      [0.3, 0.5], [1e-4, 1e-2])]
    raise ValueError(args.stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["enc", "rmax", "hp"], required=True)
    ap.add_argument("--enc", default="bilstm")
    ap.add_argument("--rep", default="cum", choices=["cum", "diff"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    ap.add_argument("--repeats", type=int, default=1, help="repetições da CV 5-fold")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    configs = make_configs(args)
    rows = []
    t0 = time.time()
    for ds in args.datasets:
        X, y = load(ds, args.data_dir, args.dummy)
        idx = dev_split(y)
        Xd, yd = X[idx], y[idx]
        n_classes = len(np.unique(y))
        # as MESMAS partições para todas as configurações
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                              random_state=SEED).split(Xd, yd))
        for cfg in configs:
            Xc = prep(Xd, cfg["rep"], cfg["rmax"])
            for k, (tr, va) in enumerate(splits):
                r = fit_eval(Xc[tr], yd[tr], Xc[va], yd[va], n_classes, cfg, seed=SEED + k)
                rows.append({"dataset": ds, "fold": k, **{c: cfg[c] for c in
                             ["enc", "rep", "rmax", "hidden", "lr", "wd", "dropout"]}, **r})
            print(f"[{time.time() - t0:6.0f}s] {ds} {cfg['enc']}/{cfg['rep']} "
                  f"rmax={cfg['rmax']} h={cfg['hidden']} lr={cfg['lr']} "
                  f"-> F1={np.mean([x['f1'] for x in rows[-len(splits):]]):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/{args.stage}_raw.csv", index=False)

    keys = ["enc", "rep", "rmax", "hidden", "lr", "wd", "dropout"]
    summ = df.groupby(["dataset"] + keys).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        acc_mean=("acc", "mean"), auc_mean=("auc", "mean"),
        ep_mean=("best_ep", "mean")).reset_index()
    summ.to_csv(f"{args.out_dir}/{args.stage}_summary.csv", index=False)

    # ranking médio entre datasets (1 = melhor F1 no dataset)
    summ["rank"] = summ.groupby("dataset")["f1_mean"].rank(ascending=False)
    rk = summ.groupby(keys)["rank"].mean().sort_values().reset_index()
    print("\n=== ranking médio entre datasets (menor = melhor) ===")
    print(rk.to_string(index=False))
    print("\n=== F1 macro médio por dataset ===")
    print(summ.pivot_table(index=keys, columns="dataset", values="f1_mean").round(3).to_string())


if __name__ == "__main__":
    main()
