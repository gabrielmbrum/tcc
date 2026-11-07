#!/usr/bin/env python
"""
Ablação da via topológica (protocolo de tex/ablacao.tex).

Modelos : lstm, bilstm, gru, bigru, saclstm, mlp.
Entrada : data/<DS>/topological.npz  (X: (N, 15, 18) medidas ACUMULADAS por raio, y: (N,)).
Protocolo (por dataset):
  1. holdout estratificado 80/20 (SEED=42, idêntico a ablacao_topologica.py); os 20% NUNCA são lidos.
  2. Sobre os 80%: StratifiedKFold(5, shuffle=True, random_state=42). As mesmas 5 partições valem
     para todos os modelos e todas as combinações.
  3. Cada combinação da grade completa é treinada em 5 rodadas (4 partições de treino, 1 de
     validação, rodando a de validação). Padronização z-score por coordenada (15 x 18), ajustada
     só no treino da rodada.
  4. Treino: entropia cruzada, AdamW, norma do gradiente <= 1,0, máx. 50 épocas, parada antecipada
     com paciência 10 sobre a perda de validação, pesos da melhor época restaurados.
  5. Métrica da rodada: F1 macro na partição de validação (com os pesos restaurados). O desempenho
     da combinação é a média das 5 rodadas; o resumo escolhe a melhor combinação por
     (dataset, modelo) e gera as duas tabelas do capítulo.

Grade (tabela tab:espaco-busca-top):
  lr {1e-4, 3e-4, 1e-3, 3e-3} x wd {1e-4, 1e-3, 1e-2} x dropout {0.1, 0.3, 0.5} x h {32, 64, 128}
  x lote {4, 8, 16} (UCSB) ou {16, 32, 64} (CR, LG, NHL, LA)  ->  324 combinações por modelo.

Embedding h: LSTM/GRU usam h unidades; BiLSTM/BiGRU usam h/2 por direção ([h_fwd_final || h_bwd_final]);
SAC-LSTM: LSTM(h) -> autoatenção de cabeça única sobre os 15 estados -> média sobre os raios;
MLP: Flatten(270) -> Linear(h) -> ReLU. Cabeça comum: LayerNorm -> Linear(128) -> ReLU -> Dropout -> Linear(C).

Saída (resumível; tarefas concluídas são puladas):
  results_topologica/folds/<DS>__<modelo>.csv   (uma linha por combinação x rodada)
  results_topologica/summary_*.csv, tabelas LaTeX (modo --summary)

Exemplos:
  python src/ablacao_via_topologica.py --dataset UCSB --model bilstm
  python src/ablacao_via_topologica.py --dataset UCSB --model saclstm --quick --dummy   # fumaça
  python src/ablacao_via_topologica.py --summary
"""
import argparse
import copy
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FEAT = 18
DATASETS = ["UCSB", "CR", "LA", "LG", "NHL"]
MODELS = ["lstm", "bilstm", "gru", "bigru", "saclstm", "mlp"]
MODEL_LABEL = {"lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU", "bigru": "BiGRU",
               "saclstm": "SAC-LSTM", "mlp": "MLP"}
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
WDS = [1e-4, 1e-3, 1e-2]
DROPS = [0.1, 0.3, 0.5]
HS = [32, 64, 128]
HP_KEYS = ["lr", "wd", "dropout", "h", "bs"]


# ------------------------------------------------------------------ dados
def load(ds, data_dir, dummy):
    p = Path(data_dir) / ds / "topological.npz"
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
    """Mesmo holdout de ablacao_topologica.py; devolve só os índices de desenvolvimento (80%)."""
    idx_dev, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return idx_dev


def standardize(Xtr, Xva):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    return ((Xtr - mu) / sd).astype(np.float32), ((Xva - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------ modelos
class SelfAttention(nn.Module):
    """Autoatenção escalonada de cabeça única sobre os passos (raios), com residual."""

    def __init__(self, h):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.scale = h ** -0.5

    def forward(self, s):  # s: (B, T, h)
        a = torch.softmax(self.q(s) @ self.k(s).transpose(1, 2) * self.scale, dim=-1)
        return s + a @ self.v(s)


class Encoder(nn.Module):
    def __init__(self, kind, T, h):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(N_FEAT, h, batch_first=True)
        elif kind in ("bilstm", "bigru"):
            self.rnn = (nn.LSTM if kind == "bilstm" else nn.GRU)(
                N_FEAT, h // 2, batch_first=True, bidirectional=True)
        elif kind == "saclstm":
            self.rnn = nn.LSTM(N_FEAT, h, batch_first=True)
            self.att = SelfAttention(h)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(T * N_FEAT, h), nn.ReLU())
        else:
            raise ValueError(kind)

    def forward(self, x):  # x: (B, T, 18)
        if self.kind == "mlp":
            return self.net(x)
        if self.kind == "saclstm":
            s, _ = self.rnn(x)
            return self.att(s).mean(1)
        _, hn = self.rnn(x)
        if self.kind in ("lstm", "bilstm"):
            hn = hn[0]  # LSTM devolve (h, c)
        # (dirs, B, H): unidirecional -> h_T; bidirecional -> [h_fwd_T || h_bwd_1]
        return torch.cat(list(hn), dim=-1)


class Net(nn.Module):
    def __init__(self, kind, T, n_classes, h, dropout):
        super().__init__()
        self.enc = Encoder(kind, T, h)
        self.ln = nn.LayerNorm(h)
        self.head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.ln(self.enc(x)))


# ------------------------------------------------------------------ treino
def fit_eval(kind, hp, Xtr, ytr, Xva, yva, C, seed, a):
    torch.manual_seed(seed)
    Xtr, Xva = standardize(Xtr, Xva)
    xtr, xva = torch.as_tensor(Xtr, device=a.device), torch.as_tensor(Xva, device=a.device)
    ytr_t, yva_t = torch.as_tensor(ytr, device=a.device), torch.as_tensor(yva, device=a.device)
    model = Net(kind, Xtr.shape[1], C, hp["h"], hp["dropout"]).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), hp["bs"]
    best, best_state, best_ep, wait = float("inf"), None, 1, 0
    for ep in range(a.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(a.device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:  # LayerNorm/Dropout toleram, mas evita lote de 1 amostra espúrio
                continue
            opt.zero_grad()
            F.cross_entropy(model(xtr[b]), ytr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(xva), yva_t).item()
        if vl < best - 1e-4:
            best, best_state, best_ep, wait = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            wait += 1
            if wait >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xva).argmax(1).cpu().numpy()
    return dict(f1=f1_score(yva, pred, average="macro"), acc=accuracy_score(yva, pred),
                val_loss=best, best_ep=best_ep)


def grid(ds):
    bss = [4, 8, 16] if ds == "UCSB" else [16, 32, 64]
    return [dict(zip(HP_KEYS, v)) for v in itertools.product(LRS, WDS, DROPS, HS, bss)]


# ------------------------------------------------------------------ execução
def run(a):
    out = Path(a.out_dir) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{a.dataset}__{a.model}.csv"
    X, y = load(a.dataset, a.data_dir, a.dummy)
    idx = dev_split(y)
    Xd, yd = X[idx], y[idx]  # o holdout de 20% não é lido a partir daqui
    C = len(np.unique(y))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd))
    cfgs = grid(a.dataset)
    if a.quick:
        cfgs = cfgs[::len(cfgs) // 3][:3]
    rows = pd.read_csv(csv).to_dict("records") if (csv.exists() and not a.force) else []
    done = {json.dumps({k: r[k] for k in HP_KEYS}, sort_keys=True) for r in rows
            if int(r["fold"]) == 4}  # combinação concluída = as 5 rodadas gravadas
    t0 = time.time()
    for ci, hp in enumerate(cfgs):
        if json.dumps(hp, sort_keys=True) in done:
            continue
        res = []
        for k, (tr, va) in enumerate(folds):
            r = fit_eval(a.model, hp, Xd[tr], yd[tr], Xd[va], yd[va], C, SEED + k, a)
            res.append(dict(dataset=a.dataset, model=a.model, fold=k, **hp, n_train=len(tr),
                            n_val=len(va), **r))
        rows += res
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{a.dataset}/{a.model}] {ci + 1}/{len(cfgs)} {hp} "
              f"F1={np.mean([r['f1'] for r in res]):.3f} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ resumo
def summary(a):
    files = sorted((Path(a.out_dir) / "folds").glob("*.csv"))
    if not files:
        raise SystemExit("sem resultados em " + str(Path(a.out_dir) / "folds"))
    df = pd.concat([pd.read_csv(f) for f in files])
    per = df.groupby(["dataset", "model"] + HP_KEYS).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), acc_mean=("acc", "mean"),
        ep_mean=("best_ep", "mean"), n_folds=("fold", "nunique")).reset_index()
    per = per[per.n_folds == 5]  # só combinações completas
    per.to_csv(Path(a.out_dir) / "summary_all_configs.csv", index=False)
    best = per.loc[per.groupby(["dataset", "model"])["f1_mean"].idxmax()].reset_index(drop=True)
    best.to_csv(Path(a.out_dir) / "summary_best.csv", index=False)
    print(best[["dataset", "model", "f1_mean", "f1_std"] + HP_KEYS].round(4).to_string(index=False))

    def pm(r):
        return f"{r.f1_mean:.3f} $\\pm$ {r.f1_std:.3f}"

    ds_order = [d for d in ["UCSB", "CR", "LA", "LG", "NHL"] if d in set(best.dataset)]
    lines = ["% tab:abl-f1"]
    for m in MODELS:
        cells = []
        for d in ds_order:
            r = best[(best.dataset == d) & (best.model == m)]
            if r.empty:
                cells.append("\\pend")
                continue
            r = r.iloc[0]
            top = best[best.dataset == d].f1_mean.max()
            s = pm(r)
            cells.append(f"\\textbf{{{r.f1_mean:.3f}}} $\\pm$ {r.f1_std:.3f}" if r.f1_mean == top else s)
        lines.append(f"{MODEL_LABEL[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["", "% tab:abl-top-params"]
    for d in ds_order:
        for m in MODELS:
            r = best[(best.dataset == d) & (best.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{d} & {MODEL_LABEL[m]} & ${r.lr:g}$ & ${r.wd:g}$ & {r.dropout:g} & "
                         f"{int(r.h)} & {int(r.bs)} \\\\")
        lines.append("\\midrule")
    (Path(a.out_dir) / "tabelas.tex").write_text("\n".join(lines) + "\n")
    print(f"\nTabelas LaTeX em {Path(a.out_dir) / 'tabelas.tex'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_topologica"))
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="teste de fumaça (3 combinações)")
    ap.add_argument("--dummy", action="store_true", help="dados sintéticos se não houver .npz")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.summary:
        summary(a)
    else:
        if not (a.dataset and a.model):
            ap.error("--dataset e --model são obrigatórios (ou use --summary)")
        run(a)


if __name__ == "__main__":
    main()
