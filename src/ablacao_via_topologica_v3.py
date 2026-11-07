#!/usr/bin/env python
"""
Ablação da via topológica, v3. Reaproveita dados, partições, modelos e loop de treino de
ablacao_via_topologica.py (mesmos 5 folds e mesmo holdout 80/20), com estas mudanças:

  * lr estendido: {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}   (1e-4 nunca foi o melhor; 3e-3 era o teto)
  * h  estendido: {64, 128, 256, 512}               (128 era o teto)
  * wd FIXO em 1e-4 (efeito desprezível na v2; 1e-4 foi o melhor em 20 das 30 combinações)
  * max_epochs 100 (antes 50: várias combinações terminavam no teto sem convergir)
  * grade: 5 lr x 3 dropout x 4 h x 3 lotes = 180 combinações por modelo
  * seleção da melhor combinação por (dataset, modelo) pelo F1 médio do CV (como antes) e
    AVALIAÇÃO FINAL, UMA VEZ, no holdout de 20% (--holdout), que na v2 nunca foi usado.
    O F1 do CV é otimista (selecionado entre 180 combinações no mesmo CV); o do holdout não.
  * --check-leakage: auditoria dos vetores topológicos (duplicatas entre folds).

Uso:
  python src/ablacao_via_topologica_v3.py --dataset UCSB --model mlp
  python src/ablacao_via_topologica_v3.py --summary           # melhor combinação por (ds, modelo)
  python src/ablacao_via_topologica_v3.py --holdout           # avalia as melhores no holdout 20%
  python src/ablacao_via_topologica_v3.py --check-leakage
"""
import argparse
import itertools
import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablacao_via_topologica as base  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = base.ROOT
SEED = base.SEED
DATASETS, MODELS, MODEL_LABEL = base.DATASETS, base.MODELS, base.MODEL_LABEL
LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
DROPS = [0.1, 0.3, 0.5]
HS = [64, 128, 256, 512]
WD = 1e-4
HP_KEYS = ["lr", "dropout", "h", "bs"]
HOLDOUT_SEEDS = 5


def grid(ds):
    bss = [4, 8, 16] if ds == "UCSB" else [16, 32, 64]
    return [dict(zip(HP_KEYS, v)) for v in itertools.product(LRS, DROPS, HS, bss)]


def key(hp):
    return json.dumps({k: hp[k] for k in HP_KEYS}, sort_keys=True)


def folds_of(y):
    """Idênticas às da v2: holdout 80/20 (SEED) e StratifiedKFold(5) sobre os 80%."""
    idx = base.dev_split(y)
    Xy = np.zeros(len(idx))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xy, y[idx]))
    return idx, folds


# ------------------------------------------------------------------ CV
def run(a):
    out = Path(a.out_dir) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{a.dataset}__{a.model}.csv"
    X, y = base.load(a.dataset, a.data_dir, a.dummy)
    idx, folds = folds_of(y)
    Xd, yd = X[idx], y[idx]  # o holdout não é lido aqui
    C = len(np.unique(y))
    cfgs = grid(a.dataset)
    if a.quick:
        cfgs = cfgs[::len(cfgs) // 3][:3]
    rows = pd.read_csv(csv).to_dict("records") if (csv.exists() and not a.force) else []
    done = {key(r) for r in rows if int(r["fold"]) == 4}
    t0 = time.time()
    for ci, hp in enumerate(cfgs):
        if key(hp) in done:
            continue
        full = dict(hp, wd=WD)
        res = []
        for k, (tr, va) in enumerate(folds):
            r = base.fit_eval(a.model, full, Xd[tr], yd[tr], Xd[va], yd[va], C, SEED + k, a)
            res.append(dict(dataset=a.dataset, model=a.model, fold=k, **hp, wd=WD,
                            n_train=len(tr), n_val=len(va), **r))
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
    per = per[per.n_folds == 5]
    per.to_csv(Path(a.out_dir) / "summary_all_configs.csv", index=False)
    best = per.loc[per.groupby(["dataset", "model"])["f1_mean"].idxmax()].reset_index(drop=True)
    best.to_csv(Path(a.out_dir) / "summary_best.csv", index=False)
    print(best[["dataset", "model", "f1_mean", "f1_std"] + HP_KEYS + ["ep_mean"]]
          .round(4).to_string(index=False))
    # sinaliza melhores que ficaram na borda da grade (ótimo possivelmente fora dela)
    edge = best[(best.lr == max(LRS)) | (best.h == max(HS)) | (best.lr == min(LRS))]
    if len(edge):
        print(f"\nATENÇÃO: {len(edge)} melhores combinações ainda estão na borda da grade (lr ou h).")


# ------------------------------------------------------------------ holdout
def fit_fixed(kind, hp, Xtr, ytr, Xte, yte, C, seed, epochs, device):
    """Treina no desenvolvimento (80%) por `epochs` épocas fixas, sem olhar o teste."""
    torch.manual_seed(seed)
    Xtr, Xte = base.standardize(Xtr, Xte)  # estatísticas só do treino
    xtr, xte = torch.as_tensor(Xtr, device=device), torch.as_tensor(Xte, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)
    model = base.Net(kind, Xtr.shape[1], C, int(hp["h"]), hp["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), int(hp["bs"])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(xtr[b]), ytr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte).argmax(1).cpu().numpy()
    return f1_score(yte, pred, average="macro"), accuracy_score(yte, pred)


def holdout(a):
    best = pd.read_csv(Path(a.out_dir) / "summary_best.csv")
    rows = []
    for _, r in best.iterrows():
        X, y = base.load(r.dataset, a.data_dir, a.dummy)
        idx_dev = base.dev_split(y)
        idx_te = np.setdiff1d(np.arange(len(y)), idx_dev)
        epochs = max(1, int(round(r.ep_mean)))  # épocas típicas de parada no CV
        C = len(np.unique(y))
        res = [fit_fixed(r.model, r, X[idx_dev], y[idx_dev], X[idx_te], y[idx_te], C,
                         SEED + s, epochs, a.device) for s in range(HOLDOUT_SEEDS)]
        f1s = [x[0] for x in res]
        rows.append(dict(dataset=r.dataset, model=r.model, cv_f1=r.f1_mean,
                         holdout_f1=np.mean(f1s), holdout_std=np.std(f1s),
                         holdout_acc=np.mean([x[1] for x in res]), n_test=len(idx_te),
                         epochs=epochs))
        print(f"[{r.dataset}/{r.model}] CV={r.f1_mean:.3f}  holdout={np.mean(f1s):.3f}"
              f"±{np.std(f1s):.3f}  (n_test={len(idx_te)})", flush=True)
    h = pd.DataFrame(rows)
    h.to_csv(Path(a.out_dir) / "holdout_results.csv", index=False)
    print("\nGap médio (CV - holdout):", round((h.cv_f1 - h.holdout_f1).mean(), 4))
    print(h.pivot(index="model", columns="dataset", values="holdout_f1").round(3).to_string())


# ------------------------------------------------------------------ vazamento
def check_leakage(a):
    """Vetores topológicos são por imagem (sem estatística entre imagens), então a geração prévia
    não vaza. O risco real é imagem duplicada ou do mesmo paciente/lâmina em folds distintos."""
    for ds in DATASETS:
        p = Path(a.data_dir) / ds / "topological.npz"
        if not p.exists():
            print(f"[{ds}] sem {p}")
            continue
        d = np.load(p, allow_pickle=True)
        X, y = d["X"].reshape(len(d["X"]), -1), d["y"]
        Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
        D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        nn_i, nn_d = D.argmin(1), D.min(1)
        thr = np.percentile(D[np.isfinite(D)], 0.5)
        exact = int((nn_d < 1e-6).sum())
        close = int((nn_d < thr).sum())
        same_cls = float((y[nn_i] == y).mean())
        print(f"[{ds}] N={len(y)}  duplicatas exatas={exact}  vizinhos muito próximos={close}  "
              f"vizinho mais próximo da mesma classe={same_cls:.2f}")
        if "files" in d.files:
            print(f"        exemplo de arquivos: {list(d['files'][:3])} "
                  "(veja se o nome indica paciente/lâmina/animal repetido)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--check-leakage", action="store_true")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_topologica_v3"))
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.check_leakage:
        check_leakage(a)
    elif a.summary:
        summary(a)
    elif a.holdout:
        holdout(a)
    else:
        if not (a.dataset and a.model):
            ap.error("--dataset e --model são obrigatórios")
        run(a)


if __name__ == "__main__":
    main()
