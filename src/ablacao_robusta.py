#!/usr/bin/env python
"""
Ablação robusta da via topológica isolada (v2).

Corrige as fraquezas do primeiro teste (ablacao_topologica.py):
  1. Hiperparâmetros: antes só o MLP foi ajustado, e a BiLSTM rodou com valores padrão. Agora
     CADA codificador recebe o mesmo orçamento de busca (--trials configurações; a primeira é a
     configuração padrão antiga) sobre o mesmo espaço compartilhado (lr, wd, dropout,
     dimensão do embedding, lote) mais UM eixo próprio da família (camadas na RNN, profundidade
     no MLP, kernel na CNN). Os sorteios compartilhados são idênticos entre codificadores.
  2. Seleção otimista: antes o mesmo F1 de CV escolhia a configuração e era reportado. Agora a
     CV é ANINHADA: em cada fold externo, a seleção usa só o treino desse fold (CV interna) e o
     fold externo avalia uma única vez.
  3. Parada antecipada no fold que mede a métrica: agora a parada antecipada ocorre só na CV
     interna; o ajuste final no treino externo usa um número FIXO de épocas (média das épocas
     ótimas internas), como no protocolo da tese, e o fold externo nunca é visto.
  4. 'hidden' não tinha efeito no MLP: agora 'emb' (dimensão do embedding) vale para todos.
  5. Sem referência: inclui baselines (regressão logística, SVM-RBF, floresta aleatória, kNN e
     classificador de maioria) sobre as mesmas 270 características, mais um teste de sanidade com
     rótulos permutados (--permute), que deve render desempenho de acaso.
  6. Cada fold externo também avalia a configuração padrão, o que mede o ganho do ajuste.

O holdout de 20% (mesma divisão da v1, SEED=42) NUNCA é usado aqui.
As partições externas são idênticas entre codificadores (mesmo SEED), o que permite testes
pareados fold a fold em analise_ablacao.py.

Entrada: data/<DS>/topological.npz (X: (N, 15, 18), y: (N,)); com --res 256, topological_256.npz.
Saída:   results_v2/folds/<DS>__<enc>__<rep>__r<rmax>[__res256]__R<repeats>.csv
         (uma linha por fold externo; o nº de repetições e a resolução fazem parte do nome e das colunas).

Exemplos:
  python src/ablacao_robusta.py --dataset UCSB --enc bilstm --rep diff
  python src/ablacao_robusta.py --dataset NHL --enc mlp --rep cum --rmax 7
  python src/ablacao_robusta.py --dataset LA --enc bilstm --rep cum --permute
  python src/ablacao_robusta.py --dataset UCSB --enc gru --quick --dummy     # teste de fumaça
"""
import argparse
import copy
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42  # mesma divisão de holdout da v1 e do pipeline completo
N_FEAT = 18
RNN_ENCS = ("bilstm", "lstm", "gru")
NN_ENCS = RNN_ENCS + ("mlp", "cnn1d")
SK_ENCS = ("logreg", "svm", "rf", "knn", "dummy")
ALL_ENCS = NN_ENCS + SK_ENCS
NLL_CLIP = 1e-3


# ------------------------------------------------------------------ dados
def npz_name(res):
    return "topological.npz" if res == "native" else f"topological_{res}.npz"


def load(ds, data_dir, dummy, res="native"):
    p = Path(data_dir) / ds / npz_name(res)
    if p.exists():
        d = np.load(p)
        if res != "native":  # garante o mesmo conjunto/ordem de imagens da versão nativa (pareamento)
            nat = Path(data_dir) / ds / npz_name("native")
            if nat.exists():
                dn = np.load(nat)
                if not (np.array_equal(d["files"], dn["files"]) and np.array_equal(d["y"], dn["y"])):
                    raise ValueError(f"{p} não está alinhado com {nat} (arquivos/rótulos diferentes)")
        return d["X"].astype(np.float32), d["y"].astype(np.int64)
    if not dummy:
        raise FileNotFoundError(p)
    n = {"UCSB": 58, "CR": 165, "LA": 528, "LG": 265, "NHL": 374}.get(ds, 60)
    c = {"UCSB": 2, "CR": 2, "LA": 4, "LG": 2, "NHL": 3}.get(ds, 2)
    rng = np.random.default_rng(0)
    y = rng.integers(0, c, n)
    X = np.cumsum(rng.random((n, 15, N_FEAT)) + y[:, None, None] * 0.05, axis=1)
    if res != "native":  # sintético: mesmos rótulos, ruído extra para a comparação não ser trivial
        X = X + np.random.default_rng(1).normal(0, 0.05, X.shape)
    return X.astype(np.float32), y.astype(np.int64)


def dev_split(y):
    """Idêntico à v1: separa o holdout (não usado) e devolve os índices de desenvolvimento."""
    idx_dev, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return idx_dev


def prep(X, rep, rmax):
    """Recorta em r_max (medidas acumuladas) e, opcionalmente, toma diferenças entre raios."""
    X = X[:, :rmax]
    if rep == "diff":
        X = np.concatenate([X[:, :1], np.diff(X, axis=1)], axis=1)
    return X


def standardize(Xtr, *others):
    """Padronização por coordenada (T x 18), ajustada SÓ no treino recebido."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    return [((A - mu) / sd).astype(np.float32) for A in (Xtr,) + others]


# ------------------------------------------------------------------ espaço de busca
def bs_choices(n_train):
    return [8, 16, 32] if n_train < 100 else [16, 32, 64]


def default_cfg(enc, n_train):
    if enc in SK_ENCS:
        return {"logreg": {"C": 1.0}, "svm": {"C": 1.0, "gmul": 1.0},
                "rf": {"max_features": "sqrt", "min_samples_leaf": 1},
                "knn": {"k": 5}, "dummy": {}}[enc]
    c = dict(lr=1e-3, wd=1e-4, dropout=0.5, emb=128, bs=8 if n_train < 100 else 16)
    c.update({"mlp": {"depth": 1}, "cnn1d": {"kernel": 3}}.get(enc, {"layers": 1}))
    return c


def sample_configs(enc, n_trials, n_train, seed=0):
    """A configuração 0 é a padrão. Os sorteios compartilhados (lr, wd, dropout, emb, lote) são
    IDÊNTICOS entre codificadores; o eixo próprio da família usa outro fluxo, também com um
    sorteio por tentativa para todos."""
    cfgs = [default_cfg(enc, n_train)]
    rng, rng_x = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    bsc = bs_choices(n_train)
    for _ in range(n_trials - 1):
        lr = float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3)))
        wd = float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1]))
        dr = float(rng.choice([0.1, 0.3, 0.5]))
        emb = int(rng.choice([32, 64, 128]))
        bs = int(bsc[int(rng.integers(0, 3))])
        ux = int(rng_x.integers(0, 2))
        ur = float(rng_x.uniform(-3, 3))  # fluxo extra para baselines
        if enc in SK_ENCS:
            c = {"logreg": {"C": float(10 ** ur)},
                 "svm": {"C": float(10 ** (ur + 1)), "gmul": [0.1, 1.0, 10.0][int(rng_x.integers(0, 3))]},
                 "rf": {"max_features": ["sqrt", 0.3, 0.5][int(rng_x.integers(0, 3))],
                        "min_samples_leaf": [1, 2, 4][int(rng_x.integers(0, 3))]},
                 "knn": {"k": [1, 3, 5, 9][int(rng_x.integers(0, 4))]},
                 "dummy": {}}[enc]
        else:
            c = dict(lr=lr, wd=wd, dropout=dr, emb=emb, bs=bs)
            if enc == "mlp":
                c["depth"] = 1 + ux
            elif enc == "cnn1d":
                c["kernel"] = [3, 5][ux]
            else:
                c["layers"] = 1 + ux
        cfgs.append(c)
    if enc == "dummy":
        cfgs = cfgs[:1]
    return cfgs


# ------------------------------------------------------------------ modelos
class Encoder(nn.Module):
    """Todos os codificadores produzem embedding de dimensão cfg['emb'] (comparação justa)."""

    def __init__(self, kind, T, cfg):
        super().__init__()
        self.kind, emb = kind, cfg["emb"]
        if kind in RNN_ENCS:
            self.bi = kind == "bilstm"
            H = emb // 2 if self.bi else emb  # BiLSTM: [h_fwd || h_bwd] soma emb
            L = cfg["layers"]
            cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = cls(N_FEAT, H, num_layers=L, batch_first=True, bidirectional=self.bi,
                           dropout=cfg["dropout"] if L > 1 else 0.0)
        elif kind == "mlp":
            layers = [nn.Flatten(), nn.Linear(T * N_FEAT, emb), nn.ReLU()]
            if cfg["depth"] == 2:
                layers += [nn.Dropout(cfg["dropout"]), nn.Linear(emb, emb), nn.ReLU()]
            self.net = nn.Sequential(*layers)
        elif kind == "cnn1d":
            k = cfg["kernel"]
            self.net = nn.Sequential(
                nn.Conv1d(N_FEAT, emb, k, padding=k // 2), nn.ReLU(),
                nn.Conv1d(emb, emb, k, padding=k // 2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten())
        else:
            raise ValueError(kind)

    def forward(self, x):  # x: (B, T, 18)
        if self.kind in RNN_ENCS:
            _, h = self.rnn(x)
            if self.kind != "gru":
                h = h[0]  # LSTM devolve (h, c)
            # h: (camadas*direções, B, H); usa a última camada: [h_fwd_T || h_bwd_1] na BiLSTM
            return torch.cat([h[-2], h[-1]], dim=-1) if self.bi else h[-1]
        if self.kind == "cnn1d":
            x = x.transpose(1, 2)
        return self.net(x)


class Net(nn.Module):
    """Cabeça idêntica para todos (LN -> 128 -> ReLU -> Dropout -> C), como o MLP da tese."""

    def __init__(self, kind, T, n_classes, cfg):
        super().__init__()
        self.enc = Encoder(kind, T, cfg)
        self.ln = nn.LayerNorm(cfg["emb"])
        self.head = nn.Sequential(nn.Linear(cfg["emb"], 128), nn.ReLU(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.ln(self.enc(x)))


def n_params(enc, cfg, T, C):
    if enc in SK_ENCS:
        return 0
    return int(sum(p.numel() for p in Net(enc, T, C, cfg).parameters()))


def fit_nn(enc, cfg, Xtr, ytr, C, seed, device, max_epochs, patience=None, Xva=None, yva=None,
           epochs=None):
    """Com (Xva, yva): parada antecipada pela perda de validação, restaura os melhores pesos.
    Sem validação: treina exatamente `epochs` épocas."""
    torch.manual_seed(seed)
    model = Net(enc, Xtr.shape[1], C, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    xtr = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)
    early = Xva is not None
    if early:
        xva = torch.as_tensor(Xva, device=device)
        yva_t = torch.as_tensor(yva, device=device)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), cfg["bs"]
    best, best_state, best_ep, wait = float("inf"), None, 1, 0
    for ep in range(max_epochs if early else epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(xtr[b]), ytr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if early:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(xva), yva_t).item()
            if vl < best - 1e-4:
                best, best_state, best_ep, wait = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if early and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep


def proba_nn(model, X, device):
    with torch.no_grad():
        return F.softmax(model(torch.as_tensor(X, device=device)), dim=1).cpu().numpy()


def make_sk(enc, cfg, n_feat, seed):
    if enc == "logreg":
        return LogisticRegression(C=cfg["C"], max_iter=3000)
    if enc == "svm":
        return SVC(C=cfg["C"], gamma=cfg["gmul"] / n_feat, probability=True, random_state=seed)
    if enc == "rf":
        return RandomForestClassifier(300, max_features=cfg["max_features"],
                                      min_samples_leaf=cfg["min_samples_leaf"],
                                      random_state=seed, n_jobs=1)
    if enc == "knn":
        return KNeighborsClassifier(n_neighbors=cfg["k"])
    return DummyClassifier(strategy="prior")


def proba_sk(enc, cfg, Xtr, ytr, Xte, C, seed):
    A, B = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    m = make_sk(enc, cfg, A.shape[1], seed).fit(A, ytr)
    p = np.zeros((len(B), C))
    p[:, m.classes_] = m.predict_proba(B)
    return p


# ------------------------------------------------------------------ métricas
def nll(y, p):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], NLL_CLIP, 1.0))))


def metrics(y, p, C):
    p = p / p.sum(1, keepdims=True)
    try:
        auc = (roc_auc_score(y, p[:, 1]) if C == 2
               else roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return dict(acc=accuracy_score(y, p.argmax(1)), f1=f1_score(y, p.argmax(1), average="macro"),
                auc=auc, nll=nll(y, p))


# ------------------------------------------------------------------ CV aninhada
def inner_eval(enc, cfg, Xa, ya, Xb, yb, C, seed, a):
    Xa_s, Xb_s = standardize(Xa, Xb)
    if enc in NN_ENCS:
        model, ep = fit_nn(enc, cfg, Xa_s, ya, C, seed, a.device, a.max_epochs,
                           patience=a.patience, Xva=Xb_s, yva=yb)
        p = proba_nn(model, Xb_s, a.device)
    else:
        p, ep = proba_sk(enc, cfg, Xa_s, ya, Xb_s, C, seed), 0
    p = p / p.sum(1, keepdims=True)
    return nll(yb, p), ep


def final_predict(enc, cfg, epochs, Xtr, ytr, Xte, C, seed, a):
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    if enc in NN_ENCS:
        model, _ = fit_nn(enc, cfg, Xtr_s, ytr, C, seed, a.device, a.max_epochs, epochs=epochs)
        return proba_nn(model, Xte_s, a.device)
    return proba_sk(enc, cfg, Xtr_s, ytr, Xte_s, C, seed)


def run(a):
    out_dir = Path(a.out_dir) / ("perm" if a.permute else "folds")
    out_dir.mkdir(parents=True, exist_ok=True)
    # o nome inclui a resolução (se não nativa) e o nº de repetições da CV externa: mudar --repeats
    # gera arquivos novos em vez de pular/sobrescrever resultados de outra configuração
    name = (f"{a.dataset}__{a.enc}__{a.rep}__r{a.rmax}"
            f"{'' if a.res == 'native' else '__res' + a.res}__R{a.repeats}")
    final_csv, part_csv = out_dir / f"{name}.csv", out_dir / f"{name}.partial.csv"
    if final_csv.exists() and not a.force:
        print(f"[{name}] já concluído, pulando")
        return

    X, y = load(a.dataset, a.data_dir, a.dummy, a.res)
    idx = dev_split(y)
    Xd, yd = prep(X[idx], a.rep, a.rmax), y[idx].copy()
    C = len(np.unique(y))
    if a.permute:  # sanidade: preserva as contagens por classe e destrói a relação X-y
        yd = np.random.default_rng(SEED).permutation(yd)

    outer = list(RepeatedStratifiedKFold(n_splits=a.outer_k, n_repeats=a.repeats,
                                         random_state=SEED).split(Xd, yd))
    rows = pd.read_csv(part_csv).to_dict("records") if (part_csv.exists() and not a.force) else []
    done = {r["fold"] for r in rows}
    t_job = time.time()
    for k, (tr, te) in enumerate(outer):
        if k in done:
            continue
        t0 = time.time()
        Xtr, ytr, Xte, yte = Xd[tr], yd[tr], Xd[te], yd[te]
        cfgs = sample_configs(a.enc, a.trials, len(ytr))
        inner = list(StratifiedKFold(a.inner_k, shuffle=True, random_state=SEED + k).split(Xtr, ytr))
        loss, eps = [], []
        for ti, cfg in enumerate(cfgs):
            r = [inner_eval(a.enc, cfg, Xtr[i], ytr[i], Xtr[j], ytr[j], C,
                            SEED + 1000 * k + 37 * ti + ii, a) for ii, (i, j) in enumerate(inner)]
            loss.append(float(np.mean([x[0] for x in r])))
            eps.append(float(np.mean([x[1] for x in r])))
        sel = int(np.argmin(loss))

        def fin(ti):
            ep = max(3, int(round(eps[ti])))
            p = final_predict(a.enc, cfgs[ti], ep, Xtr, ytr, Xte, C, SEED + 1000 * k + 999, a)
            return metrics(yte, p, C), ep

        m_t, ep_t = fin(sel)
        m_d, ep_d = (m_t, ep_t) if sel == 0 else fin(0)
        T = Xd.shape[1]
        rows.append(dict(
            dataset=a.dataset, enc=a.enc, rep=a.rep, rmax=a.rmax, res=a.res, repeats=a.repeats,
            permute=int(a.permute), fold=k, split_id=hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8],
            n_train=len(tr), n_test=len(te), n_classes=C,
            **m_t, **{f"d_{kk}": v for kk, v in m_d.items()},
            sel_idx=sel, sel_cfg=json.dumps(cfgs[sel]), epochs=ep_t, d_epochs=ep_d,
            inner_nll=loss[sel], d_inner_nll=loss[0],
            n_params=n_params(a.enc, cfgs[sel], T, C), d_n_params=n_params(a.enc, cfgs[0], T, C),
            seconds=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(part_csv, index=False)  # checkpoint por fold
        print(f"[{name}] fold {k + 1}/{len(outer)}  F1={m_t['f1']:.3f} (padrão {m_d['f1']:.3f})  "
              f"cfg#{sel}  {rows[-1]['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    part_csv.unlink(missing_ok=True)
    print(f"[{name}] concluído em {time.time() - t_job:.0f}s  "
          f"F1 médio={np.mean([r['f1'] for r in rows]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--enc", required=True, choices=ALL_ENCS)
    ap.add_argument("--rep", default="cum", choices=["cum", "diff"])
    ap.add_argument("--rmax", type=int, default=15)
    ap.add_argument("--res", default="native", choices=["native", "256"],
                    help="resolução da entrada topológica: native (topological.npz) ou 256 (topological_256.npz)")
    ap.add_argument("--repeats", type=int, default=3, help="repetições da CV externa 5-fold")
    ap.add_argument("--outer_k", type=int, default=5)
    ap.add_argument("--inner_k", type=int, default=3)
    ap.add_argument("--trials", type=int, default=12, help="configurações por codificador (a 0 é a padrão)")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2"))
    ap.add_argument("--device", default="cpu", help="cpu (padrão; modelos minúsculos) ou cuda")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--permute", action="store_true", help="sanidade: rótulos permutados")
    ap.add_argument("--quick", action="store_true", help="teste de fumaça (rápido)")
    ap.add_argument("--dummy", action="store_true", help="dados sintéticos se não houver .npz")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.inner_k, a.repeats, a.max_epochs, a.patience = 3, 2, 1, 8, 3
    torch.set_num_threads(a.threads)
    run(a)


if __name__ == "__main__":
    main()
