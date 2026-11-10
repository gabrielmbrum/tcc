#!/usr/bin/env python
"""
Treinamento e Avaliação da Rede de Dupla Via (Two-Stream Network).
Conforme o protocolo metodológico oficial (Abordagem 1):
  * Partição: 80% desenvolvimento / 20% teste cego (holdout), semente 42 estratificada.
  * Campeões herdados das ablações isoladas:
      - Via Topológica: results_topologica_v3/summary_best.csv
      - Via Visual:     results_visual/summary_best.csv
  * Avaliação confirmatória: 5 sementes aleatórias distintas sobre os 20% de teste.
  * Comparação pareada entre 3 abordagens:
      1. Via Topológica Isolada
      2. Via Visual Isolada
      3. Rede de Dupla Via (Two-Stream Network)
  * Métricas: F1 macro, Acurácia e AUC-ROC (One-vs-Rest macro para multiclasse).
  * Saída: results_dupla_via/ e tabelas LaTeX para tex/resultados-e-discussao.tex.

Uso:
  python src/treino_dupla_via.py --dataset UCSB --device cuda
  python src/treino_dupla_via.py --all --device cuda
  python src/treino_dupla_via.py --summary
  python src/treino_dupla_via.py --dataset UCSB --quick --dummy --device cpu
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATASETS = ["UCSB", "CR", "LA", "LG", "NHL"]
N_RAIOS, N_FEAT = 15, 18
WD = 1e-4
HOLDOUT_SEEDS = 5

MODEL_INFO = {
    "resnet50": {
        "family": "Convolucional",
        "label": "ResNet-50",
        "timm": "resnet50.a1_in1k",
        "kwargs": {},
    },
    "efficientnet_b0": {
        "family": "Convolucional",
        "label": "EfficientNet-B0",
        "timm": "efficientnet_b0.ra_in1k",
        "kwargs": {},
    },
    "vit_tiny": {
        "family": "Transformer",
        "label": "ViT-Tiny",
        "timm": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "kwargs": {"dynamic_img_size": True},
    },
    "deit_tiny": {
        "family": "Transformer",
        "label": "DeiT-Tiny",
        "timm": "deit_tiny_patch16_224.fb_in1k",
        "kwargs": {"dynamic_img_size": True},
    },
    "mobilevit_s": {
        "family": "Híbrida",
        "label": "MobileViT-S",
        "timm": "mobilevit_s.cvnets_in1k",
        "kwargs": {},
    },
    "coatnet_0": {
        "family": "Híbrida",
        "label": "CoAtNet-0",
        "timm": "coatnet_0_rw_224.sw_in1k",
        "kwargs": {"img_size": 256},
    },
}


# ------------------------------------------------------------------ Carregamento de Dados
def load_topo(ds, data_dir, dummy=False):
    p = Path(data_dir) / ds / "topological.npz"
    if p.exists():
        d = np.load(p)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)
    if not dummy:
        alt = Path("/home/brum/ic/data") / ds / "topological.npz"
        if alt.exists():
            d = np.load(alt)
            return d["X"].astype(np.float32), d["y"].astype(np.int64)
        raise FileNotFoundError(f"Arquivo topological.npz não encontrado para {ds}")
    n = {"UCSB": 58, "CR": 165, "LA": 528, "LG": 265, "NHL": 374}[ds]
    c = {"UCSB": 2, "CR": 2, "LA": 4, "LG": 2, "NHL": 3}[ds]
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, c, n).astype(np.int64)
    X = np.cumsum(rng.random((n, N_RAIOS, N_FEAT)) + y[:, None, None] * 0.05, axis=1)
    return X.astype(np.float32), y


def prep_img(path, size=256):
    im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - size) // 2, (nw - size) // 2
    return im[top:top + size, left:left + size]


def list_raw_dataset(ds_dir):
    IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    classes = sorted([d.name for d in ds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    items = []
    for ci, c in enumerate(classes):
        for p in sorted((ds_dir / c).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, ci))
    return classes, items


def load_visual(ds, data_dir, size=256, dummy=False):
    ds_dir = Path(data_dir) / ds
    cache = ds_dir / f"visual_{size}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    if ds_dir.exists():
        topo_path = ds_dir / "topological.npz"
        ref_files, ref_y = None, None
        if topo_path.exists():
            ref = np.load(topo_path)
            ref_files = [str(f) for f in ref["files"]]
            ref_y = ref["y"].astype(np.int64)

        classes, items = list_raw_dataset(ds_dir)
        files = [str(p.relative_to(ds_dir)) for p, _ in items]
        y = np.array([c for _, c in items], dtype=np.int64)

        if ref_files is not None and files != ref_files:
            idx_map = {f: i for i, f in enumerate(files)}
            order = [idx_map[f] for f in ref_files]
            items = [items[i] for i in order]
            files = ref_files
            y = ref_y

        X = np.stack([prep_img(p, size) for p, _ in items])
        np.savez(cache, X=X, y=y, files=np.array(files))
        return X, y

    if not dummy:
        alt_dir = Path("/home/brum/ic/data") / ds
        if alt_dir.exists():
            return load_visual(ds, str(alt_dir.parent), size=size, dummy=dummy)
        raise FileNotFoundError(f"Dados visuais de {ds} não encontrados em {data_dir}")

    n = {"UCSB": 58, "CR": 165, "LA": 528, "LG": 265, "NHL": 374}[ds]
    c = {"UCSB": 2, "CR": 2, "LA": 4, "LG": 2, "NHL": 3}[ds]
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, c, n).astype(np.int64)
    X = rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)
    return X, y


def dev_split(y):
    idx_dev, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return idx_dev


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    return ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def augment(x):
    k = torch.randint(0, 4, (x.shape[0],), device=x.device)
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, (2, 3))
    if flip.any():
        out[flip] = out[flip].flip(3)
    return out


# ------------------------------------------------------------------ Componentes de Rede
class SelfAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.scale = h ** -0.5

    def forward(self, s):
        a = torch.softmax(self.q(s) @ self.k(s).transpose(1, 2) * self.scale, dim=-1)
        return s + a @ self.v(s)


class TopEncoder(nn.Module):
    def __init__(self, kind, h, emb_dim=128):
        super().__init__()
        self.kind = kind
        self.h = h
        self.emb_dim = emb_dim

        if kind in ("lstm", "gru"):
            self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(N_FEAT, h, batch_first=True)
            out_dim = h
        elif kind in ("bilstm", "bigru"):
            self.rnn = (nn.LSTM if kind == "bilstm" else nn.GRU)(
                N_FEAT, h // 2, batch_first=True, bidirectional=True
            )
            out_dim = h
        elif kind == "saclstm":
            self.rnn = nn.LSTM(N_FEAT, h, batch_first=True)
            self.att = SelfAttention(h)
            out_dim = h
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(N_RAIOS * N_FEAT, h), nn.ReLU())
            out_dim = h
        else:
            raise ValueError(f"Codificador topológico desconhecido: {kind}")

        # Projeção linear para 128 dimensões conforme especificação metodológica
        if out_dim != emb_dim:
            self.proj = nn.Sequential(nn.Linear(out_dim, emb_dim), nn.ReLU())
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        if self.kind == "mlp":
            feat = self.net(x)
        elif self.kind == "saclstm":
            s, _ = self.rnn(x)
            feat = self.att(s).mean(1)
        else:
            _, hn = self.rnn(x)
            if self.kind in ("lstm", "bilstm"):
                hn = hn[0]
            feat = torch.cat(list(hn), dim=-1)
        return self.proj(feat)


class VisualBackbone(nn.Module):
    def __init__(self, arch, emb_dim=128, pretrained=True):
        super().__init__()
        info = MODEL_INFO[arch]
        timm_name = info["timm"]
        kwargs = info["kwargs"]

        cfg = timm.get_pretrained_cfg(timm_name)
        mean = tuple(float(v) for v in cfg.mean)
        std = tuple(float(v) for v in cfg.std)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        self.bb = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, **kwargs)
        dim = self.bb.num_features
        self.dim = dim

        # Projeção linear W_p \in R^{128 x D} (Equação 3.6 de materiais-e-metodos.tex)
        self.proj = nn.Linear(dim, emb_dim)

    def forward(self, x, amp=False):
        device_type = "cuda" if x.is_cuda else "cpu"
        use_amp = amp and device_type == "cuda"
        dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp):
            x = (x.float() / 255.0 - self.mean) / self.std
            feat = self.bb(x)
            z_vis = self.proj(feat)
        return z_vis.float()


class IsolatedTopologicalNet(nn.Module):
    def __init__(self, kind, h, C, dropout, emb_dim=128):
        super().__init__()
        self.enc = TopEncoder(kind, h, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, C),
        )

    def forward(self, x_top, amp=False):
        z = self.ln(self.enc(x_top))
        return self.head(z)


class IsolatedVisualNet(nn.Module):
    def __init__(self, arch, C, dropout, emb_dim=128, pretrained=True):
        super().__init__()
        self.bb = VisualBackbone(arch, emb_dim, pretrained=pretrained)
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, C),
        )

    def forward(self, x_vis, amp=False):
        z = self.ln(self.bb(x_vis, amp=amp))
        return self.head(z)


class TwoStreamNet(nn.Module):
    """Arquitetura oficial Two-Stream Network (Materiais e Métodos - Seção 3.2)."""
    def __init__(self, top_kind, top_h, vis_arch, C, dropout, emb_dim=128, pretrained=True):
        super().__init__()
        self.top = TopEncoder(top_kind, top_h, emb_dim)
        self.vis = VisualBackbone(vis_arch, emb_dim, pretrained=pretrained)

        # Normalizações por camadas independentes (LN)
        self.ln_t = nn.LayerNorm(emb_dim)
        self.ln_v = nn.LayerNorm(emb_dim)

        # Classificador MLP sobre vetor concatenado [LN(z_top) || LN(z_vis)] \in R^{256}
        self.cls = nn.Sequential(
            nn.Linear(2 * emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, C),
        )

    def forward(self, x_vis, x_top, amp=False):
        z_v = self.ln_v(self.vis(x_vis, amp=amp))
        z_t = self.ln_t(self.top(x_top))
        z_fus = torch.cat([z_t, z_v], dim=1)
        return self.cls(z_fus)


# ------------------------------------------------------------------ Recuperação dos Campeões
def get_top_champion(dataset, top_dir):
    p = Path(top_dir) / "summary_best.csv"
    if not p.exists():
        raise FileNotFoundError(f"Resumo topológico não encontrado em {p}")
    df = pd.read_csv(p)
    sub = df[df.dataset == dataset]
    if len(sub) == 0:
        raise ValueError(f"Dataset {dataset} ausente em {p}")
    r = sub.loc[sub.f1_mean.idxmax()]
    return {
        "model": r["model"],
        "lr": float(r["lr"]),
        "dropout": float(r["dropout"]),
        "h": int(r["h"]),
        "bs": int(r["bs"]),
        "ep_mean": float(r["ep_mean"]),
        "cv_f1": float(r["f1_mean"]),
    }


def get_vis_champion(dataset, vis_dir):
    p = Path(vis_dir) / "summary_best.csv"
    if not p.exists():
        raise FileNotFoundError(f"Resumo visual não encontrado em {p}")
    df = pd.read_csv(p)
    sub = df[df.dataset == dataset]
    if len(sub) == 0:
        raise ValueError(f"Dataset {dataset} ausente em {p}")
    r = sub.loc[sub.f1_mean.idxmax()]
    return {
        "model": r["model"],
        "lr_bb": float(r["lr_bb"]),
        "lr_hd": float(r["lr_hd"]),
        "dropout": float(r["dropout"]),
        "bs": int(r["bs"]),
        "ep_mean": float(r["ep_mean"]),
        "cv_f1": float(r["f1_mean"]),
    }


# ------------------------------------------------------------------ Métricas
def compute_metrics(y_true, y_pred, y_probs, C):
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    try:
        if C == 2:
            auc = float(roc_auc_score(y_true, y_probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"f1": f1, "acc": acc, "auc": auc}


# ------------------------------------------------------------------ Loops de Treinamento
def train_and_eval_top(top_cfg, Xtr_top, ytr, Xte_top, yte, C, seed, epochs, device):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = IsolatedTopologicalNet(top_cfg["model"], top_cfg["h"], C, top_cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=top_cfg["lr"], weight_decay=WD)

    xtr = torch.as_tensor(Xtr_top, device=dev)
    xte = torch.as_tensor(Xte_top, device=dev)
    ytr_t = torch.as_tensor(ytr, device=dev)

    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), top_cfg["bs"]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            logits = model(xtr[b])
            loss = F.cross_entropy(logits, ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(xte)
        probs = F.softmax(logits_te, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    return compute_metrics(yte, preds, probs, C)


def train_and_eval_vis(vis_cfg, Xtr_vis, ytr, Xte_vis, yte, C, seed, epochs, device, amp, no_pretrained):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = IsolatedVisualNet(
        vis_cfg["model"], C, vis_cfg["dropout"], pretrained=not no_pretrained
    ).to(dev)

    groups = [
        {"params": model.head.parameters(), "lr": vis_cfg["lr_hd"], "weight_decay": WD},
        {"params": model.bb.parameters(), "lr": vis_cfg["lr_bb"], "weight_decay": WD},
    ]
    opt = torch.optim.AdamW(groups)

    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), vis_cfg["bs"]
    ytr_t = torch.as_tensor(ytr, device=dev)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = Xtr_vis[b].to(dev)
            xb = augment(xb)
            yb = ytr_t[b]

            opt.zero_grad()
            logits = model(xb, amp=amp)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(Xte_vis), bs):
            b = Xte_vis[i:i + bs].to(dev)
            logits = model(b, amp=amp)
            probs_list.append(F.softmax(logits, dim=-1).cpu())
    probs = torch.cat(probs_list).numpy()
    preds = probs.argmax(axis=1)

    return compute_metrics(yte, preds, probs, C)


def train_and_eval_dupla(top_cfg, vis_cfg, Xtr_vis, Xtr_top, ytr,
                          Xte_vis, Xte_top, yte, C, seed, epochs,
                          device, amp, no_pretrained):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = TwoStreamNet(
        top_kind=top_cfg["model"],
        top_h=top_cfg["h"],
        vis_arch=vis_cfg["model"],
        C=C,
        dropout=vis_cfg["dropout"],
        pretrained=not no_pretrained,
    ).to(dev)

    # 3 grupos de parâmetros com taxas de aprendizado herdadas
    groups = [
        {"params": model.vis.bb.parameters(), "lr": vis_cfg["lr_bb"], "weight_decay": WD},
        {"params": model.top.parameters(), "lr": top_cfg["lr"], "weight_decay": WD},
        {
            "params": (
                list(model.vis.proj.parameters())
                + list(model.ln_t.parameters())
                + list(model.ln_v.parameters())
                + list(model.cls.parameters())
            ),
            "lr": vis_cfg["lr_hd"],
            "weight_decay": WD,
        },
    ]
    opt = torch.optim.AdamW(groups)

    g = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), vis_cfg["bs"]
    xtr_top_t = torch.as_tensor(Xtr_top, device=dev)
    xte_top_t = torch.as_tensor(Xte_top, device=dev)
    ytr_t = torch.as_tensor(ytr, device=dev)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb_vis = Xtr_vis[b].to(dev)
            xb_vis = augment(xb_vis)
            xb_top = xtr_top_t[b]
            yb = ytr_t[b]

            opt.zero_grad()
            logits = model(xb_vis, xb_top, amp=amp)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(Xte_vis), bs):
            bv = Xte_vis[i:i + bs].to(dev)
            bt = xte_top_t[i:i + bs]
            logits = model(bv, bt, amp=amp)
            probs_list.append(F.softmax(logits, dim=-1).cpu())
    probs = torch.cat(probs_list).numpy()
    preds = probs.argmax(axis=1)

    return compute_metrics(yte, preds, probs, C)


# ------------------------------------------------------------------ Execução de um Dataset
def run_dataset(ds, a):
    print("\n" + "=" * 80)
    print(f"EXPERIMENTO DA REDE DE DUPLA VIA: DATASET {ds}")
    print("=" * 80)

    # 1. Recupera campeões
    top_cfg = get_top_champion(ds, a.top_dir)
    vis_cfg = get_vis_champion(ds, a.vis_dir)

    print(f"Campeão Topológico: {top_cfg['model'].upper()} (h={top_cfg['h']}, lr={top_cfg['lr']}, bs={top_cfg['bs']}, CV-F1={top_cfg['cv_f1']:.3f})")
    print(f"Campeão Visual:     {MODEL_INFO[vis_cfg['model']]['label']} (lr_bb={vis_cfg['lr_bb']}, lr_hd={vis_cfg['lr_hd']}, bs={vis_cfg['bs']}, CV-F1={vis_cfg['cv_f1']:.3f})")

    # 2. Carrega dados
    X_top, y_top = load_topo(ds, a.data_dir, dummy=a.dummy)
    X_vis_raw, y_vis = load_visual(ds, a.data_dir, dummy=a.dummy)

    if not np.array_equal(y_top, y_vis):
        raise ValueError(f"Inconsistência de rótulos entre topological.npz e visual_256.npz em {ds}!")

    y = y_top
    C = len(np.unique(y))
    idx_dev = dev_split(y)
    idx_te = np.setdiff1d(np.arange(len(y)), idx_dev)

    print(f"Divisão amostral: Desenvolvimento={len(idx_dev)} (80%), Teste Cego={len(idx_te)} (20%), Classes={C}")

    # Padronização topológica (estatísticas calculadas apenas no desenvolvimento)
    Xtr_top, Xte_top = standardize(X_top[idx_dev], X_top[idx_te])

    # Tensores visuais (uint8 em RAM)
    X_vis = torch.from_numpy(X_vis_raw).permute(0, 3, 1, 2).contiguous()
    Xtr_vis = X_vis[idx_dev]
    Xte_vis = X_vis[idx_te]

    ytr = y[idx_dev]
    yte = y[idx_te]

    # Número fixo de épocas baseado nas ablações
    epochs_top = max(1, int(round(top_cfg["ep_mean"])))
    epochs_vis = max(1, int(round(vis_cfg["ep_mean"])))
    epochs_dupla = max(epochs_vis, 1)  # via dupla segue época visual como base principal

    if a.quick:
        epochs_top, epochs_vis, epochs_dupla = 1, 1, 1

    out_seeds = Path(a.out_dir) / "seeds"
    out_seeds.mkdir(parents=True, exist_ok=True)
    seed_csv = out_seeds / f"{ds}__holdout_seeds.csv"

    rows = []
    seeds = range(1 if a.quick else HOLDOUT_SEEDS)

    for s in seeds:
        seed = SEED + s
        t0 = time.time()
        print(f"\n---> Rodada {s + 1}/{len(seeds)} (Semente {seed}):")

        # 1. Topológica Isolada
        m_top = train_and_eval_top(top_cfg, Xtr_top, ytr, Xte_top, yte, C, seed, epochs_top, a.device)
        print(f"  [Topológica] F1={m_top['f1']:.4f}  Acc={m_top['acc']:.4f}  AUC={m_top['auc']:.4f}")

        # 2. Visual Isolada
        m_vis = train_and_eval_vis(
            vis_cfg, Xtr_vis, ytr, Xte_vis, yte, C, seed, epochs_vis, a.device, a.amp, a.no_pretrained
        )
        print(f"  [Visual]     F1={m_vis['f1']:.4f}  Acc={m_vis['acc']:.4f}  AUC={m_vis['auc']:.4f}")

        # 3. Rede de Dupla Via
        m_dup = train_and_eval_dupla(
            top_cfg, vis_cfg, Xtr_vis, Xtr_top, ytr, Xte_vis, Xte_top, yte, C, seed, epochs_dupla,
            a.device, a.amp, a.no_pretrained
        )
        print(f"  [Dupla Via]  F1={m_dup['f1']:.4f}  Acc={m_dup['acc']:.4f}  AUC={m_dup['auc']:.4f} ({time.time() - t0:.1f}s)")

        for app, m in [("top", m_top), ("vis", m_vis), ("dupla", m_dup)]:
            rows.append({
                "dataset": ds,
                "seed": seed,
                "approach": app,
                "top_model": top_cfg["model"],
                "vis_model": vis_cfg["model"],
                "f1": m["f1"],
                "acc": m["acc"],
                "auc": m["auc"],
            })

    df_seeds = pd.DataFrame(rows)
    df_seeds.to_csv(seed_csv, index=False)
    return df_seeds


# ------------------------------------------------------------------ Resumo e Tabelas
def generate_summary(a):
    out = Path(a.out_dir)
    seed_files = sorted((out / "seeds").glob("*__holdout_seeds.csv"))
    if not seed_files:
        raise SystemExit(f"Nenhum arquivo de resultados encontrado em {out / 'seeds'}")

    df = pd.concat([pd.read_csv(f) for f in seed_files])
    summary = df.groupby(["dataset", "approach"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()

    summary.to_csv(out / "summary_comparison.csv", index=False)
    print("\n" + "=" * 90)
    print("RESUMO COMPARATIVO NO CONJUNTO DE TESTE FINAL (20% HOLDOUT, 5 SEMENTES)")
    print("=" * 90)
    print(summary.round(4).to_string(index=False))

    # Geração do snippet LaTeX para tex/resultados-e-discussao.tex
    tex_path = out / "tabela_final.tex"
    lines = [
        "% Tabela tab:res-comparacao-final gerada automaticamente por treino_dupla_via.py\n",
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Comparação de desempenho (média $\\pm$ desvio padrão ao longo de 5 sementes) no conjunto de teste independente (20\\% das amostras) entre as vias isoladas e a arquitetura unificada de dupla via.}",
        "\\label{tab:res-comparacao-final}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{\\textit{Dataset}} & \\textbf{Métrica} & \\textbf{Via Topológica Isolada} & \\textbf{Via Visual Isolada} & \\textbf{Rede de Dupla Via} \\\\",
        "\\midrule",
    ]

    for ds in DATASETS:
        sub = summary[summary.dataset == ds]
        if len(sub) == 0:
            continue
        row_top = sub[sub.approach == "top"]
        row_vis = sub[sub.approach == "vis"]
        row_dup = sub[sub.approach == "dupla"]

        if len(row_top) and len(row_vis) and len(row_dup):
            t, v, d = row_top.iloc[0], row_vis.iloc[0], row_dup.iloc[0]

            def fmt(mean, std):
                return f"{mean:.3f} $\\pm$ {std:.3f}" if not np.isnan(mean) else "---"

            lines.append(f"{ds} & F1 macro & {fmt(t['f1_mean'], t['f1_std'])} & {fmt(v['f1_mean'], v['f1_std'])} & \\textbf{{{fmt(d['f1_mean'], d['f1_std'])}}} \\\\")
            lines.append(f"{ds} & Acurácia & {fmt(t['acc_mean'], t['acc_std'])} & {fmt(v['acc_mean'], v['acc_std'])} & \\textbf{{{fmt(d['acc_mean'], d['acc_std'])}}} \\\\")
            lines.append(f"{ds} & AUC-ROC  & {fmt(t['auc_mean'], t['auc_std'])} & {fmt(v['auc_mean'], v['auc_std'])} & \\textbf{{{fmt(d['auc_mean'], d['auc_std'])}}} \\\\")
            lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}%\n}\n\\end{table}\n")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTabela LaTeX salva em: {tex_path}")


# ------------------------------------------------------------------ Main
def main():
    ap = argparse.ArgumentParser(description="Treino e Avaliação da Rede de Dupla Via")
    ap.add_argument("--dataset", choices=DATASETS, help="Executa em um dataset específico")
    ap.add_argument("--all", action="store_true", help="Executa em todos os datasets sequencialmente")
    ap.add_argument("--summary", action="store_true", help="Consolida os resultados e gera tabelas LaTeX")
    ap.add_argument("--data_dir", default=str(ROOT / "data"), help="Diretório de dados")
    ap.add_argument("--top_dir", default=str(ROOT / "results_topologica_v3"), help="Diretório da ablação topológica")
    ap.add_argument("--vis_dir", default=str(ROOT / "results_visual"), help="Diretório da ablação visual")
    ap.add_argument("--out_dir", default=str(ROOT / "results_dupla_via"), help="Diretório de saída")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Ativa precisão mista bfloat16/float16")
    ap.add_argument("--quick", action="store_true", help="Modo rápido de teste")
    ap.add_argument("--dummy", action="store_true", help="Gera dados sintéticos aleatórios")
    ap.add_argument("--no_pretrained", action="store_true", help="Sem baixar pesos do ImageNet")

    args = ap.parse_args()

    if args.summary:
        generate_summary(args)
    elif args.all:
        for ds in DATASETS:
            run_dataset(ds, args)
        generate_summary(args)
    elif args.dataset:
        run_dataset(args.dataset, args)
        generate_summary(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
