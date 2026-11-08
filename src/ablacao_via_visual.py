#!/usr/bin/env python
"""
Ablação da via visual isolada conforme o protocolo de tex/ablacao.tex e materiais-e-metodos.tex.

Protocolo:
  1. Holdout estratificado 80/20 (SEED=42). A partição de 20% é totalmente reservada.
  2. Validação cruzada estratificada em 5 dobras (k=5, SEED=42) sobre os 80% de desenvolvimento.
     As mesmas 5 dobras são compartilhadas com a via topológica (mesmas divisões por dataset).
  3. Imagens de entrada em 256x256 pixels com preservação de proporção e recorte central.
  4. 6 arquiteturas distribuídas em 3 famílias:
       - Convolucional: ResNet-50, EfficientNet-B0
       - Vision Transformer: ViT-Tiny, DeiT-Tiny
       - Híbrida: MobileViT-S, CoAtNet-0
     Todas inicializadas com pesos ImageNet-1k (timm) e ajustadas de ponta a ponta.
  5. Cabeça de classificação:
       Linear(D, 128) -> LayerNorm(128) -> Linear(128, 128) -> ReLU -> Dropout(p) -> Linear(128, C)
  6. Aumento de dados rotacional no treino: rotações ortogonais (0, 90, 180, 270) e inversões horizontais.
  7. Espaço de busca (Tabela tab:espaco-busca-vis):
       - lr_bb (backbone): {1e-5, 3e-5, 1e-4, 3e-4}
       - lr_hd (cabeça): {1e-4, 3e-4, 1e-3}
       - wd (weight decay): 1e-4 (fixo)
       - dropout: {0.1, 0.3, 0.5}
       - lote: {4, 8, 16} (UCSB) ou {16, 32, 64} (CR, LG, NHL, LA)
       Total: 108 combinações por modelo x 5 dobras = 540 treinos por (dataset, modelo).
  8. Treino: AdamW, grad_clip <= 1.0, máx 50 épocas, parada antecipada com paciência de 10 épocas
     sobre a perda de validação com restauração dos melhores pesos.
  9. Métrica: F1 macro na validação. O resumo seleciona a melhor combinação por (dataset, modelo)
     e gera as tabelas tab:abl-vis-f1 e tab:abl-vis-params prontas para o LaTeX.
 10. Holdout final (--holdout): avalia a melhor configuração no teste cego de 20% com 5 sementes.

Uso:
  python src/ablacao_via_visual.py --dataset UCSB --model resnet50 --device cuda
  python src/ablacao_via_visual.py --dataset UCSB --model resnet50 --quick --dummy   # fumaça
  python src/ablacao_via_visual.py --summary                                        # gera tabelas LaTeX
  python src/ablacao_via_visual.py --holdout                                        # teste final 20%
"""
import argparse
import copy
import itertools
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

DATASETS = ["UCSB", "CR", "LA", "LG", "NHL"]
MODELS = ["resnet50", "efficientnet_b0", "vit_tiny", "deit_tiny", "mobilevit_s", "coatnet_0"]

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
        "family": "\\textit{Transformer}",
        "label": "ViT-Tiny",
        "timm": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "kwargs": {"dynamic_img_size": True},
    },
    "deit_tiny": {
        "family": "\\textit{Transformer}",
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

LR_BBS = [1e-5, 3e-5, 1e-4, 3e-4]
LR_HDS = [1e-4, 3e-4, 1e-3]
DROPS = [0.1, 0.3, 0.5]
WD = 1e-4
HP_KEYS = ["lr_bb", "lr_hd", "dropout", "bs"]
HOLDOUT_SEEDS = 5


# ------------------------------------------------------------------ dados
def prep_img(path, size=256):
    """Redimensiona o menor lado para `size` (INTER_AREA) e extrai o recorte central."""
    im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - size) // 2, (nw - size) // 2
    return im[top:top + size, left:left + size]


def list_raw_dataset(ds_dir):
    """Lista imagens ordenadas por classe e nome de arquivo."""
    IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    classes = sorted([d.name for d in ds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    items = []
    for ci, c in enumerate(classes):
        for p in sorted((ds_dir / c).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, ci))
    return classes, items


def load_images(ds, data_dir, size=256, dummy=False):
    """Carrega as imagens 256x256 (uint8) e os rótulos y. Cria visual_256.npz se necessário."""
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
            # Reordena para casar rigorosamente com topological.npz
            idx_map = {f: i for i, f in enumerate(files)}
            order = [idx_map[f] for f in ref_files]
            items = [items[i] for i in order]
            files = ref_files
            y = ref_y

        print(f"[{ds}] Processando {len(items)} imagens para {size}x{size}...")
        X = np.stack([prep_img(p, size) for p, _ in items])
        np.savez(cache, X=X, y=y, files=np.array(files))
        return X, y

    if not dummy:
        # Tenta em /home/brum/ic/data se existir
        alt_dir = Path("/home/brum/ic/data") / ds
        if alt_dir.exists():
            return load_images(ds, str(alt_dir.parent), size=size, dummy=dummy)
        raise FileNotFoundError(f"Dados do dataset {ds} não encontrados em {data_dir} nem em /home/brum/ic/data")

    # Modo sintético / teste de fumaça
    n = {"UCSB": 58, "CR": 165, "LA": 528, "LG": 265, "NHL": 374}[ds]
    c = {"UCSB": 2, "CR": 2, "LA": 4, "LG": 2, "NHL": 3}[ds]
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, c, n).astype(np.int64)
    X = rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)
    return X, y


def dev_split(y):
    """Holdout estratificado 80/20 com SEED=42 (idêntico à via topológica)."""
    idx_dev, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return idx_dev


def folds_of(y):
    """Validação cruzada estratificada em 5 dobras sobre os 80% de desenvolvimento."""
    idx = dev_split(y)
    Xy = np.zeros(len(idx))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xy, y[idx]))
    return idx, folds


# ------------------------------------------------------------------ aumento de dados
def augment(x):
    """Aumento de dados rotacional: inversões horizontais e rotações ortogonais de 90°."""
    # x: tensor (B, 3, 256, 256)
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


# ------------------------------------------------------------------ modelo
class VisualAblationNet(nn.Module):
    def __init__(self, arch, C, dropout, pretrained=True):
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

        # Cabeça de classificação adaptada conforme tex/ablacao.tex e materiais-e-metodos.tex:
        # Linear(D, 128) -> LayerNorm(128) -> Linear(128, 128) -> ReLU -> Dropout(p) -> Linear(128, C)
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, C),
        )

    def extract_features(self, x):
        # Normalização nos tensores de entrada
        x = (x.float() / 255.0 - self.mean) / self.std
        return self.bb(x)

    def forward(self, x, amp=False):
        device_type = "cuda" if x.is_cuda else "cpu"
        use_amp = amp and device_type == "cuda"
        dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp):
            feat = self.extract_features(x)
            out = self.head(feat)
        return out.float()


# ------------------------------------------------------------------ treino e avaliação
def grid(ds):
    bss = [4, 8, 16] if ds == "UCSB" else [16, 32, 64]
    return [dict(zip(HP_KEYS, v)) for v in itertools.product(LR_BBS, LR_HDS, DROPS, bss)]


def key(hp):
    return json.dumps({k: hp[k] for k in HP_KEYS}, sort_keys=True)


@torch.no_grad()
def predict(model, X, idx, dev, bs=64, amp=False):
    model.eval()
    preds = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xb = X[b].to(dev)
        logits = model(xb, amp=amp)
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()


def fit_eval(arch, hp, X, y, tr_idx, va_idx, C, seed, a):
    dev = torch.device(a.device)
    torch.manual_seed(seed)
    model = VisualAblationNet(arch, C, hp["dropout"], pretrained=not a.no_pretrained).to(dev)

    groups = [
        {"params": model.head.parameters(), "lr": hp["lr_hd"], "weight_decay": WD},
        {"params": model.bb.parameters(), "lr": hp["lr_bb"], "weight_decay": WD},
    ]
    opt = torch.optim.AdamW(groups)

    g = torch.Generator().manual_seed(seed)
    n, bs = len(tr_idx), hp["bs"]
    best_loss, best_state, best_ep, wait = float("inf"), None, 1, 0

    tr_tensor = torch.as_tensor(tr_idx, dtype=torch.long)
    va_tensor = torch.as_tensor(va_idx, dtype=torch.long)
    y_dev = torch.as_tensor(y, device=dev)

    max_epochs = a.max_epochs
    patience = a.patience

    for ep in range(max_epochs):
        model.train()
        perm = tr_tensor[torch.randperm(n, generator=g)]
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = X[b].to(dev)
            xb = augment(xb)
            yb = y_dev[b]

            opt.zero_grad()
            logits = model(xb, amp=a.amp)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validação
        model.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(va_tensor), bs):
                b = va_tensor[i:i + bs]
                xb = X[b].to(dev)
                yb = y_dev[b]
                logits = model(xb, amp=a.amp)
                val_losses.append(F.cross_entropy(logits, yb).item() * len(b))
            vl = sum(val_losses) / len(va_tensor)

        if vl < best_loss - 1e-4:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = predict(model, X, va_idx, dev, bs=bs, amp=a.amp)
    y_true = y[va_idx]
    f1 = float(f1_score(y_true, preds, average="macro"))
    acc = float(accuracy_score(y_true, preds))
    return dict(f1=f1, acc=acc, val_loss=best_loss, best_ep=best_ep)


# ------------------------------------------------------------------ execução principal (CV)
def run(a):
    out = Path(a.out_dir) / "folds"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{a.dataset}__{a.model}.csv"

    X_np, y = load_images(a.dataset, a.data_dir, dummy=a.dummy)
    idx_dev, folds = folds_of(y)
    C = len(np.unique(y))

    # X em tensor uint8 (N, 3, 256, 256) mantido em RAM para evitar estouro de VRAM
    X = torch.from_numpy(X_np).permute(0, 3, 1, 2).contiguous()

    cfgs = grid(a.dataset)
    if a.quick:
        cfgs = cfgs[::len(cfgs) // 2][:2]

    rows = pd.read_csv(csv).to_dict("records") if (csv.exists() and not a.force) else []
    done = {key(r) for r in rows if int(r["fold"]) == 4}

    t0 = time.time()
    for ci, hp in enumerate(cfgs):
        if key(hp) in done:
            continue
        res = []
        for k, (tr_local, va_local) in enumerate(folds):
            # Mapeia índices locais de desenvolvimento para os índices globais
            tr_global = idx_dev[tr_local]
            va_global = idx_dev[va_local]
            t_fold = time.time()
            r = fit_eval(a.model, hp, X, y, tr_global, va_global, C, SEED + k, a)
            sec = round(time.time() - t_fold, 1)
            res.append(dict(
                dataset=a.dataset, model=a.model, fold=k, **hp, wd=WD,
                n_train=len(tr_global), n_val=len(va_global), seconds=sec, **r
            ))
        rows += res
        pd.DataFrame(rows).to_csv(csv, index=False)
        f1_m = np.mean([r["f1"] for r in res])
        print(f"[{a.dataset}/{a.model}] {ci + 1}/{len(cfgs)} {hp} F1={f1_m:.3f} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ resumo e tabelas LaTeX
def summary(a):
    files = sorted((Path(a.out_dir) / "folds").glob("*.csv"))
    if not files:
        raise SystemExit(f"Sem arquivos de resultados em {Path(a.out_dir) / 'folds'}")
    df = pd.concat([pd.read_csv(f) for f in files])
    per = df.groupby(["dataset", "model"] + HP_KEYS).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        acc_mean=("acc", "mean"),
        ep_mean=("best_ep", "mean"),
        n_folds=("fold", "nunique"),
    ).reset_index()

    per = per[per.n_folds == 5]
    per.to_csv(Path(a.out_dir) / "summary_all_configs.csv", index=False)

    best = per.loc[per.groupby(["dataset", "model"])["f1_mean"].idxmax()].reset_index(drop=True)
    best.to_csv(Path(a.out_dir) / "summary_best.csv", index=False)

    print("\n" + "=" * 80)
    print("MELHORES CONFIGURAÇÕES POR DATASET E MODELO (F1 Macro Médio nas 5 Dobras de Validação)")
    print("=" * 80)
    print(best[["dataset", "model", "f1_mean", "f1_std"] + HP_KEYS + ["ep_mean"]].round(4).to_string(index=False))

    # Geração dos snippets LaTeX para inserção direta no TCC
    gen_latex_tables(best, Path(a.out_dir) / "tabelas.tex")


def format_lr(val):
    if val >= 1e-3:
        return f"{val:g}"
    s = f"{val:.0e}"
    coef, exp = s.split("e")
    exp = int(exp)
    if coef == "1":
        return f"10^{{{exp}}}"
    return f"{coef}\\cdot 10^{{{exp}}}"


def gen_latex_tables(best_df, out_tex):
    lines = ["% Tabelas geradas automaticamente por ablacao_via_visual.py\n"]

    # 1. Tabela tab:abl-vis-f1
    lines.append("% Tabela tab:abl-vis-f1")
    lines.append("\\begin{table}[htb]\n\\centering")
    lines.append("\\caption{F1 macro (média $\\pm$ desvio padrão entre as 5 partições de validação) da via visual isolada, na melhor combinação de hiperparâmetros de cada modelo. Em negrito, o maior valor médio de cada conjunto de dados.}")
    lines.append("\\label{tab:abl-vis-f1}\n\\small\n\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccccc}\n\\toprule")
    lines.append("\\textbf{Família} & \\textbf{Modelo} & \\textbf{UCSB} & \\textbf{CR} & \\textbf{LA} & \\textbf{LG} & \\textbf{NHL} \\\\\n\\midrule")

    # Encontra o valor máximo por dataset
    max_by_ds = best_df.groupby("dataset")["f1_mean"].max().to_dict()

    for m in MODELS:
        info = MODEL_INFO[m]
        fam, lab = info["family"], info["label"]
        row_str = [f"{fam} & {lab}"]
        for ds in DATASETS:
            sub = best_df[(best_df.dataset == ds) & (best_df.model == m)]
            if len(sub) == 0:
                row_str.append("\\pend")
            else:
                r = sub.iloc[0]
                m_val, s_val = r["f1_mean"], r["f1_std"]
                is_max = np.isclose(m_val, max_by_ds.get(ds, -1), atol=1e-5)
                val_str = f"{m_val:.3f} $\\pm$ {s_val:.3f}"
                if is_max:
                    val_str = f"\\textbf{{{m_val:.3f}}} $\\pm$ {s_val:.3f}"
                row_str.append(val_str)
        lines.append(" & ".join(row_str) + " \\\\")

    lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n\n")

    # 2. Tabela tab:abl-vis-params
    lines.append("% Tabela tab:abl-vis-params")
    lines.append("\\begin{table}[htb]\n\\centering")
    lines.append("\\caption{Melhores combinações de hiperparâmetros obtidas na validação cruzada para cada modelo visual nos cinco conjuntos de dados.}")
    lines.append("\\label{tab:abl-vis-params}\n\\small\n\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lllccccc}\n\\toprule")
    lines.append("\\textbf{\\textit{Dataset}} & \\textbf{Família} & \\textbf{Modelo} & $\\eta_{bb}$ & $\\eta_{hd}$ & \\textbf{\\textit{Dropout}} & \\textbf{Lote} & \\textbf{F1 macro} \\\\\n\\midrule")

    for ds in DATASETS:
        for m in MODELS:
            info = MODEL_INFO[m]
            fam, lab = info["family"], info["label"]
            sub = best_df[(best_df.dataset == ds) & (best_df.model == m)]
            if len(sub) == 0:
                lines.append(f"{ds} & {fam} & {lab} & \\pend & \\pend & \\pend & \\pend & \\pend \\\\")
            else:
                r = sub.iloc[0]
                lines.append(f"{ds} & {fam} & {lab} & ${format_lr(r['lr_bb'])}$ & ${format_lr(r['lr_hd'])}$ & {r['dropout']} & {int(r['bs'])} & {r['f1_mean']:.3f} $\\pm$ {r['f1_std']:.3f} \\\\")
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}%\n}\n\\end{table}\n")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTabelas LaTeX salvas em: {out_tex}")


# ------------------------------------------------------------------ holdout (teste final 20%)
def fit_fixed(arch, hp, X, y, tr_global, te_global, C, seed, epochs, a):
    """Treina no desenvolvimento completo (80%) por número fixo de épocas e infere no holdout 20%."""
    dev = torch.device(a.device)
    torch.manual_seed(seed)
    model = VisualAblationNet(arch, C, hp["dropout"], pretrained=not a.no_pretrained).to(dev)

    groups = [
        {"params": model.head.parameters(), "lr": hp["lr_hd"], "weight_decay": WD},
        {"params": model.bb.parameters(), "lr": hp["lr_bb"], "weight_decay": WD},
    ]
    opt = torch.optim.AdamW(groups)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(tr_global), int(hp["bs"])

    tr_tensor = torch.as_tensor(tr_global, dtype=torch.long)
    y_dev = torch.as_tensor(y, device=dev)

    for _ in range(epochs):
        model.train()
        perm = tr_tensor[torch.randperm(n, generator=g)]
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = X[b].to(dev)
            xb = augment(xb)
            yb = y_dev[b]

            opt.zero_grad()
            logits = model(xb, amp=a.amp)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    preds = predict(model, X, te_global, dev, bs=bs, amp=a.amp)
    y_true = y[te_global]
    return f1_score(y_true, preds, average="macro"), accuracy_score(y_true, preds)


def holdout(a):
    best_csv = Path(a.out_dir) / "summary_best.csv"
    if not best_csv.exists():
        raise SystemExit(f"Arquivo {best_csv} não encontrado. Execute primeiro com --summary.")
    best = pd.read_csv(best_csv)
    rows = []

    for _, r in best.iterrows():
        X_np, y = load_images(r.dataset, a.data_dir, dummy=a.dummy)
        idx_dev = dev_split(y)
        idx_te = np.setdiff1d(np.arange(len(y)), idx_dev)
        epochs = max(1, int(round(r.ep_mean)))
        C = len(np.unique(y))
        X = torch.from_numpy(X_np).permute(0, 3, 1, 2).contiguous()

        res = [
            fit_fixed(r.model, r, X, y, idx_dev, idx_te, C, SEED + s, epochs, a)
            for s in range(HOLDOUT_SEEDS)
        ]
        f1s = [x[0] for x in res]
        accs = [x[1] for x in res]

        rows.append(dict(
            dataset=r.dataset, model=r.model, cv_f1=r.f1_mean,
            holdout_f1=float(np.mean(f1s)), holdout_std=float(np.std(f1s)),
            holdout_acc=float(np.mean(accs)), n_test=len(idx_te), epochs=epochs
        ))
        print(f"[{r.dataset}/{r.model}] CV={r.f1_mean:.3f} | Holdout 20%={np.mean(f1s):.3f} ± {np.std(f1s):.3f} (n_test={len(idx_te)})", flush=True)

    h = pd.DataFrame(rows)
    h.to_csv(Path(a.out_dir) / "holdout_results.csv", index=False)
    print("\nGap médio (CV - Holdout):", round((h.cv_f1 - h.holdout_f1).mean(), 4))
    print(h.pivot(index="model", columns="dataset", values="holdout_f1").round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description="Ablação da Via Visual Isolada (Protocolo TCC)")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--summary", action="store_true", help="Gera resumo das melhores configurações e tabelas LaTeX")
    ap.add_argument("--holdout", action="store_true", help="Avalia as melhores configurações no holdout 20%%")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_visual"))
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Usa precisão mista (bfloat16/float16)")
    ap.add_argument("--no_pretrained", action="store_true", help="Não baixa pesos (apenas para teste de fumaça)")
    ap.add_argument("--quick", action="store_true", help="Executa apenas subconjunto mínimo para teste rápido")
    ap.add_argument("--dummy", action="store_true", help="Usa dados sintéticos aleatórios")
    ap.add_argument("--force", action="store_true", help="Sobrescreve resultados existentes")
    a = ap.parse_args()

    if a.summary:
        summary(a)
    elif a.holdout:
        holdout(a)
    else:
        if not (a.dataset and a.model):
            ap.error("Os argumentos --dataset e --model são obrigatórios para a execução do treinamento.")
        run(a)


if __name__ == "__main__":
    main()
