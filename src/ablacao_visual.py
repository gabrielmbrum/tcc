#!/usr/bin/env python
"""
Ablação da via visual isolada (mesmo protocolo da ablacao_robusta.py).

Reaproveita de ablacao_robusta.py a divisão de desenvolvimento (SEED=42, holdout 20% NUNCA usado), as
partições externas (5-fold x repeats), as internas (StratifiedKFold SEED+k), as sementes e as métricas.
Os folds são portanto IDÊNTICOS aos da via topológica (mesmo split_id), o que permite comparar a via
visual com a topológica fold a fold em analise_visual.py.

Entrada visual (como no M&M): lado menor -> 256 (INTER_AREA, proporção preservada), recorte central
256x256, normalização ImageNet. As imagens (uint8) são pré-processadas uma vez e guardadas em
data/<DS>/visual_256.npz (a ordem de arquivos é conferida com topological.npz).

Arquiteturas (timm, pesos ImageNet): mobilevit_s (híbrida, a proposta), efficientnet_b0 (CNN) e
deit_tiny (ViT), todas com ~5-6 M de parâmetros.
Modos:
  ft      ajuste fino de todo o modelo (AdamW, lr do backbone e da cabeça separados), com aumento de dados
          (--aug flip_rot: inversões e rotações de 90 graus, só no treino).
  frozen  backbone congelado: embeddings extraídos uma vez (sem aumento) e só a cabeça é treinada.
Cabeça (igual ao M&M): Linear(D,128) -> LayerNorm -> Linear(128,128) -> ReLU -> Dropout -> Linear(128,C).

Busca: --trials configurações por tarefa (a 0 é a padrão: lr backbone 1e-5, lr cabeça 1e-4, wd 1e-4,
dropout 0.5, lote 8/16), nos mesmos moldes da CV aninhada da via topológica (seleção pela entropia cruzada
interna, parada antecipada só na CV interna, ajuste final com nº fixo de épocas = média das épocas internas).

Saída: results_v2/visual/folds/<DS>__<arch>__<mode>[__<aug>]__T<trials>__R<repeats>.csv (uma linha por fold
externo). O rótulo `enc` é <arch>/frozen, <arch>/ft_aug (inversões e rotações) ou <arch>/ft_noaug (sem aumento).
O aumento de dados é um fator da ablação: o ajuste fino é rodado com --aug flip_rot e com --aug none.

Exemplos:
  uv run src/ablacao_visual.py --dataset UCSB --arch mobilevit_s --mode ft --device cuda
  uv run src/ablacao_visual.py --dataset UCSB --arch mobilevit_s --mode frozen --quick --no_pretrained
"""
import argparse
import copy
import hashlib
import json
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacao_robusta import SEED, dev_split, metrics, nll  # noqa: E402
from build_topological import list_dataset  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ARCHS = {"mobilevit_s": "mobilevit_s.cvnets_in1k",
         "efficientnet_b0": "efficientnet_b0.ra_in1k",
         "deit_tiny": "deit_tiny_patch16_224.fb_in1k"}
AMP = False  # precisão mista bfloat16 (--amp); definido em main()
IMAGENET_MEAN = (0.485, 0.456, 0.406)


# ------------------------------------------------------------------ dados
def prep_img(path, size):
    im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    t, l = (nh - size) // 2, (nw - size) // 2
    return im[t:t + size, l:l + size]


def load_images(ds, data_dir, size=256):
    ds_dir = Path(data_dir) / ds
    ref = np.load(ds_dir / "topological.npz")
    ref_files = [str(f) for f in ref["files"]]
    cache = ds_dir / f"visual_{size}.npz"
    if cache.exists():
        d = np.load(cache)
        if [str(f) for f in d["files"]] == ref_files:
            return d["X"], d["y"]
    _, items = list_dataset(ds_dir)
    files = [str(p.relative_to(ds_dir)) for p, _ in items]
    if files != ref_files or not np.array_equal([c for _, c in items], ref["y"]):
        raise ValueError(f"{ds}: ordem de imagens diferente de topological.npz")
    X = np.stack([prep_img(p, size) for p, _ in items])
    y = ref["y"].astype(np.int64)
    np.savez(cache, X=X, y=y, files=np.array(files))
    return X, y


# ------------------------------------------------------------------ modelo
class VNet(nn.Module):
    def __init__(self, backbone, dim, C, dropout, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        super().__init__()
        # normalização do pretrained_cfg de cada backbone (a MobileViT-S usa média 0 e desvio 1)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.bb = backbone  # None no modo frozen (a entrada já são embeddings)
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.LayerNorm(128), nn.Linear(128, 128),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, C))

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            if self.bb is not None:
                x = (x.float() / 255 - self.mean) / self.std
                x = self.bb(x)
            x = self.head(x)
        return x.float()


def augment(x):
    """Inversões e rotações de 90 graus (grupo diedral), por imagem; só no treino."""
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


def norm_of(arch):
    """Média e desvio do pretrained_cfg do backbone (sem baixar pesos)."""
    cfg = make_backbone(arch, False).pretrained_cfg
    return tuple(float(v) for v in cfg["mean"]), tuple(float(v) for v in cfg["std"])


def make_backbone(arch, pretrained):
    kw = {"dynamic_img_size": True} if arch == "deit_tiny" else {}
    return timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=0, **kw)


@torch.no_grad()
def extract(bb, X, dev, mean, std, bs=64):
    bb.eval().to(dev)
    out = []
    for i in range(0, len(X), bs):
        x = (X[i:i + bs].float() / 255 - torch.tensor(mean, device=dev).view(1, 3, 1, 1)) \
            / torch.tensor(std, device=dev).view(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            out.append(bb(x).float())
    return torch.cat(out)


@torch.no_grad()
def predict(model, X, idx, dev, bs=64):
    model.eval()
    return torch.cat([F.softmax(model(X[idx[i:i + bs]]), 1) for i in range(0, len(idx), bs)]).cpu().numpy()


def fit(ctx, cfg, tr, seed, max_epochs, patience=None, va=None, epochs=None):
    """Com `va`: parada antecipada pela perda de validação (restaura os melhores pesos).
    Sem `va`: treina exatamente `epochs` épocas."""
    dev, X, y = ctx["dev"], ctx["X"], ctx["y"]
    torch.manual_seed(seed)
    bb = copy.deepcopy(ctx["bb"]).to(dev) if ctx["mode"] == "ft" else None
    model = VNet(bb, ctx["dim"], ctx["C"], cfg["dropout"], ctx["mean"], ctx["std"]).to(dev)
    groups = [{"params": model.head.parameters(), "lr": cfg["lr_hd"]}]
    if bb is not None:
        groups.append({"params": bb.parameters(), "lr": cfg["lr_bb"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"])
    g = torch.Generator().manual_seed(seed)
    tr_t = torch.as_tensor(tr, device=dev)
    n, bs = len(tr), cfg["bs"]
    best, best_state, best_ep, wait = float("inf"), None, 1, 0
    for ep in range(max_epochs if va is not None else epochs):
        model.train()
        perm = tr_t[torch.randperm(n, generator=g).to(dev)]
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:  # BatchNorm exige >1 amostra
                continue
            xb = augment(X[b]) if (ctx["mode"] == "ft" and ctx["aug"]) else X[b]
            opt.zero_grad()
            F.cross_entropy(model(xb), y[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if va is not None:
            p = predict(model, X, torch.as_tensor(va, device=dev), dev)
            vl = float(F.nll_loss(torch.log(torch.as_tensor(p).clamp_min(1e-12)), y[va].cpu()))
            if vl < best - 1e-4:
                best, best_state, best_ep, wait = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if va is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


def sample_cfgs(n_trials, n_train, seed=0):
    # espaço ampliado: na 1ª versão, lr_bb=1e-4 e wd=1e-2 (valores máximos) foram os mais selecionados
    bsc = [8, 16, 32] if n_train < 100 else [16, 32]
    cfgs = [dict(lr_bb=1e-5, lr_hd=1e-4, wd=1e-4, dropout=0.5, bs=8 if n_train < 100 else 16)]
    rng = np.random.default_rng(seed)
    for _ in range(n_trials - 1):
        cfgs.append(dict(lr_bb=float(rng.choice([1e-5, 3e-5, 1e-4, 3e-4])),
                         lr_hd=float(rng.choice([1e-4, 3e-4, 1e-3, 3e-3])),
                         wd=float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1])),
                         dropout=float(rng.choice([0.1, 0.3, 0.5])),
                         bs=int(rng.choice(bsc))))
    return cfgs


# ------------------------------------------------------------------ CV aninhada
def run(a):
    out_dir = Path(a.out_dir) / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = a.aug if a.mode == "ft" else "none"
    enc = (f"{a.arch}/frozen" if a.mode == "frozen"
           else f"{a.arch}/ft_{'aug' if aug == 'flip_rot' else 'noaug'}")
    tag = f"__{aug}" if a.mode == "ft" else ""
    mean, std = norm_of(a.arch)
    ntag = "" if tuple(round(v, 3) for v in mean) == IMAGENET_MEAN else "__cfgnorm"
    name = f"{a.dataset}__{a.arch}__{a.mode}{tag}__T{a.trials}__R{a.repeats}{ntag}{'__amp' if a.amp else ''}"
    final_csv, part_csv = out_dir / f"{name}.csv", out_dir / f"{name}.partial.csv"
    if final_csv.exists() and not a.force:
        print(f"[{name}] já concluído, pulando")
        return
    dev = torch.device(a.device)
    Ximg, y_all = load_images(a.dataset, a.data_dir)
    C = len(np.unique(y_all))
    idx = dev_split(y_all)                      # mesmo desenvolvimento da via topológica
    yd = y_all[idx]
    X = torch.from_numpy(Ximg[idx]).permute(0, 3, 1, 2).contiguous().to(dev)  # uint8 (N,3,256,256)
    y = torch.as_tensor(yd, device=dev)
    bb = make_backbone(a.arch, not a.no_pretrained)
    dim = bb.num_features
    if a.mode == "frozen":
        X, bb = extract(bb, X, dev, mean, std), None       # embeddings uma vez (backbone pré-treinado, sem ajuste aos dados)
    ctx = dict(dev=dev, X=X, y=y, bb=bb, dim=dim, C=C, mode=a.mode, aug=a.aug == "flip_rot", mean=mean, std=std)
    n_bb = sum(p.numel() for p in make_backbone(a.arch, False).parameters()) if a.mode == "ft" else 0

    outer = list(RepeatedStratifiedKFold(n_splits=a.outer_k, n_repeats=a.repeats, random_state=SEED)
                 .split(np.zeros((len(yd), 1)), yd))
    rows = pd.read_csv(part_csv).to_dict("records") if (part_csv.exists() and not a.force) else []
    done = {r["fold"] for r in rows}
    t_job = time.time()
    for k, (tr, te) in enumerate(outer):
        if k in done:
            continue
        t0 = time.time()
        cfgs = sample_cfgs(a.trials, len(tr))
        inner = list(StratifiedKFold(a.inner_k, shuffle=True, random_state=SEED + k).split(tr, yd[tr]))
        loss, eps = [], []
        for ti, cfg in enumerate(cfgs):
            r = []
            for ii, (i, j) in enumerate(inner):
                model, ep = fit(ctx, cfg, tr[i], SEED + 1000 * k + 37 * ti + ii, a.max_epochs,
                                patience=a.patience, va=tr[j])
                p = predict(model, X, torch.as_tensor(tr[j], device=dev), dev)
                r.append((nll(yd[tr[j]], p / p.sum(1, keepdims=True)), ep))
            loss.append(float(np.mean([x[0] for x in r])))
            eps.append(float(np.mean([x[1] for x in r])))
        sel = int(np.argmin(loss))

        def fin(ti):
            ep = max(3, int(round(eps[ti])))
            model, _ = fit(ctx, cfgs[ti], tr, SEED + 1000 * k + 999, a.max_epochs, epochs=ep)
            p = predict(model, X, torch.as_tensor(te, device=dev), dev)
            return metrics(yd[te], p, C), ep

        m_t, ep_t = fin(sel)
        m_d, ep_d = (m_t, ep_t) if sel == 0 else fin(0)
        n_hd = sum(p.numel() for p in VNet(None, dim, C, 0.5).parameters())
        rows.append(dict(
            dataset=a.dataset, enc=enc, arch=a.arch, mode=a.mode, repeats=a.repeats, trials=a.trials,
            amp=int(a.amp), norm="cfg",
            aug=aug, fold=k,
            split_id=hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8],
            n_train=len(tr), n_test=len(te), n_classes=C,
            **m_t, **{f"d_{kk}": v for kk, v in m_d.items()},
            sel_idx=sel, sel_cfg=json.dumps(cfgs[sel]), epochs=ep_t, d_epochs=ep_d,
            inner_nll=loss[sel], d_inner_nll=loss[0], n_params=n_bb + n_hd,
            seconds=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(part_csv, index=False)
        print(f"[{name}] fold {k + 1}/{len(outer)}  F1={m_t['f1']:.3f} (padrão {m_d['f1']:.3f})  "
              f"cfg#{sel}  {rows[-1]['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    part_csv.unlink(missing_ok=True)
    print(f"[{name}] concluído em {time.time() - t_job:.0f}s  F1 médio={np.mean([r['f1'] for r in rows]):.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", required=True, choices=list(ARCHS))
    ap.add_argument("--mode", default="ft", choices=["ft", "frozen"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outer_k", type=int, default=5)
    ap.add_argument("--inner_k", type=int, default=3)
    ap.add_argument("--trials", type=int, default=12,
                    help="configurações por tarefa (a 0 é a padrão); 12 = mesmo orçamento da via topológica")
    ap.add_argument("--max_epochs", type=int, default=None, help="padrão: 30 (ft) ou 200 (frozen)")
    ap.add_argument("--patience", type=int, default=None, help="padrão: 5 (ft) ou 20 (frozen)")
    ap.add_argument("--aug", default="flip_rot", choices=["flip_rot", "none"], help="aumento de dados (só ft)")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2" / "visual"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="precisão mista bfloat16 (GPU Ada/Ampere): acelera o treino; muda levemente a numérica")
    ap.add_argument("--no_pretrained", action="store_true", help="só para teste de fumaça (sem baixar pesos)")
    ap.add_argument("--quick", action="store_true", help="teste de fumaça (rápido)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    global AMP
    AMP = bool(a.amp) and a.device.startswith("cuda")
    a.amp = AMP
    torch.backends.cudnn.benchmark = a.device.startswith("cuda")
    a.max_epochs = a.max_epochs or (30 if a.mode == "ft" else 200)
    a.patience = a.patience or (5 if a.mode == "ft" else 20)
    if a.quick:
        a.trials, a.inner_k, a.repeats, a.max_epochs, a.patience = 2, 2, 1, 3, 2
    run(a)


if __name__ == "__main__":
    main()
