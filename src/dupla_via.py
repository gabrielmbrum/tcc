#!/usr/bin/env python
"""
Teste final: modelo de dupla via (topológica + visual), treinado de ponta a ponta.

Segue o protocolo do capítulo de Materiais e Métodos e reaproveita das ablações:
  - ablacao_robusta.py: SEED, dev_split (holdout 20% NUNCA usado na CV), metrics, nll;
  - ablacao_visual.py : load_images (256x256, mesma ordem de topological.npz), augment, make_backbone, norm_of.
Os folds externos (RepeatedStratifiedKFold 5 x repeats, SEED) e internos (StratifiedKFold SEED+k) são IDÊNTICOS
aos das ablações (mesmo split_id), o que permite comparar a dupla via com cada via isolada fold a fold.

Arquitetura (Seções 3.2 a 3.4):
  via topológica : X_top (15, 18) -> padronização por coordenada (estatísticas do treino do ajuste)
                   -> BiLSTM (1 camada, 64 por direção, [h_fwd_15 || h_bwd_1]) ou MLP (Linear(270,128)+ReLU) -> z_top (128)
  via visual     : MobileViT-S (timm, ImageNet, normalização do pretrained_cfg) -> z_vis (640) -> Linear(640,128)
  fusão          : LN(z_top) || LN(z_vis proj.) -> 256
  classificador  : Linear(256,128) -> ReLU -> Dropout(0.5) -> Linear(128,C)

Treino: AdamW, norma do gradiente <= 1, sem agendador; três grupos de parâmetros (backbone lr_bb; codificador
topológico lr 1e-3; projeção+LN+classificador lr_hd). Espaço de busca = produto de --lr_bb x --lr_hd (4 configurações
por padrão); wd 1e-4, dropout 0.5, lote 8 (treino < 100) ou 16. Aumento (inversão + rotação de 90 graus) só na via
visual e só no treino. Seleção pela entropia cruzada interna (CV interna, parada antecipada só nela); ajuste final com
nº fixo de épocas = média das épocas internas (mínimo 3).

Modos:
  (padrão)      CV aninhada nos 80% de desenvolvimento: 15 folds externos -> <out>/folds/<nome>.csv
  --holdout     seleção na CV interna sobre os 80% inteiros, treino final nos 80% e avaliação ÚNICA nos 20%
                -> <out>/holdout/<nome>.csv
Controle de capacidade (--ctrl): none | permtrain (descritor topológico permutado entre as imagens do TREINO) |
  permtest (permutado entre as imagens do TESTE). A permutação mantém o par (imagem visual, rótulo) e troca só a
  sequência topológica.
As probabilidades preditas de cada fold/holdout ficam em <out>/probs/*.npz (análise de complementaridade dos erros).

Exemplos:
  uv run src/dupla_via.py --dataset UCSB --enc mlp --rep diff --device cuda
  uv run src/dupla_via.py --dataset UCSB --enc mlp --rep diff --holdout
  uv run src/dupla_via.py --dataset UCSB --enc bilstm --rep cum --quick --no_pretrained --device cpu
"""
import argparse
import copy
import hashlib
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacao_robusta import SEED, dev_split, metrics, nll  # noqa: E402
from ablacao_visual import augment, load_images, make_backbone, norm_of  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ARCH = "mobilevit_s"
LR_TOP = 1e-3       # codificador topológico (aleatório), padrão da ablação topológica
AMP = False         # precisão mista bfloat16 (--amp); definido em main()
N_RAIOS, N_FEAT = 15, 18


# ------------------------------------------------------------------ dados
def prep_topo(X, rep):
    """X (N, 15, 18) acumulado; 'diff' = diferenças entre raios consecutivos (dx_1 = x_1)."""
    if rep == "diff":
        X = np.concatenate([X[:, :1], np.diff(X, axis=1)], axis=1)
    return X.astype(np.float32)


def load_topo(ds, data_dir, y_ref):
    d = np.load(Path(data_dir) / ds / "topological.npz")
    if not np.array_equal(d["y"], y_ref):
        raise ValueError(f"{ds}: rótulos de topological.npz diferem dos de visual_256.npz")
    X = d["X"]
    if X.shape[1:] != (N_RAIOS, N_FEAT):
        raise ValueError(f"{ds}: esperado (N, {N_RAIOS}, {N_FEAT}), obtido {X.shape}")
    return X


# ------------------------------------------------------------------ modelo
class TopEnc(nn.Module):
    """Codificador topológico: embedding de 128 dimensões."""

    def __init__(self, kind, hidden=64, emb=128):
        super().__init__()
        self.kind = kind
        if kind == "bilstm":
            self.rnn = nn.LSTM(N_FEAT, hidden, batch_first=True, bidirectional=True)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(N_RAIOS * N_FEAT, emb), nn.ReLU())
        else:
            raise ValueError(kind)

    def forward(self, x):  # x: (B, 15, 18)
        if self.kind == "bilstm":
            _, (h, _) = self.rnn(x)          # h: (2, B, 64)
            return torch.cat([h[0], h[1]], dim=-1)   # [h_fwd_final || h_bwd_final]
        return self.net(x)


class DualNet(nn.Module):
    def __init__(self, backbone, dim_vis, enc, C, dropout, mean, std, emb=128):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        # padronização topológica (estatísticas do treino do ajuste), preenchida em fit()
        self.register_buffer("mu", torch.zeros(1, N_RAIOS, N_FEAT))
        self.register_buffer("sd", torch.ones(1, N_RAIOS, N_FEAT))
        self.bb = backbone
        self.proj = nn.Linear(dim_vis, emb)
        self.top = TopEnc(enc, emb // 2, emb)
        self.ln_t = nn.LayerNorm(emb)
        self.ln_v = nn.LayerNorm(emb)
        self.cls = nn.Sequential(nn.Linear(2 * emb, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, C))

    def forward(self, img, top):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            v = self.proj(self.bb((img.float() / 255 - self.mean) / self.std))
        t = self.top((top - self.mu) / self.sd)          # codificador topológico sempre em float32
        z = torch.cat([self.ln_t(t), self.ln_v(v.float())], dim=1)
        return self.cls(z)


def make_tmap(n, subset, seed, dev):
    """Mapa de índices topológicos: identidade, exceto em `subset`, onde as sequências são permutadas entre si."""
    subset = np.asarray(subset)
    m = torch.arange(n, device=dev)
    perm = np.random.default_rng(seed).permutation(len(subset))
    m[torch.as_tensor(subset, device=dev)] = torch.as_tensor(subset[perm], device=dev)
    return m


@torch.no_grad()
def predict(model, ctx, idx, tmap=None, bs=64):
    model.eval()
    tmap = ctx["ident"] if tmap is None else tmap
    idx_t = torch.as_tensor(idx, device=ctx["dev"])
    out = []
    for i in range(0, len(idx_t), bs):
        b = idx_t[i:i + bs]
        out.append(F.softmax(model(ctx["X"][b], ctx["T"][tmap[b]]), 1))
    return torch.cat(out).cpu().numpy()


def fit(ctx, cfg, tr, seed, max_epochs, patience=None, va=None, epochs=None):
    """Com `va`: parada antecipada pela perda de validação (restaura os melhores pesos).
    Sem `va`: treina exatamente `epochs` épocas."""
    dev, X, T, y = ctx["dev"], ctx["X"], ctx["T"], ctx["y"]
    torch.manual_seed(seed)
    tr_t = torch.as_tensor(tr, device=dev)
    tmap = make_tmap(len(y), tr, seed, dev) if ctx["ctrl"] == "permtrain" else ctx["ident"]
    rows = T[tmap[tr_t]]
    bb = copy.deepcopy(ctx["bb"]).to(dev)
    model = DualNet(bb, ctx["dim"], ctx["enc"], ctx["C"], cfg["dropout"], ctx["mean"], ctx["std"]).to(dev)
    with torch.no_grad():   # padronização por coordenada (15 x 18), só com o treino deste ajuste
        model.mu.copy_(rows.mean(0, keepdim=True))
        model.sd.copy_(rows.std(0, unbiased=False, keepdim=True) + 1e-8)   # ddof=0, igual a standardize() da ablação
    groups = [{"params": list(model.bb.parameters()), "lr": cfg["lr_bb"]},
              {"params": list(model.top.parameters()), "lr": LR_TOP},
              {"params": (list(model.proj.parameters()) + list(model.ln_t.parameters())
                          + list(model.ln_v.parameters()) + list(model.cls.parameters())), "lr": cfg["lr_hd"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"])
    g = torch.Generator().manual_seed(seed)
    n, bs = len(tr), cfg["bs"]
    best, best_state, best_ep, wait = float("inf"), None, 1, 0
    for ep in range(max_epochs if va is not None else epochs):
        model.train()
        perm = tr_t[torch.randperm(n, generator=g).to(dev)]
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:  # BatchNorm exige >1 amostra
                continue
            opt.zero_grad()
            F.cross_entropy(model(augment(X[b]), T[tmap[b]]), y[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if va is not None:
            p = predict(model, ctx, va)
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


def make_cfgs(n_train, lr_bb, lr_hd):
    bs = 8 if n_train < 100 else 16
    return [dict(lr_bb=float(b), lr_hd=float(h), wd=1e-4, dropout=0.5, bs=bs)
            for b, h in itertools.product(lr_bb, lr_hd)]


# ------------------------------------------------------------------ seleção + ajuste final
def select_and_eval(ctx, a, tr, te, k):
    """Seleciona a configuração na CV interna de `tr`, treina em todo `tr` (nº fixo de épocas) e avalia em `te`."""
    dev, yn = ctx["dev"], ctx["yn"]
    cfgs = make_cfgs(len(tr), a.lr_bb, a.lr_hd)
    inner = list(StratifiedKFold(a.inner_k, shuffle=True, random_state=SEED + k).split(tr, yn[tr]))
    loss, eps = [], []
    for ti, cfg in enumerate(cfgs):
        r = []
        for ii, (i, j) in enumerate(inner):
            model, ep = fit(ctx, cfg, tr[i], SEED + 1000 * k + 37 * ti + ii, a.max_epochs,
                            patience=a.patience, va=tr[j])
            p = predict(model, ctx, tr[j])
            r.append((nll(yn[tr[j]], p / p.sum(1, keepdims=True)), ep))
        loss.append(float(np.mean([x[0] for x in r])))
        eps.append(float(np.mean([x[1] for x in r])))
    sel = int(np.argmin(loss))
    ep = max(3, int(round(eps[sel])))
    model, _ = fit(ctx, cfgs[sel], tr, SEED + 1000 * k + 999, a.max_epochs, epochs=ep)
    tmap = make_tmap(len(yn), te, SEED + k + 555, dev) if ctx["ctrl"] == "permtest" else None
    p = predict(model, ctx, te, tmap)
    n_params = sum(p_.numel() for p_ in model.parameters())
    return p, sel, cfgs, loss, ep, n_params


# ------------------------------------------------------------------ execução
def run(a):
    root = Path(a.out_dir)
    out_dir = root / ("holdout" if a.holdout else "folds")
    probs_dir = root / "probs"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)
    label = f"dupla/{a.enc}_{a.rep}" + ("" if a.ctrl == "none" else f"/{a.ctrl}")
    name = (f"{a.dataset}__dupla__{a.enc}__{a.rep}__{a.ctrl}"
            + ("__holdout" if a.holdout else f"__R{a.repeats}") + a.tag + ("__amp" if a.amp else ""))
    final_csv, part_csv = out_dir / f"{name}.csv", out_dir / f"{name}.partial.csv"
    if final_csv.exists() and not a.force:
        print(f"[{name}] já concluído, pulando")
        return
    dev = torch.device(a.device)

    # dados: desenvolvimento (posições 0..nd-1) seguido do holdout (posições nd..N-1)
    Ximg, y_all = load_images(a.dataset, a.data_dir)
    Xtop = prep_topo(load_topo(a.dataset, a.data_dir, y_all), a.rep)
    C = len(np.unique(y_all))
    idx = dev_split(y_all)
    hold = np.setdiff1d(np.arange(len(y_all)), idx)
    order = np.concatenate([idx, hold])
    nd = len(idx)
    yn = y_all[order].astype(np.int64)
    X = torch.from_numpy(Ximg[order]).permute(0, 3, 1, 2).contiguous().to(dev)   # uint8 (N,3,256,256)
    T = torch.from_numpy(Xtop[order]).to(dev)                                    # float32 (N,15,18)
    mean, std = norm_of(ARCH)
    bb = make_backbone(ARCH, not a.no_pretrained)
    ctx = dict(dev=dev, X=X, T=T, y=torch.as_tensor(yn, device=dev), yn=yn, bb=bb, dim=bb.num_features,
               C=C, enc=a.enc, ctrl=a.ctrl, mean=mean, std=std, ident=torch.arange(len(yn), device=dev))
    common = dict(dataset=a.dataset, enc=label, top_enc=a.enc, rep=a.rep, ctrl=a.ctrl, arch=ARCH,
                  repeats=a.repeats, trials=len(a.lr_bb) * len(a.lr_hd), amp=int(a.amp), tag=a.tag,
                  max_epochs=a.max_epochs, patience=a.patience, n_classes=C)

    if a.holdout:
        tr, te = np.arange(nd), np.arange(nd, len(yn))
        t0 = time.time()
        p, sel, cfgs, loss, ep, n_params = select_and_eval(ctx, a, tr, te, k=99)
        m = metrics(yn[te], p, C)
        row = dict(**common, protocol="holdout", n_train=len(tr), n_test=len(te), **m, sel_idx=sel,
                   sel_cfg=json.dumps(cfgs[sel]), epochs=ep, inner_nll=loss[sel], n_params=n_params,
                   seconds=round(time.time() - t0, 1))
        np.savez(probs_dir / f"{name}.npz", idx=order[te], y=yn[te], proba=p)
        pd.DataFrame([row]).to_csv(final_csv, index=False)
        print(f"[{name}] holdout  F1={m['f1']:.3f}  acc={m['acc']:.3f}  cfg#{sel}  {row['seconds']}s", flush=True)
        return

    outer = list(RepeatedStratifiedKFold(n_splits=a.outer_k, n_repeats=a.repeats, random_state=SEED)
                 .split(np.zeros((nd, 1)), yn[:nd]))
    rows = pd.read_csv(part_csv).to_dict("records") if (part_csv.exists() and not a.force) else []
    done = {r["fold"] for r in rows}
    t_job = time.time()
    for k, (tr, te) in enumerate(outer):
        if k in done:
            continue
        t0 = time.time()
        p, sel, cfgs, loss, ep, n_params = select_and_eval(ctx, a, tr, te, k)
        m = metrics(yn[te], p, C)
        np.savez(probs_dir / f"{name}__fold{k}.npz", idx=order[te], y=yn[te], proba=p)
        rows.append(dict(
            **common, protocol="cv", fold=k, split_id=hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8],
            n_train=len(tr), n_test=len(te), **m, sel_idx=sel, sel_cfg=json.dumps(cfgs[sel]), epochs=ep,
            inner_nll=loss[sel], n_params=n_params, seconds=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(part_csv, index=False)
        print(f"[{name}] fold {k + 1}/{len(outer)}  F1={m['f1']:.3f}  cfg#{sel}  {rows[-1]['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    part_csv.unlink(missing_ok=True)
    print(f"[{name}] concluído em {time.time() - t_job:.0f}s  F1 médio={np.mean([r['f1'] for r in rows]):.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--enc", default="mlp", choices=["mlp", "bilstm"], help="codificador da via topológica")
    ap.add_argument("--rep", default="diff", choices=["cum", "diff"], help="medidas acumuladas ou diferenças")
    ap.add_argument("--ctrl", default="none", choices=["none", "permtrain", "permtest"],
                    help="controle de capacidade: descritor topológico permutado no treino ou no teste")
    ap.add_argument("--holdout", action="store_true", help="avaliação única no holdout de 20%%")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outer_k", type=int, default=5)
    ap.add_argument("--inner_k", type=int, default=3)
    ap.add_argument("--lr_bb", type=float, nargs="+", default=[1e-4, 3e-4], help="espaço da taxa do backbone")
    ap.add_argument("--lr_hd", type=float, nargs="+", default=[1e-4, 1e-3], help="espaço da taxa de projeção+classificador")
    ap.add_argument("--max_epochs", type=int, default=50)   # [PILOTO] valor a confirmar
    ap.add_argument("--patience", type=int, default=10)     # [PILOTO] valor a confirmar
    ap.add_argument("--tag", default="", help="sufixo no nome dos arquivos (ex.: __lrbb1e-3 num piloto)")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2" / "final"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="precisão mista bfloat16 (GPU Ada/Ampere)")
    ap.add_argument("--no_pretrained", action="store_true", help="só para teste de fumaça (sem baixar pesos)")
    ap.add_argument("--quick", action="store_true", help="teste de fumaça (rápido)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    global AMP
    AMP = bool(a.amp) and a.device.startswith("cuda")
    a.amp = AMP
    torch.backends.cudnn.benchmark = a.device.startswith("cuda")
    if a.quick:
        a.inner_k, a.repeats, a.max_epochs, a.patience = 2, 1, 3, 2
        a.lr_bb, a.lr_hd = a.lr_bb[:1], a.lr_hd[:1]
    run(a)


if __name__ == "__main__":
    main()
