#!/usr/bin/env python
"""
Análise da ablação da via visual (results_v2/visual/folds) -> results_v2/visual/report_visual.md.

  1. F1 macro (média ± desvio) das configurações visuais e das referências topológicas (BiLSTM e MLP,
     cum e diff, r_max=15, resolução nativa, lidas de results_v2/folds), mais o ranking médio.
  2. Configurações visuais contra a de referência (--ref, padrão mobilevit_s/ft), pareado fold a fold.
  3. Referência visual contra cada referência topológica, pareado (mesmos folds: split_id conferido).
  4. Ganho do ajuste de hiperparâmetros e configurações selecionadas.
Mesmo teste t corrigido (Nadeau & Bengio), IC 95%, Holm por dataset e margem de equivalência do
analise_ablacao.py (importado).

Uso: python src/analise_visual.py [--ref mobilevit_s/ft] [--margin 0.02] [--repeats 3] [--out_dir results_v2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analise_ablacao import DS_ORDER, ROOT, corrected_ttest, holm, load_folds, md, ms, order  # noqa: E402

TOPO = [("bilstm", "cum"), ("bilstm", "diff"), ("mlp", "cum"), ("mlp", "diff")]


def paired(a, b, ratio, margin):
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return None
    assert (a.loc[j, "split_id"] == b.loc[j, "split_id"]).all(), "partições desalinhadas"
    m, ci, p = corrected_ttest((a.loc[j, "f1"] - b.loc[j, "f1"]).values, ratio)
    return dict(dif=m, ic_inf=ci[0], ic_sup=ci[1], p=p, equiv=bool(ci[0] > -margin and ci[1] < margin))


def cell(r):
    return f"{r.dif:+.3f} [{r.ic_inf:+.3f}, {r.ic_sup:+.3f}] p={r.p_holm:.2f}{' eq.' if r.equiv else ''}"


def table(rows, dss, name):
    d = pd.DataFrame(rows)
    d["p_holm"] = np.nan
    for _, ix in d.groupby("dataset").groups.items():
        d.loc[ix, "p_holm"] = holm(d.loc[ix, "p"].values)
    t = d.assign(c=d.apply(cell, axis=1)).pivot(index="alvo", columns="dataset", values="c")
    t = t.reindex([x for x in d.alvo.unique()])[[x for x in dss if x in t.columns]]
    t.index.name = name
    return d, md(t) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2"))
    ap.add_argument("--ref", default="mobilevit_s/ft_aug")
    ap.add_argument("--trials", type=int, default=0, help="configurações por tarefa a analisar (0 = o maior encontrado)")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=3, help="repetições dos resultados visuais")
    ap.add_argument("--top_repeats", type=int, default=3, help="repetições dos resultados topológicos")
    a = ap.parse_args()
    out = Path(a.out_dir)
    files = [f for f in sorted((out / "visual" / "folds").glob("*.csv")) if not f.name.endswith(".partial.csv")]
    if not files:
        raise SystemExit(f"nenhum resultado em {out}/visual/folds")
    vis = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "trials" not in vis:
        vis["trials"] = 6   # primeira versão: 6 configurações, sem a distinção ft_aug/ft_noaug
    vis["trials"] = vis["trials"].fillna(6).astype(int)
    if "norm" not in vis:
        vis["norm"] = "legacy"
    vis["norm"] = vis["norm"].fillna("legacy")   # "legacy": 1ª e 2ª rodadas (normalização ImageNet em todas)
    trials = a.trials or int(vis.trials.max())
    vis = vis[(vis.repeats == a.repeats) & (vis.trials == trials)]
    # se uma configuração foi refeita com a normalização do pretrained_cfg (norm="cfg"), ela substitui a antiga
    vis = vis.sort_values("norm", key=lambda c: c == "cfg", kind="stable")
    vis = vis.drop_duplicates(["dataset", "enc", "fold"], keep="last").copy()
    if vis.empty:
        raise SystemExit(f"nenhum resultado visual com repeats={a.repeats} e trials={trials}")
    vis["rep"] = "rgb"
    top = load_folds(out)
    top = top[(top.repeats == a.top_repeats) & (top.res == "native") & (top.rmax == 15)]
    ratio = float((vis.n_test / vis.n_train).mean())
    dss = order(vis.dataset.unique(), DS_ORDER)
    vis["alvo"] = vis.enc
    if top.empty:
        raise SystemExit(f"nenhum resultado topológico com repeats={a.top_repeats} em {out}/folds "
                         f"(use --top_repeats)")
    top = top[pd.Series([(e, r) in TOPO for e, r in zip(top.enc, top.rep)], index=top.index)].copy()
    top["alvo"] = "topológica " + top.enc + "/" + top.rep
    allr = pd.concat([vis, top], ignore_index=True)
    R = [f"# Ablação da via visual (repeats={a.repeats}, configurações por tarefa={trials}, "
         f"razão teste/treino={ratio:.2f}, margem ±{a.margin})\n"]

    R.append("## 1. F1 macro (média ± desvio entre folds externos)\n")
    t = allr.groupby(["alvo", "dataset"])["f1"].apply(ms).unstack("dataset")[dss]
    t.index.name = "configuração"
    R.append(md(t) + "\n")
    piv = allr.groupby(["dataset", "alvo"])["f1"].mean().unstack("alvo")
    rk = piv.rank(axis=1, ascending=False).mean().sort_values().round(2).to_frame("rank_medio")
    rk.index.name = "configuração"
    R.append("Ranking médio entre datasets (descritivo):\n\n" + md(rk, 2) + "\n")

    if a.ref in set(vis.alvo):
        R.append(f"## 2. Configurações visuais contra `{a.ref}`\n")
        R.append("Diferença = F1(configuração) − F1(referência). Holm por dataset.\n")
        rows = []
        for ds, g in vis.groupby("dataset"):
            ref = g[g.alvo == a.ref].set_index("fold")
            for al, ge in g[g.alvo != a.ref].groupby("alvo"):
                res = paired(ge.set_index("fold"), ref, ratio, a.margin)
                if res:
                    rows.append(dict(dataset=ds, alvo=al, **res))
        if rows:
            d, s = table(rows, dss, f"vs {a.ref}")
            R.append(s + f"Significativas após Holm (p<0.05): {int((d.p_holm < 0.05).sum())} de {len(d)}.\n")

        R.append(f"## 3. `{a.ref}` contra a via topológica (mesmos folds)\n")
        R.append("Diferença = F1(visual) − F1(topológica); positiva favorece a via visual. Holm por dataset.\n")
        rows = []
        for ds, g in allr.groupby("dataset"):
            v = g[g.alvo == a.ref].set_index("fold")
            for al, ge in g[g.alvo.str.startswith("topológica")].groupby("alvo"):
                res = paired(v, ge.set_index("fold"), ratio, a.margin)
                if res:
                    rows.append(dict(dataset=ds, alvo=al, **res))
        if rows:
            d, s = table(rows, dss, "visual − topológica")
            R.append(s + f"Significativas após Holm (p<0.05): {int((d.p_holm < 0.05).sum())} de {len(d)}.\n")

    R.append("## 4. Ganho do ajuste (F1 selecionado − F1 padrão) e configurações selecionadas\n")
    g = vis.assign(ganho=vis.f1 - vis.d_f1).groupby(["alvo", "dataset"])["ganho"].mean().unstack("dataset")[dss]
    g.index.name = "configuração"
    R.append(md(g.round(3)) + "\n")
    for al, ge in vis.groupby("alvo"):
        cfg = pd.DataFrame([json.loads(c) for c in ge.sel_cfg])
        parts = [f"{c}: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg[c].value_counts(normalize=True).round(2).items()))
                 for c in ("lr_bb", "lr_hd", "wd", "dropout", "bs") if c in cfg and (c != "lr_bb" or "/ft" in al)]
        parts += [f"config padrão em {(ge.sel_idx == 0).mean():.0%} dos folds",
                  f"épocas medianas={ge.epochs.median():.0f}", f"parâmetros={ge.n_params.median():.0f}",
                  f"tempo mediano por fold={ge.seconds.median():.0f}s"]
        R.append(f"- **{al}**: " + "; ".join(parts))
    R.append("")

    # 5. efeito do aumento de dados (ajuste fino, mesmo backbone, mesmos folds)
    archs = sorted({x.split("/")[0] for x in vis.alvo if x.endswith("/ft_aug")
                    and x.replace("/ft_aug", "/ft_noaug") in set(vis.alvo)})
    if archs:
        R.append("## 5. Efeito do aumento de dados (ajuste fino): com − sem aumento\n")
        R.append("Diferença = F1(inversões e rotações) − F1(sem aumento), mesmo backbone e mesmos folds. "
                 "Holm entre os datasets de cada arquitetura; positiva favorece o aumento.\n")
        rows = []
        for arch in archs:
            fam = []
            for ds, g in vis.groupby("dataset"):
                c = g[g.alvo == f"{arch}/ft_aug"].set_index("fold")
                n = g[g.alvo == f"{arch}/ft_noaug"].set_index("fold")
                res = paired(c, n, ratio, a.margin)
                if res:
                    fam.append(dict(alvo=arch, dataset=ds, f1_aug=c.f1.mean(), f1_sem=n.f1.mean(), **res))
            if fam:
                for f, q in zip(fam, holm([f["p"] for f in fam])):
                    f["p_holm"] = q
                rows += fam
        if rows:
            d = pd.DataFrame(rows)
            t = d.assign(c=d.apply(lambda r: f"{r.f1_sem:.3f} → {r.f1_aug:.3f} ({cell(r)})", axis=1)) \
                 .pivot(index="alvo", columns="dataset", values="c")
            t = t[[x for x in dss if x in t.columns]]
            t.index.name = "aumento (sem → com)"
            R.append(md(t) + "\n")
            sm = d.groupby("alvo").agg(dif_media=("dif", "mean"), aumento_melhor=("dif", lambda s: int((s > 0).sum())),
                                       datasets=("dif", "size"),
                                       signif_holm=("p_holm", lambda s: int((s < 0.05).sum())))
            sm.index.name = "arquitetura"
            R.append(md(sm.round({"dif_media": 3}).astype(object)) + "\n")

    (out / "visual" / "report_visual.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
