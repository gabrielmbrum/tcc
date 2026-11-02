#!/usr/bin/env python
"""
Testes extras da ablação (não sobrescreve report.md):
  1. todos os codificadores contra a referência MLP (mesma representação), por dataset;
  2. representação diff vs cum, por codificador e dataset.
Mesmo teste t corrigido (Nadeau & Bengio), IC 95%, Holm e margem de equivalência do analise_ablacao.py.

Uso (da raiz do projeto, ao lado de src/analise_ablacao.py):
  python src/testes_extra.py [--ref mlp] [--margin 0.02] [--repeats 3] [--out_dir results_v2]
Saída: results_v2/report_extra.md, pareado_vs_mlp.csv, diff_vs_cum.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analise_ablacao import (DS_ORDER, ENC_ORDER, KEY, ROOT, corrected_ttest, holm, load_folds,
                             md, order)


def cell(r):
    return (f"{r.dif:+.3f} [{r.ic_inf:+.3f}, {r.ic_sup:+.3f}] p={r.p_holm:.2f}"
            f"{' eq.' if r.equiv else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2"))
    ap.add_argument("--ref", default="mlp")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    df = load_folds(out)
    rep_sel = a.repeats or int(df.repeats.max())
    df = df[(df.repeats == rep_sel) & (df.res == "native") & (df.rmax == 15)]
    ratio = float((df["n_test"] / df["n_train"]).mean())
    dss = order(df.dataset.unique(), DS_ORDER)
    R = [f"# Testes extras (repeats={rep_sel}, razão teste/treino={ratio:.2f}, margem ±{a.margin})\n"]

    # 1. todos contra a referência
    R.append(f"## 1. Comparação pareada contra `{a.ref}` (mesma representação)\n")
    R.append(f"Diferença = F1(codificador) − F1({a.ref}); positiva favorece o codificador. "
             f"Holm dentro de cada dataset e representação.\n")
    rows = []
    for (ds, rep), g in df.groupby(["dataset", "rep"]):
        ref = g[g.enc == a.ref].set_index("fold")
        if ref.empty:
            continue
        fam = []
        for enc, ge in g[g.enc != a.ref].groupby("enc"):
            ge = ge.set_index("fold")
            j = ge.index.intersection(ref.index)
            assert (ge.loc[j, "split_id"] == ref.loc[j, "split_id"]).all(), "partições desalinhadas"
            m, ci, p = corrected_ttest((ge.loc[j, "f1"] - ref.loc[j, "f1"]).values, ratio)
            fam.append(dict(dataset=ds, rep=rep, enc=enc, dif=m, ic_inf=ci[0], ic_sup=ci[1], p=p,
                            equiv=bool(ci[0] > -a.margin and ci[1] < a.margin)))
        for f, q in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm"] = q
        rows += fam
    p1 = pd.DataFrame(rows)
    p1.to_csv(out / f"pareado_vs_{a.ref}.csv", index=False)
    for rep in ("cum", "diff"):
        s = p1[p1.rep == rep]
        if s.empty:
            continue
        t = s.assign(c=s.apply(cell, axis=1)).pivot(index="enc", columns="dataset", values="c")
        t = t.reindex(order(t.index, ENC_ORDER))[[d for d in dss if d in t.columns]]
        t.index.name = f"vs {a.ref}/{rep}"
        R.append(md(t) + "\n")
    sig = p1[p1.p_holm < 0.05]
    R.append(f"Significativas após Holm (p<0.05): {len(sig)} de {len(p1)}.\n")
    if len(sig):
        R.append(md(sig.set_index(["dataset", "rep", "enc"])[["dif", "p_holm"]].round(3)) + "\n")

    # 2. diff vs cum
    R.append("## 2. Representação: diff vs cum (mesmo codificador)\n")
    R.append("Diferença = F1(diff) − F1(cum); positiva favorece `diff`. Holm entre os datasets de cada "
             "codificador.\n")
    rows = []
    for enc, ge in df[df.enc != "dummy"].groupby("enc"):
        fam = []
        for ds, g in ge.groupby("dataset"):
            c, d_ = g[g.rep == "cum"].set_index("fold"), g[g.rep == "diff"].set_index("fold")
            j = c.index.intersection(d_.index)
            if len(j) < 3:
                continue
            assert (c.loc[j, "split_id"] == d_.loc[j, "split_id"]).all(), "partições desalinhadas"
            m, ci, p = corrected_ttest((d_.loc[j, "f1"] - c.loc[j, "f1"]).values, ratio)
            fam.append(dict(enc=enc, dataset=ds, f1_cum=c.loc[j, "f1"].mean(), f1_diff=d_.loc[j, "f1"].mean(),
                            dif=m, ic_inf=ci[0], ic_sup=ci[1], p=p,
                            equiv=bool(ci[0] > -a.margin and ci[1] < a.margin)))
        for f, q in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm"] = q
        rows += fam
    p2 = pd.DataFrame(rows)
    p2.to_csv(out / "diff_vs_cum.csv", index=False)
    t = p2.assign(c=p2.apply(cell, axis=1)).pivot(index="enc", columns="dataset", values="c")
    t = t.reindex(order(t.index, ENC_ORDER))[[d for d in dss if d in t.columns]]
    t.index.name = "diff − cum"
    R.append(md(t) + "\n")
    sm = p2.groupby("enc").agg(dif_media=("dif", "mean"), diff_melhor=("dif", lambda s: int((s > 0).sum())),
                               datasets=("dif", "size"), signif_holm=("p_holm", lambda s: int((s < 0.05).sum())))
    sm = sm.reindex(order(sm.index, ENC_ORDER))
    sm.index.name = "codificador"
    R.append(md(sm.round({"dif_media": 3}).astype(object)) + "\n")
    R.append(f"Significativas após Holm (p<0.05): {int((p2.p_holm < 0.05).sum())} de {len(p2)}.\n")

    (out / "report_extra.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
