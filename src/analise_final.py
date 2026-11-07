#!/usr/bin/env python
"""
Análise do teste final (results_v2/final) -> results_v2/final/report_final.md.

  1. F1 macro (média ± desvio) da dupla via (variantes e controles), da via visual de referência
     (--ref_visual, padrão mobilevit_s/ft_aug) e das variantes topológicas (BiLSTM e MLP, cum e diff), com ranking médio.
  2. Variante primária (--primary, padrão dupla/mlp_diff) contra a via visual, a via topológica correspondente e os
     controles de capacidade (descritor topológico permutado no treino e no teste), pareado fold a fold.
  3. Todas as variantes da dupla via contra a via visual e contra a via topológica correspondente.
  4. Variantes contra a primária (análise secundária: codificador e representação).
  5. Configurações selecionadas, épocas, parâmetros e tempo.
  6. Holdout (avaliação única, sem medida de dispersão).
Mesmo teste t corrigido (Nadeau & Bengio), IC 95% e margem de equivalência do analise_ablacao.py (importado).
Valores de p ajustados por Holm entre os datasets, dentro de cada comparação (o alvo de cada linha da tabela).

Uso: python src/analise_final.py [--repeats 3] [--tag ""] [--primary dupla/mlp_diff] [--out_dir results_v2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analise_ablacao import DS_ORDER, ROOT, holm, load_folds, md, ms, order  # noqa: E402
from analise_visual import TOPO, cell, paired  # noqa: E402


def load_csvs(d):
    files = [f for f in sorted(Path(d).glob("*.csv")) if not f.name.endswith(".partial.csv")]
    return (pd.concat([pd.read_csv(f, dtype={"tag": str}) for f in files], ignore_index=True)
            if files else pd.DataFrame())


def table_ds(rows, dss, name):
    """Holm entre os datasets, separadamente para cada alvo (linha)."""
    d = pd.DataFrame(rows)
    d["p_holm"] = np.nan
    for _, ix in d.groupby("alvo").groups.items():
        d.loc[ix, "p_holm"] = holm(d.loc[ix, "p"].values)
    t = d.assign(c=d.apply(cell, axis=1)).pivot(index="alvo", columns="dataset", values="c")
    t = t.reindex(list(d.alvo.unique()))[[x for x in dss if x in t.columns]]
    t.index.name = name
    return d, md(t) + "\n"


def pair_rows(allr, a_name, b_name, ratio, margin, label):
    rows = []
    for ds, g in allr.groupby("dataset"):
        x = g[g.alvo == a_name].set_index("fold")
        y = g[g.alvo == b_name].set_index("fold")
        if x.empty or y.empty:
            continue
        res = paired(x, y, ratio, margin)
        if res:
            rows.append(dict(dataset=ds, alvo=label, **res))
    return rows


def emit(R, rows, dss, name, note=""):
    if not rows:
        R.append("(sem dados)\n")
        return
    d, s = table_ds(rows, dss, name)
    R.append(s + f"Significativas após Holm (p<0.05): {int((d.p_holm < 0.05).sum())} de {len(d)}. {note}\n")


def topo_of(label):
    enc, rep = label.split("/")[1].split("_")
    return enc, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2"))
    ap.add_argument("--ref_visual", default="mobilevit_s/ft_aug")
    ap.add_argument("--primary", default="dupla/mlp_diff")
    ap.add_argument("--tag", default="")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=3, help="repetições dos resultados da dupla via")
    ap.add_argument("--vis_repeats", type=int, default=3)
    ap.add_argument("--vis_trials", type=int, default=0, help="0 = o maior encontrado")
    ap.add_argument("--top_repeats", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out_dir)

    # ---- dupla via
    dual = load_csvs(out / "final" / "folds")
    if dual.empty:
        raise SystemExit(f"nenhum resultado em {out}/final/folds")
    dual["tag"] = dual["tag"].fillna("") if "tag" in dual else ""
    dual = dual[(dual.repeats == a.repeats) & (dual.tag == a.tag)]
    dual = dual.drop_duplicates(["dataset", "enc", "fold"], keep="last").copy()
    if dual.empty:
        raise SystemExit(f"nenhum resultado da dupla via com repeats={a.repeats} e tag='{a.tag}'")
    dual["alvo"] = dual.enc

    # ---- via visual (mesma lógica do analise_visual.py)
    vis = load_csvs(out / "visual" / "folds")
    if vis.empty:
        raise SystemExit(f"nenhum resultado em {out}/visual/folds")
    if "trials" not in vis:
        vis["trials"] = 6
    vis["trials"] = vis["trials"].fillna(6).astype(int)
    if "norm" not in vis:
        vis["norm"] = "legacy"
    vis["norm"] = vis["norm"].fillna("legacy")
    trials = a.vis_trials or int(vis.trials.max())
    vis = vis[(vis.repeats == a.vis_repeats) & (vis.trials == trials) & (vis.enc == a.ref_visual)]
    vis = vis.sort_values("norm", key=lambda c: c == "cfg", kind="stable")
    vis = vis.drop_duplicates(["dataset", "enc", "fold"], keep="last").copy()
    if vis.empty:
        raise SystemExit(f"referência visual '{a.ref_visual}' não encontrada")
    vis["alvo"] = "visual " + vis.enc

    # ---- via topológica
    top = load_folds(out)
    top = top[(top.repeats == a.top_repeats) & (top.res == "native") & (top.rmax == 15)]
    top = top[pd.Series([(e, r) in TOPO for e, r in zip(top.enc, top.rep)], index=top.index)].copy()
    top["alvo"] = "topológica " + top.enc + "/" + top.rep

    allr = pd.concat([dual, vis, top], ignore_index=True)
    ratio = float((dual.n_test / dual.n_train).mean())
    dss = order(dual.dataset.unique(), DS_ORDER)
    vref = "visual " + a.ref_visual
    variants = sorted(x for x in dual.alvo.unique() if x.count("/") == 1)
    controls = sorted(x for x in dual.alvo.unique() if x.count("/") == 2)
    R = [f"# Teste final: dupla via (repeats={a.repeats}, configurações por tarefa="
         f"{int(dual.trials.max())}, razão teste/treino={ratio:.2f}, margem ±{a.margin})\n"]

    # 1. F1 macro
    R.append("## 1. F1 macro (média ± desvio entre folds externos)\n")
    t = allr.groupby(["alvo", "dataset"])["f1"].apply(ms).unstack("dataset")[dss]
    t.index.name = "configuração"
    R.append(md(t) + "\n")
    piv = allr.groupby(["dataset", "alvo"])["f1"].mean().unstack("alvo")
    rk = piv.rank(axis=1, ascending=False).mean().sort_values().round(2).to_frame("rank_medio")
    rk.index.name = "configuração"
    R.append("Ranking médio entre datasets (descritivo):\n\n" + md(rk, 2) + "\n")

    # 2. variante primária
    if a.primary in set(dual.alvo):
        R.append(f"## 2. Variante primária `{a.primary}`\n")
        R.append("Diferença = F1(primária) − F1(alvo); positiva favorece a dupla via. Holm entre os datasets, por linha.\n")
        rows = pair_rows(allr, a.primary, vref, ratio, a.margin, f"− {vref}")
        enc, rep = topo_of(a.primary)
        rows += pair_rows(allr, a.primary, f"topológica {enc}/{rep}", ratio, a.margin, f"− topológica {enc}/{rep}")
        for c in controls:
            if c.startswith(a.primary + "/"):
                rows += pair_rows(allr, a.primary, c, ratio, a.margin, f"− controle {c.split('/')[-1]}")
        emit(R, rows, dss, "primária menos")
    else:
        R.append(f"## 2. Variante primária `{a.primary}`: sem resultados\n")

    # 3. todas as variantes contra as vias isoladas
    R.append("## 3. Variantes da dupla via contra as vias isoladas\n")
    R.append(f"Dupla via − `{vref}`:\n")
    emit(R, sum([pair_rows(allr, v, vref, ratio, a.margin, v) for v in variants], []), dss, "dupla − visual")
    R.append("Dupla via − via topológica do mesmo codificador e representação:\n")
    rows = []
    for v in variants:
        enc, rep = topo_of(v)
        rows += pair_rows(allr, v, f"topológica {enc}/{rep}", ratio, a.margin, v)
    emit(R, rows, dss, "dupla − topológica")

    # 4. variantes contra a primária
    if a.primary in set(dual.alvo):
        R.append(f"## 4. Variantes contra a primária `{a.primary}` (análise secundária)\n")
        R.append("Diferença = F1(variante) − F1(primária).\n")
        emit(R, sum([pair_rows(allr, v, a.primary, ratio, a.margin, v) for v in variants if v != a.primary], []),
             dss, "variante − primária")

    # 5. configurações selecionadas e custo
    R.append("## 5. Configurações selecionadas e custo\n")
    for al, ge in dual.groupby("alvo"):
        cfg = pd.DataFrame([json.loads(c) for c in ge.sel_cfg])
        parts = [f"{c}: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg[c].value_counts(normalize=True).round(2).items()))
                 for c in ("lr_bb", "lr_hd") if c in cfg]
        parts += [f"épocas medianas={ge.epochs.median():.0f}", f"parâmetros={ge.n_params.median():.0f}",
                  f"tempo mediano por fold={ge.seconds.median():.0f}s"]
        R.append(f"- **{al}**: " + "; ".join(parts))
    R.append("")

    # 6. holdout
    hold = load_csvs(out / "final" / "holdout")
    if not hold.empty:
        hold["tag"] = hold["tag"].fillna("") if "tag" in hold else ""
        hold = hold[hold.tag == a.tag]
    if not hold.empty:
        R.append("## 6. Holdout (avaliação única; sem dispersão, resultado discreto nos datasets pequenos)\n")
        hs = [x for x in dss if x in set(hold.dataset)]
        for m in ("f1", "acc", "auc"):
            t = hold.pivot_table(index="enc", columns="dataset", values=m, aggfunc="mean")[hs]
            t.index.name = {"f1": "F1 macro", "acc": "acurácia", "auc": "AUC-ROC"}[m]
            R.append(md(t.round(3)) + "\n")
        R.append("Tamanho do holdout: " + ", ".join(
            f"{ds}={int(g.n_test.iloc[0])}" for ds, g in hold.groupby("dataset")) + "\n")

    (out / "final" / "report_final.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
