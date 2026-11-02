#!/usr/bin/env python
"""
Análise estatística da ablação robusta (results_v2/folds/*.csv) -> results_v2/report.md,
CSVs e tabela LaTeX.

O que calcula:
  1. F1 macro por (dataset, codificador, representação): média ± desvio entre folds externos.
  2. Ranking médio entre datasets (descritivo; sem teste global entre datasets, que com 5
     datasets tem poder desprezível).
  3. Comparações PAREADAS contra uma referência (padrão: bilstm), fold a fold (as partições
     externas são idênticas entre tarefas), com o teste t corrigido de Nadeau & Bengio
     (a variância é inflada por 1/J + n_teste/n_treino, porque os folds da CV não são
     independentes), IC 95% da diferença e correção de Holm. Marca "equivalente" quando o IC
     cai inteiro em (-margem, +margem).
  4. Ganho do ajuste: F1 com a configuração selecionada vs. a padrão (mede o quanto a v1,
     que só ajustou o MLP, favorecia um codificador).
  5. Estabilidade dos hiperparâmetros selecionados e número de parâmetros.
  6. Sensibilidade a r_max (resolução nativa).
  7. Resolução: via topológica sobre a imagem 256x256 vs nativa, pareado fold a fold (mesmas
     partições), mesmo teste t corrigido, IC 95%, Holm dentro de cada codificador/representação.
  8. Sanidade: rótulos permutados (deve ficar no acaso) e 9. auditoria dos dados (duplicatas).

Os CSVs têm o nº de repetições da CV externa no nome e na coluna `repeats`; a análise usa um
único valor por vez (--repeats; padrão: o maior encontrado), para não misturar configurações.
Arquivos antigos (sem as colunas res/repeats) são lidos como resolução nativa e repeats = folds/5.

Uso:
  python src/analise_ablacao.py [--ref bilstm] [--margin 0.02] [--repeats 3] [--out_dir results_v2]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ENC_ORDER = ["bilstm", "lstm", "gru", "mlp", "cnn1d", "logreg", "svm", "rf", "knn", "dummy"]
DS_ORDER = ["UCSB", "CR", "LA", "LG", "NHL"]


def ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def corrected_ttest(d, ratio, level=0.95):
    """Teste t pareado corrigido (Nadeau & Bengio, 2003). d: diferenças por fold."""
    d = np.asarray(d, float)
    J = len(d)
    m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / J + ratio) * v)
    if se == 0:
        return m, (m, m), (1.0 if m == 0 else 0.0)
    t = m / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    tc = stats.t.ppf(0.5 + level / 2, J - 1)
    return m, (m - tc * se, m + tc * se), p


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


KEY = ["dataset", "enc", "rep", "rmax", "res"]


def load_folds(out, sub="folds"):
    files = sorted((Path(out) / sub).glob("*.csv"))
    files = [f for f in files if not f.name.endswith(".partial.csv")]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"res": str}) for f in files], ignore_index=True)
    if "res" not in df:
        df["res"] = "native"
    df["res"] = df["res"].fillna("native")
    if "repeats" not in df:
        df["repeats"] = np.nan
    n_folds = df.groupby(KEY)["fold"].transform("nunique")  # arquivos antigos: 5 folds por repetição
    df["repeats"] = df["repeats"].fillna(n_folds // 5).astype(int)
    return df.drop_duplicates(KEY + ["repeats", "fold"], keep="last").reset_index(drop=True)


def md(df, floatfmt=3):
    """DataFrame -> tabela markdown (sem depender de tabulate)."""
    d = df.copy()
    cols = [df.index.name or ""] + [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in d.iterrows():
        cells = [str(idx)] + [f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def order(idx, ref_order):
    return sorted(idx, key=lambda x: ref_order.index(x) if x in ref_order else 99)


def audit(data_dir):
    rows = []
    for ds in DS_ORDER:
        p = Path(data_dir) / ds / "topological.npz"
        if not p.exists():
            continue
        d = np.load(p)
        X, y = d["X"].reshape(len(d["y"]), -1), d["y"]
        dup = len(X) - len({r.round(6).tobytes() for r in X})
        rows.append(dict(dataset=ds, N=len(y), classes=np.bincount(y).tolist(),
                         duplicatas_exatas=dup, nan_inf=int((~np.isfinite(X)).sum()),
                         colunas_constantes=int((X.std(0) == 0).sum())))
    return pd.DataFrame(rows).set_index("dataset") if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "results_v2"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--ref", default="bilstm")
    ap.add_argument("--margin", type=float, default=0.02, help="margem de equivalência em F1")
    ap.add_argument("--repeats", type=int, default=0,
                    help="repetições da CV externa a analisar (0 = o maior valor encontrado)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    df = load_folds(out)
    if df.empty:
        raise SystemExit(f"nenhum resultado em {out}/folds")
    avail = sorted(df.repeats.unique())
    rep_sel = a.repeats or int(max(avail))
    if rep_sel not in avail:
        raise SystemExit(f"nenhum resultado com repeats={rep_sel} em {out}/folds (disponíveis: {avail})")
    if len(avail) > 1:
        print(f"[aviso] repeats disponíveis: {avail}; analisando repeats={rep_sel}")
    df = df[df.repeats == rep_sel]
    ratio = float((df["n_test"] / df["n_train"]).mean())
    R = []  # seções do relatório
    R.append(f"# Relatório da ablação robusta\n\nRepetições da CV externa: {rep_sel}; folds externos por tarefa: "
             f"{int(df.groupby(KEY).size().median())} "
             f"(razão teste/treino = {ratio:.2f}). Métrica principal: F1 macro, medido no fold externo "
             f"com a configuração escolhida pela CV interna.\n")

    dn = df[df.res == "native"]       # resolução nativa (todas as análises, exceto a seção 7)
    main_df = dn[dn.rmax == 15]
    dss = order(main_df.dataset.unique(), DS_ORDER)

    # 1. tabela principal
    R.append("## 1. F1 macro (média ± desvio entre folds), r_max = 15\n")
    tab = main_df.groupby(["enc", "rep", "dataset"])["f1"].apply(ms).unstack("dataset")[dss]
    tab = tab.reindex(sorted(tab.index, key=lambda t: (ENC_ORDER.index(t[0]) if t[0] in ENC_ORDER else 99, t[1])))
    tab.index = [f"{e}/{r}" for e, r in tab.index]
    tab.index.name = "codificador/rep"
    R.append(md(tab) + "\n")
    tab.to_csv(out / "tabela_f1.csv")

    # LaTeX
    tex = ["\\begin{table}[htb]", "\\centering",
           "\\caption{F1 macro (média $\\pm$ desvio entre folds externos) da via topológica isolada.}",
           "\\label{tab:ablacao-topologica}", "\\small",
           "\\begin{tabular}{l" + "c" * len(dss) + "}", "\\toprule",
           "\\textbf{Codificador} & " + " & ".join(f"\\textbf{{{d}}}" for d in dss) + " \\\\", "\\midrule"]
    for k, row in tab.iterrows():
        tex.append(k.replace("_", "\\_") + " & " + " & ".join(row[d].replace("±", "$\\pm$") for d in dss) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tabela_ablacao.tex").write_text("\n".join(tex), encoding="utf-8")

    # 2. ranking (descritivo)
    R.append("## 2. Ranking médio entre datasets (1 = melhor F1 médio)\n")
    piv = main_df.groupby(["dataset", "enc", "rep"])["f1"].mean().unstack(["enc", "rep"])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    rk = pd.DataFrame({"rank_medio": ranks.round(2)})
    rk.index = [f"{e}/{r}" for e, r in rk.index]
    rk.index.name = "codificador/rep"
    R.append(md(rk, 2) + "\n")
    rk.to_csv(out / "ranking.csv")

    # 3. pareado contra a referência
    R.append(f"## 3. Comparação pareada contra `{a.ref}` (mesma representação), teste t corrigido\n")
    R.append(f"Diferença = F1(codificador) − F1({a.ref}); positiva favorece o codificador. "
             f"IC 95% com variância corrigida; p ajustado por Holm dentro de cada dataset e "
             f"representação. 'equiv.' = IC inteiro dentro de ±{a.margin}.\n")
    pr = []
    for (ds, rep), g in main_df.groupby(["dataset", "rep"]):
        ref = g[g.enc == a.ref].set_index("fold")
        if ref.empty:
            continue
        fam = []
        for enc, ge in g[g.enc != a.ref].groupby("enc"):
            ge = ge.set_index("fold")
            j = ge.index.intersection(ref.index)
            assert (ge.loc[j, "split_id"] == ref.loc[j, "split_id"]).all(), "partições desalinhadas"
            d = (ge.loc[j, "f1"] - ref.loc[j, "f1"]).values
            m, ci, p = corrected_ttest(d, ratio)
            fam.append(dict(dataset=ds, rep=rep, enc=enc, dif=m, ic_inf=ci[0], ic_sup=ci[1], p=p,
                            equiv=bool(ci[0] > -a.margin and ci[1] < a.margin)))
        if fam:
            adj = holm([f["p"] for f in fam])
            for f, q in zip(fam, adj):
                f["p_holm"] = q
            pr += fam
    if pr:
        prd = pd.DataFrame(pr)
        prd.to_csv(out / "pareado_vs_ref.csv", index=False)
        for rep in ("cum", "diff"):
            s = prd[prd.rep == rep]
            if s.empty:
                continue
            t = s.assign(
                cell=lambda x: x.apply(lambda r: f"{r.dif:+.3f} [{r.ic_inf:+.3f}, {r.ic_sup:+.3f}] "
                                                 f"p={r.p_holm:.2f}{' eq.' if r.equiv else ''}", axis=1)
            ).pivot(index="enc", columns="dataset", values="cell")
            t = t.reindex(order(t.index, ENC_ORDER))[[d for d in dss if d in t.columns]]
            t.index.name = f"vs {a.ref}/{rep}"
            R.append(md(t) + "\n")
        sig = prd[prd.p_holm < 0.05]
        R.append(f"Diferenças significativas após Holm (p<0.05): {len(sig)} de {len(prd)}.\n")
        if len(sig):
            R.append(md(sig.set_index(["dataset", "rep", "enc"])[["dif", "p_holm"]].round(3)) + "\n")

    # 4. ganho do ajuste
    R.append("## 4. Ganho do ajuste de hiperparâmetros (F1 selecionado − F1 padrão, nos mesmos folds)\n")
    g = main_df.assign(ganho=main_df.f1 - main_df.d_f1)
    gt = g.groupby(["enc", "rep", "dataset"])["ganho"].mean().unstack("dataset")[dss]
    gt = gt.reindex(sorted(gt.index, key=lambda t: (ENC_ORDER.index(t[0]) if t[0] in ENC_ORDER else 99, t[1])))
    gt.index = [f"{e}/{r}" for e, r in gt.index]
    gt.index.name = "codificador/rep"
    R.append(md(gt.round(3)) + "\n\nGanhos próximos de zero ou negativos indicam que a busca não "
             "encontra melhora real (a seleção interna é ruidosa em datasets pequenos).\n")
    gt.to_csv(out / "ganho_ajuste.csv")

    # 5. hiperparâmetros selecionados e parâmetros
    R.append("## 5. Configurações selecionadas (todos os folds, datasets e representações)\n")
    for enc, ge in main_df[main_df.enc.isin(["bilstm", "lstm", "gru", "mlp", "cnn1d"])].groupby("enc"):
        cfg = pd.DataFrame([json.loads(c) for c in ge.sel_cfg])
        parts = []
        for col in ("emb", "dropout", "bs", "layers", "depth", "kernel"):
            if col in cfg:
                vc = cfg[col].value_counts(normalize=True).round(2).to_dict()
                parts.append(f"{col}: " + ", ".join(f"{k}={v}" for k, v in sorted(vc.items())))
        parts.append(f"lr mediano={cfg['lr'].median():.1e}")
        parts.append(f"config padrão escolhida em {(ge.sel_idx == 0).mean():.0%} dos folds")
        parts.append(f"épocas medianas={ge.epochs.median():.0f}")
        parts.append(f"parâmetros medianos={ge.n_params.median():.0f}")
        R.append(f"- **{enc}**: " + "; ".join(parts))
    R.append("")

    # 6. r_max
    multi = dn.groupby(["enc", "rep"]).rmax.nunique()
    multi = multi[multi > 1]
    if len(multi):
        R.append("## 6. Sensibilidade a r_max (F1 médio ± desvio; p do teste pareado corrigido vs r_max=15)\n")
        for (enc, rep) in multi.index:
            g = dn[(dn.enc == enc) & (dn.rep == rep)]
            cells = {}
            for (ds, rm), gg in g.groupby(["dataset", "rmax"]):
                g15 = g[(g.dataset == ds) & (g.rmax == 15)].set_index("fold")
                txt = ms(gg.f1)
                if rm != 15 and not g15.empty:
                    ge = gg.set_index("fold")
                    j = ge.index.intersection(g15.index)
                    _, _, p = corrected_ttest((ge.loc[j, "f1"] - g15.loc[j, "f1"]).values, ratio)
                    txt += f" (p={p:.2f})"
                cells[(rm, ds)] = txt
            sr = pd.Series(cells)
            sr.index = pd.MultiIndex.from_tuples(sr.index, names=["rmax", "dataset"])
            t = sr.unstack("dataset")
            t = t[[d for d in dss if d in t.columns]]
            t.index.name = f"{enc}/{rep}: r_max"
            R.append(md(t) + "\n")

    # 7. resolução: nativa vs 256x256 (mesmas partições externas)
    rs = df[df.rmax == 15]
    if (rs.res == "256").any():
        rr = []
        for (enc, rep, ds), g in rs.groupby(["enc", "rep", "dataset"]):
            nat = g[g.res == "native"].set_index("fold")
            lo = g[g.res == "256"].set_index("fold")
            j = nat.index.intersection(lo.index)
            if len(j) < 3:
                continue
            assert (nat.loc[j, "split_id"] == lo.loc[j, "split_id"]).all(), "partições desalinhadas (resolução)"
            d = (lo.loc[j, "f1"] - nat.loc[j, "f1"]).values
            m, ci, p = corrected_ttest(d, ratio)
            rr.append(dict(enc=enc, rep=rep, dataset=ds, f1_nativa=nat.loc[j, "f1"].mean(),
                           f1_256=lo.loc[j, "f1"].mean(), dif=m, ic_inf=ci[0], ic_sup=ci[1], p=p,
                           equiv=bool(ci[0] > -a.margin and ci[1] < a.margin)))
        if rr:
            rrd = pd.DataFrame(rr)
            rrd["p_holm"] = np.nan
            for _, idx in rrd.groupby(["enc", "rep"]).groups.items():
                rrd.loc[idx, "p_holm"] = holm(rrd.loc[idx, "p"].values)
            rrd.to_csv(out / "resolucao.csv", index=False)
            R.append("## 7. Resolução: nativa vs 256x256 (via topológica isolada, r_max = 15)\n")
            R.append(f"Cada célula: F1 nativa → F1 256x256 (Δ = F1(256) − F1(nativa), IC 95% corrigido, "
                     f"p ajustado por Holm entre os datasets do mesmo codificador/representação; "
                     f"'eq.' = IC inteiro dentro de ±{a.margin}). Δ negativo: reduzir a imagem piora.\n")
            cell = rrd.assign(cell=lambda x: x.apply(
                lambda r: f"{r.f1_nativa:.3f} → {r.f1_256:.3f} ({r.dif:+.3f} [{r.ic_inf:+.3f}, {r.ic_sup:+.3f}] "
                          f"p={r.p_holm:.2f}{' eq.' if r.equiv else ''})", axis=1))
            cell["linha"] = cell.enc + "/" + cell.rep
            t = cell.pivot(index="linha", columns="dataset", values="cell")
            t = t.reindex(sorted(t.index, key=lambda s: (ENC_ORDER.index(s.split("/")[0]), s.split("/")[1])))
            t = t[[d for d in dss if d in t.columns]]
            t.index.name = "codificador/rep"
            R.append(md(t) + "\n")
            sm = rrd.groupby(["enc", "rep"]).agg(dif_media=("dif", "mean"),
                                                 datasets_256_melhor=("dif", lambda s: int((s > 0).sum())),
                                                 datasets=("dif", "size"),
                                                 significativos_holm=("p_holm", lambda s: int((s < 0.05).sum())))
            sm.index = [f"{e}/{r}" for e, r in sm.index]
            sm.index.name = "codificador/rep"
            R.append(md(sm.round({"dif_media": 3}).astype(object)) + "\n")  # object: mantém inteiros como inteiros
            nsig = int((rrd.p_holm < 0.05).sum())
            R.append(f"Diferenças significativas após Holm (p<0.05): {nsig} de {len(rrd)}. "
                     f"Com folds pequenos o IC costuma ser largo: Δ pequeno com IC largo é inconclusivo, "
                     f"não prova ausência de efeito; só 'eq.' indica equivalência na margem escolhida.\n")

    # 8. sanidade
    perm = load_folds(out, "perm")
    if not perm.empty:
        R.append("## 8. Sanidade: rótulos permutados (deve ficar no acaso)\n")
        pt = perm.groupby(["dataset", "enc"]).agg(f1=("f1", "mean"), acc=("acc", "mean"),
                                                  classes=("n_classes", "first")).round(3)
        pt["acaso(1/C)"] = (1 / pt["classes"]).round(3)
        R.append(md(pt) + "\n\nSe o F1 com rótulos permutados ficar bem acima do acaso, há vazamento "
                 "ou erro no protocolo.\n")
    au = audit(a.data_dir)
    if not au.empty:
        R.append("## 9. Auditoria dos dados\n")
        R.append(md(au) + "\n\nDuplicatas exatas entre as sequências topológicas sugerem imagens "
                 "repetidas, que inflariam a CV.\n")

    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print(f"\n[salvo em {out}/report.md, tabela_ablacao.tex e CSVs]")


if __name__ == "__main__":
    main()
