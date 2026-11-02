#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path

RES_DIR = Path("results")
DATASETS = ["UCSB", "CR", "LG", "LA", "NHL"]

def summarize_stage(stage, group_cols):
    raw_path = RES_DIR / f"{stage}_raw.csv"
    if not raw_path.exists():
        print(f"[!] Arquivo {raw_path} não encontrado.")
        return None

    df = pd.read_csv(raw_path)
    
    # Agrupa por dataset e configuração calculando média e desvio padrão
    agg = df.groupby(["dataset"] + group_cols).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ep_mean=("best_ep", "mean")
    ).reset_index()

    # Calcula o ranking de F1 dentro de cada dataset (1 = melhor)
    agg["rank"] = agg.groupby("dataset")["f1_mean"].rank(ascending=False, method="average")
    
    # Ranking médio entre os 5 datasets
    ranking = agg.groupby(group_cols).agg(
        avg_rank=("rank", "mean"),
        global_f1_mean=("f1_mean", "mean"),
        global_acc_mean=("acc_mean", "mean"),
        global_auc_mean=("auc_mean", "mean")
    ).sort_values(["avg_rank", "global_f1_mean"], ascending=[True, False]).reset_index()

    print(f"\n{'='*70}")
    print(f" ESTÁGIO: {stage.upper()} — Ranking Geral (Menor Rank = Melhor)")
    print(f"{'='*70}")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Formata tabela Média ± Desvio Padrão (em %) de F1 por dataset
    agg["f1_fmt"] = (agg["f1_mean"]*100).round(2).astype(str) + " ± " + (agg["f1_std"]*100).round(2).astype(str)
    pivot = agg.pivot(index=group_cols, columns="dataset", values="f1_fmt")
    pivot = pivot[[d for d in DATASETS if d in pivot.columns]]
    # Ordena o pivot pela mesma ordem do ranking geral
    pivot = pivot.loc[ranking.set_index(group_cols).index]

    print(f"\n--- F1-Macro (%) [Média ± DP] por Dataset ({stage.upper()}) ---")
    print(pivot.to_string())
    
    # Salva versão em LaTeX
    tex_file = RES_DIR / f"tabela_{stage}.tex"
    pivot.to_latex(tex_file, escape=False)
    print(f"[✓] Tabela LaTeX salva em: {tex_file}")
    
    return ranking, agg

if __name__ == "__main__":
    # 1. Ablação do Codificador e Representação (cum vs diff)
    summarize_stage("enc", ["enc", "rep"])
    
    # 2. Ablação do Raio Máximo (rmax)
    summarize_stage("rmax", ["enc", "rep", "rmax"])
    
    # 3. Busca de Hiperparâmetros (Top 10 configurações)
    rk_hp, _ = summarize_stage("hp", ["enc", "rep", "rmax", "hidden", "lr", "wd", "dropout"])