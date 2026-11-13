import pandas as pd
import glob
from pathlib import Path

def fmt_dec(v, digits=3):
    if pd.isna(v):
        return "-"
    return f"{v:.{digits}f}".replace(".", "{,}")

def fmt_sci(v):
    if v >= 0.01:
        return f"{v:.2f}".replace(".", "{,}")
    s = f"{v:.0e}"
    base, exp = s.split("e")
    exp = int(exp)
    if base == "1":
        return f"10^{{{exp}}}"
    return f"{base} \\cdot 10^{{{exp}}}"

# ------------------------------------------------------------------
# 1. Apêndice A: Via Topológica
# ------------------------------------------------------------------
df_top = pd.read_csv("results_topologica_v3/summary_best.csv")
model_names_top = {
    "lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU",
    "bigru": "BiGRU", "saclstm": "SAC-LSTM", "mlp": "MLP"
}

lines_top = []
lines_top.append(r"\chapter{Resultados Completos da Validação Cruzada da Via Topológica}")
lines_top.append(r"\label{apendice:topologica}")
lines_top.append("")
lines_top.append(r"Os desempenhos obtidos pelos codificadores da via topológica durante o procedimento de validação cruzada estratificada em cinco partições estão reunidos neste apêndice. Para cada conjunto de dados e para cada modelo avaliado, são detalhados os hiperparâmetros ótimos selecionados ($\eta, p, h, B$), a média de épocas executadas até a ativação da parada antecipada, a acurácia média e a pontuação de F1 macro acompanhada do desvio padrão.")
lines_top.append("")
lines_top.append(r"\begin{table}[H]")
lines_top.append(r"\centering")
lines_top.append(r"\caption{Desempenho detalhado por validação cruzada (5 partições) e configurações ótimas da via topológica.}")
lines_top.append(r"\label{tab:apendice-top}")
lines_top.append(r"\footnotesize")
lines_top.append(r"\resizebox{\textwidth}{!}{%")
lines_top.append(r"\begin{tabular}{llccccccc}")
lines_top.append(r"\toprule")
lines_top.append(r"\textbf{\textit{Dataset}} & \textbf{Codificador} & $\eta$ & $p$ & $h$ & $B$ & \textbf{Épocas} & \textbf{Acurácia} & \textbf{F1 macro} \\")
lines_top.append(r"\midrule")

current_ds = None
for _, row in df_top.sort_values(by=["dataset", "model"]).iterrows():
    ds = row["dataset"]
    if current_ds is not None and ds != current_ds:
        lines_top.append(r"\midrule")
    current_ds = ds
    m_name = model_names_top.get(row["model"], row["model"])
    lr_val = float(row["lr"])
    lr_str = f"${fmt_sci(lr_val)}$"
    p_str = fmt_dec(row["dropout"], 1)
    h_str = str(int(row["h"]))
    b_str = str(int(row["bs"]))
    ep_str = fmt_dec(row["ep_mean"], 1)
    acc_str = fmt_dec(row["acc_mean"], 3)
    f1_str = f"{fmt_dec(row['f1_mean'], 3)} $\\pm$ {fmt_dec(row['f1_std'], 3)}"
    lines_top.append(f"{ds} & {m_name} & {lr_str} & {p_str} & {h_str} & {b_str} & {ep_str} & {acc_str} & {f1_str} \\\\")

lines_top.append(r"\bottomrule")
lines_top.append(r"\end{tabular}%")
lines_top.append(r"}")
lines_top.append(r"\end{table}")
lines_top.append("")

with open("tex/apendice_topologica.tex", "w") as f:
    f.write("\n".join(lines_top))
print("Wrote tex/apendice_topologica.tex")


# ------------------------------------------------------------------
# 2. Apêndice B: Via Visual
# ------------------------------------------------------------------
df_vis = pd.read_csv("results_visual/summary_best.csv")
vis_info = {
    "resnet50": ("Convolucional", "ResNet-50"),
    "efficientnet_b0": ("Convolucional", "EfficientNet-B0"),
    "vit_tiny": (r"\textit{Transformer}", "ViT-Tiny"),
    "deit_tiny": (r"\textit{Transformer}", "DeiT-Tiny"),
    "mobilevit_s": ("Híbrida", "MobileViT-S"),
    "coatnet_0": ("Híbrida", "CoAtNet-0"),
}

lines_vis = []
lines_vis.append(r"\chapter{Resultados Completos da Validação Cruzada da Via Visual}")
lines_vis.append(r"\label{apendice:visual}")
lines_vis.append("")
lines_vis.append(r"Os desempenhos obtidos pelas arquiteturas de visão computacional na via visual durante a busca de hiperparâmetros sobre as cinco partições de validação cruzada estão reunidos neste apêndice. Para cada modelo e conjunto de dados, são reportadas as taxas de aprendizado do \textit{backbone} ($\eta_{bb}$) e da cabeça linear ($\eta_{hd}$), a taxa de \textit{dropout} ($p$), o tamanho do lote ($B$), a média de épocas até a convergência, a acurácia média e a pontuação de F1 macro.")
lines_vis.append("")
lines_vis.append(r"\begin{table}[H]")
lines_vis.append(r"\centering")
lines_vis.append(r"\caption{Desempenho detalhado por validação cruzada (5 partições) e configurações ótimas da via visual.}")
lines_vis.append(r"\label{tab:apendice-vis}")
lines_vis.append(r"\small")
lines_vis.append(r"\resizebox{\textwidth}{!}{%")
lines_vis.append(r"\begin{tabular}{lllccccccc}")
lines_vis.append(r"\toprule")
lines_vis.append(r"\textbf{\textit{Dataset}} & \textbf{Família} & \textbf{Modelo} & $\eta_{bb}$ & $\eta_{hd}$ & $p$ & $B$ & \textbf{Épocas} & \textbf{Acurácia} & \textbf{F1 macro} \\")
lines_vis.append(r"\midrule")

current_ds = None
for _, row in df_vis.sort_values(by=["dataset", "model"]).iterrows():
    ds = row["dataset"]
    if current_ds is not None and ds != current_ds:
        lines_vis.append(r"\midrule")
    current_ds = ds
    fam, m_label = vis_info.get(row["model"], ("Outra", row["model"]))
    lr_bb_str = f"${fmt_sci(float(row['lr_bb']))}$"
    lr_hd_str = f"${fmt_sci(float(row['lr_hd']))}$"
    p_str = fmt_dec(row["dropout"], 1)
    b_str = str(int(row["bs"]))
    ep_str = fmt_dec(row["ep_mean"], 1)
    acc_str = fmt_dec(row["acc_mean"], 3)
    f1_str = f"{fmt_dec(row['f1_mean'], 3)} $\\pm$ {fmt_dec(row['f1_std'], 3)}"
    lines_vis.append(f"{ds} & {fam} & {m_label} & {lr_bb_str} & {lr_hd_str} & {p_str} & {b_str} & {ep_str} & {acc_str} & {f1_str} \\\\")

lines_vis.append(r"\bottomrule")
lines_vis.append(r"\end{tabular}%")
lines_vis.append(r"}")
lines_vis.append(r"\end{table}")
lines_vis.append("")

with open("tex/apendice_visual.tex", "w") as f:
    f.write("\n".join(lines_vis))
print("Wrote tex/apendice_visual.tex")


# ------------------------------------------------------------------
# 3. Apêndice C: Teste Independente por Semente
# ------------------------------------------------------------------
seed_files = sorted(glob.glob("results_dupla_via/seeds/*.csv"))

lines_hold = []
lines_hold.append(r"\chapter{Desempenho por Semente na Partição de Teste Independente}")
lines_hold.append(r"\label{apendice:holdout}")
lines_hold.append("")
lines_hold.append(r"O desempenho alcançado em cada uma das cinco sementes pré-fixadas (42 a 46) avaliadas sobre a partição de teste independente (20\% dos dados) está registrado neste apêndice. Os valores individuais de F1 macro, acurácia e área sob a curva ROC (AUC-ROC) são discriminados para a via topológica isolada, a via visual isolada e a rede unificada de dupla via em todos os cinco conjuntos de dados analisados.")
lines_hold.append("")
lines_hold.append(r"\begin{table}[H]")
lines_hold.append(r"\centering")
lines_hold.append(r"\caption{Resultados detalhados por semente e médias consolidadas no teste independente (20\% dos dados).}")
lines_hold.append(r"\label{tab:apendice-holdout}")
lines_hold.append(r"\small")
lines_hold.append(r"\resizebox{\textwidth}{!}{%")
lines_hold.append(r"\begin{tabular}{llccccccccc}")
lines_hold.append(r"\toprule")
lines_hold.append(r"\textbf{\textit{Dataset}} & \textbf{Semente} & \multicolumn{3}{c}{\textbf{Via Topológica Isolada}} & \multicolumn{3}{c}{\textbf{Via Visual Isolada}} & \multicolumn{3}{c}{\textbf{Rede de Dupla Via}} \\")
lines_hold.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}")
lines_hold.append(r"& & \textbf{F1} & \textbf{Acc} & \textbf{AUC} & \textbf{F1} & \textbf{Acc} & \textbf{AUC} & \textbf{F1} & \textbf{Acc} & \textbf{AUC} \\")
lines_hold.append(r"\midrule")

for i, sf in enumerate(seed_files):
    df_s = pd.read_csv(sf)
    ds = df_s["dataset"].iloc[0]
    if i > 0:
        lines_hold.append(r"\midrule")
    
    seeds = sorted(df_s["seed"].unique())
    for s in seeds:
        sub = df_s[df_s["seed"] == s].set_index("approach")
        
        top_f1 = fmt_dec(sub.loc["top", "f1"], 3) if "top" in sub.index else "-"
        top_acc = fmt_dec(sub.loc["top", "acc"], 3) if "top" in sub.index else "-"
        top_auc = fmt_dec(sub.loc["top", "auc"], 3) if "top" in sub.index else "-"
        
        vis_f1 = fmt_dec(sub.loc["vis", "f1"], 3) if "vis" in sub.index else "-"
        vis_acc = fmt_dec(sub.loc["vis", "acc"], 3) if "vis" in sub.index else "-"
        vis_auc = fmt_dec(sub.loc["vis", "auc"], 3) if "vis" in sub.index else "-"
        
        dup_f1 = fmt_dec(sub.loc["dupla", "f1"], 3) if "dupla" in sub.index else "-"
        dup_acc = fmt_dec(sub.loc["dupla", "acc"], 3) if "dupla" in sub.index else "-"
        dup_auc = fmt_dec(sub.loc["dupla", "auc"], 3) if "dupla" in sub.index else "-"
        
        lines_hold.append(f"{ds} & {s} & {top_f1} & {top_acc} & {top_auc} & {vis_f1} & {vis_acc} & {vis_auc} & {dup_f1} & {dup_acc} & {dup_auc} \\\\")
    
    # Compute and add Mean row
    sub_top = df_s[df_s["approach"] == "top"]
    sub_vis = df_s[df_s["approach"] == "vis"]
    sub_dup = df_s[df_s["approach"] == "dupla"]
    
    m_top_f1 = f"{fmt_dec(sub_top['f1'].mean(), 3)} $\\pm$ {fmt_dec(sub_top['f1'].std(), 3)}"
    m_top_acc = f"{fmt_dec(sub_top['acc'].mean(), 3)} $\\pm$ {fmt_dec(sub_top['acc'].std(), 3)}"
    m_top_auc = f"{fmt_dec(sub_top['auc'].mean(), 3)} $\\pm$ {fmt_dec(sub_top['auc'].std(), 3)}"
    
    m_vis_f1 = f"{fmt_dec(sub_vis['f1'].mean(), 3)} $\\pm$ {fmt_dec(sub_vis['f1'].std(), 3)}"
    m_vis_acc = f"{fmt_dec(sub_vis['acc'].mean(), 3)} $\\pm$ {fmt_dec(sub_vis['acc'].std(), 3)}"
    m_vis_auc = f"{fmt_dec(sub_vis['auc'].mean(), 3)} $\\pm$ {fmt_dec(sub_vis['auc'].std(), 3)}"
    
    m_dup_f1 = f"{fmt_dec(sub_dup['f1'].mean(), 3)} $\\pm$ {fmt_dec(sub_dup['f1'].std(), 3)}"
    m_dup_acc = f"{fmt_dec(sub_dup['acc'].mean(), 3)} $\\pm$ {fmt_dec(sub_dup['acc'].std(), 3)}"
    m_dup_auc = f"{fmt_dec(sub_dup['auc'].mean(), 3)} $\\pm$ {fmt_dec(sub_dup['auc'].std(), 3)}"
    
    lines_hold.append(r"\cmidrule(lr){2-11}")
    lines_hold.append(f"& \\textbf{{Média}} & \\textbf{{{m_top_f1}}} & \\textbf{{{m_top_acc}}} & \\textbf{{{m_top_auc}}} & \\textbf{{{m_vis_f1}}} & \\textbf{{{m_vis_acc}}} & \\textbf{{{m_vis_auc}}} & \\textbf{{{m_dup_f1}}} & \\textbf{{{m_dup_acc}}} & \\textbf{{{m_dup_auc}}} \\\\")

lines_hold.append(r"\bottomrule")
lines_hold.append(r"\end{tabular}%")
lines_hold.append(r"}")
lines_hold.append(r"\end{table}")
lines_hold.append("")

with open("tex/apendice_holdout.tex", "w") as f:
    f.write("\n".join(lines_hold))
print("Wrote tex/apendice_holdout.tex")
