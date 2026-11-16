#!/usr/bin/env python
"""
Geração e Análise de Matrizes de Confusão: Via Topológica vs. Via Visual vs. Dupla Via.
Protocolo Oficial do TCC (Holdout 20%, 5 sementes consecutivas: 42 a 46).

Este script avalia os conjuntos em que houve divergência de desempenho entre as vias
(foco principal no NHL e LA), salva os arquivos de predição (.npz) e gera as matrizes
de confusão individuais e agregadas, além das métricas detalhadas por classe
(Precisão, Recall e F1-Score).

Uso no servidor (Jupyter Lab Terminal):
    uv run src/gerar_matrizes_confusao.py --dataset NHL --device cuda
    uv run src/gerar_matrizes_confusao.py --datasets NHL LA --device cuda --amp
    python src/gerar_matrizes_confusao.py --dataset NHL --device cuda
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

# Importa a infraestrutura existente do treino oficial
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from treino_dupla_via import (
    HOLDOUT_SEEDS,
    MODEL_INFO,
    N_FEAT,
    N_RAIOS,
    SEED,
    WD,
    IsolatedTopologicalNet,
    IsolatedVisualNet,
    TwoStreamNet,
    augment,
    compute_metrics,
    dev_split,
    get_top_champion,
    get_vis_champion,
    load_topo,
    load_visual,
    standardize,
)

CLASS_NAMES = {
    "NHL": ["CLL", "FL", "MCL"],
    "LA": ["1_Mes", "2_Meses", "3_Meses", "4_Meses"],
    "UCSB": ["Benigno", "Maligno"],
    "CR": ["Benigno", "Maligno"],
    "LG": ["Femea", "Macho"],
}


def train_eval_capture_top(top_cfg, Xtr_top, ytr, Xte_top, yte, C, seed, epochs, device):
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

    m = compute_metrics(yte, preds, probs, C)
    return preds, probs, m


def train_eval_capture_vis(vis_cfg, Xtr_vis, ytr, Xte_vis, yte, C, seed, epochs, device, amp, no_pretrained):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = IsolatedVisualNet(vis_cfg["model"], C, vis_cfg["dropout"], pretrained=not no_pretrained).to(dev)

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
            bv = Xte_vis[i:i + bs].to(dev)
            logits = model(bv, amp=amp)
            probs_list.append(F.softmax(logits, dim=-1).cpu())
    probs = torch.cat(probs_list).numpy()
    preds = probs.argmax(axis=1)

    m = compute_metrics(yte, preds, probs, C)
    return preds, probs, m


def train_eval_capture_dupla(top_cfg, vis_cfg, Xtr_vis, Xtr_top, ytr, Xte_vis, Xte_top, yte, C, seed, epochs, device, amp, no_pretrained):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = TwoStreamNet(
        top_cfg["model"], top_cfg["h"], vis_cfg["model"], C, vis_cfg["dropout"], pretrained=not no_pretrained
    ).to(dev)

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
            xb = Xtr_vis[b].to(dev)
            xb = augment(xb)
            yb = ytr_t[b]

            opt.zero_grad()
            logits = model(xb_vis := xb, xb_top := xtr_top_t[b], amp=amp)
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

    m = compute_metrics(yte, preds, probs, C)
    return preds, probs, m


def format_latex_matrix(cm, class_names, caption, label):
    C = len(class_names)
    cols = "l" + "c" * C + "cc"
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{" + str(C) + r"}{c}{\textbf{Classe Predita}} & & \\")
    lines.append(r"\cmidrule(lr){2-" + str(C + 1) + r"}")
    header = r"\textbf{Classe Real} & " + " & ".join([f"\\textbf{{{c}}}" for c in class_names]) + r" & \textbf{Total} & \textbf{Recall} \\"
    lines.append(header)
    lines.append(r"\midrule")

    for i in range(C):
        row_vals = " & ".join([str(cm[i, j]) for j in range(C)])
        total_row = int(cm[i].sum())
        rec = (cm[i, i] / total_row * 100) if total_row > 0 else 0.0
        lines.append(f"\\textbf{{{class_names[i]}}} & {row_vals} & {total_row} & {rec:.1f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def run_confusion_analysis(ds, a):
    print("\n" + "=" * 90)
    print(f"ANÁLISE DE MATRIZES DE CONFUSÃO: DATASET {ds}")
    print("=" * 90)

    top_cfg = get_top_champion(ds, a.top_dir)
    vis_cfg = get_vis_champion(ds, a.vis_dir)

    print(f"Campeão Topológico: {top_cfg['model'].upper()} (h={top_cfg['h']}, lr={top_cfg['lr']}, bs={top_cfg['bs']})")
    print(f"Campeão Visual:     {MODEL_INFO[vis_cfg['model']]['label']} (lr_bb={vis_cfg['lr_bb']}, lr_hd={vis_cfg['lr_hd']}, bs={vis_cfg['bs']})")

    # Carregamento de dados com fallback automático para múltiplos locais
    data_candidates = [Path(a.data_dir), ROOT / "data", Path.home() / "tcc/data", Path.home() / "ic/data"]
    data_dir_valid = None
    for cand in data_candidates:
        if (cand / ds / "topological.npz").exists():
            data_dir_valid = str(cand)
            break

    if data_dir_valid is None:
        raise FileNotFoundError(f"Dados do dataset {ds} não encontrados em nenhum dos locais: {data_candidates}")

    print(f"Diretório de dados ativo: {data_dir_valid}")
    X_top, y_top = load_topo(ds, data_dir_valid, dummy=a.dummy)
    X_vis_raw, y_vis = load_visual(ds, data_dir_valid, dummy=a.dummy)

    y = y_top
    C = len(np.unique(y))
    class_names = CLASS_NAMES.get(ds, [f"Classe {i}" for i in range(C)])
    idx_dev = dev_split(y)
    idx_te = np.setdiff1d(np.arange(len(y)), idx_dev)

    print(f"Divisão: Desenvolvimento={len(idx_dev)} (80%), Teste={len(idx_te)} (20%), Classes={C} ({', '.join(class_names)})")

    Xtr_top, Xte_top = standardize(X_top[idx_dev], X_top[idx_te])
    X_vis = torch.from_numpy(X_vis_raw).permute(0, 3, 1, 2).contiguous()
    Xtr_vis = X_vis[idx_dev]
    Xte_vis = X_vis[idx_te]
    ytr, yte = y[idx_dev], y[idx_te]

    epochs_top = max(1, int(round(top_cfg["ep_mean"])))
    epochs_vis = max(1, int(round(vis_cfg["ep_mean"])))
    epochs_dupla = max(epochs_vis, 1)

    out_probs = Path(a.out_dir) / "probs"
    out_probs.mkdir(parents=True, exist_ok=True)

    seeds = range(HOLDOUT_SEEDS)
    cm_top_total = np.zeros((C, C), dtype=int)
    cm_vis_total = np.zeros((C, C), dtype=int)
    cm_dup_total = np.zeros((C, C), dtype=int)

    all_preds_top, all_preds_vis, all_preds_dup = [], [], []

    for s in seeds:
        seed = SEED + s
        t0 = time.time()
        print(f"\n---> Rodada {s + 1}/{len(seeds)} (Semente {seed}):")

        # 1. Topológica Isolada
        p_top, pr_top, m_top = train_eval_capture_top(top_cfg, Xtr_top, ytr, Xte_top, yte, C, seed, epochs_top, a.device)
        cm_top = confusion_matrix(yte, p_top, labels=range(C))
        cm_top_total += cm_top
        all_preds_top.append(p_top)
        np.savez(out_probs / f"{ds}__top__seed{seed}.npz", idx=idx_te, y_true=yte, y_pred=p_top, proba=pr_top)
        print(f"  [Topológica] F1={m_top['f1']:.4f}  Acc={m_top['acc']:.4f}  Erros: {(yte != p_top).sum()}/{len(yte)}")

        # 2. Visual Isolada
        p_vis, pr_vis, m_vis = train_eval_capture_vis(
            vis_cfg, Xtr_vis, ytr, Xte_vis, yte, C, seed, epochs_vis, a.device, a.amp, a.no_pretrained
        )
        cm_vis = confusion_matrix(yte, p_vis, labels=range(C))
        cm_vis_total += cm_vis
        all_preds_vis.append(p_vis)
        np.savez(out_probs / f"{ds}__vis__seed{seed}.npz", idx=idx_te, y_true=yte, y_pred=p_vis, proba=pr_vis)
        print(f"  [Visual]     F1={m_vis['f1']:.4f}  Acc={m_vis['acc']:.4f}  Erros: {(yte != p_vis).sum()}/{len(yte)}")

        # 3. Rede de Dupla Via
        p_dup, pr_dup, m_dup = train_eval_capture_dupla(
            top_cfg, vis_cfg, Xtr_vis, Xtr_top, ytr, Xte_vis, Xte_top, yte, C, seed, epochs_dupla,
            a.device, a.amp, a.no_pretrained
        )
        cm_dup = confusion_matrix(yte, p_dup, labels=range(C))
        cm_dup_total += cm_dup
        all_preds_dup.append(p_dup)
        np.savez(out_probs / f"{ds}__dupla__seed{seed}.npz", idx=idx_te, y_true=yte, y_pred=p_dup, proba=pr_dup)
        print(f"  [Dupla Via]  F1={m_dup['f1']:.4f}  Acc={m_dup['acc']:.4f}  Erros: {(yte != p_dup).sum()}/{len(yte)} ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 90)
    print(f"MATRIZES DE CONFUSÃO AGREGADAS (5 SEMENTES, TOTAL = {len(yte) * len(seeds)} PREDIÇÕES)")
    print("=" * 90)

    print("\n[1] VIA TOPOLÓGICA ISOLADA:")
    print(cm_top_total)

    print("\n[2] VIA VISUAL ISOLADA (CoAtNet-0):")
    print(cm_vis_total)

    print("\n[3] REDE DE DUPLA VIA (CoAtNet-0 + Codificador Topológico):")
    print(cm_dup_total)

    # Métricas detalhadas por classe agregadas
    y_true_all = np.tile(yte, len(seeds))
    y_pred_top_all = np.concatenate(all_preds_top)
    y_pred_vis_all = np.concatenate(all_preds_vis)
    y_pred_dup_all = np.concatenate(all_preds_dup)

    p_top_c, r_top_c, f_top_c, _ = precision_recall_fscore_support(y_true_all, y_pred_top_all, labels=range(C), zero_division=0)
    p_vis_c, r_vis_c, f_vis_c, _ = precision_recall_fscore_support(y_true_all, y_pred_vis_all, labels=range(C), zero_division=0)
    p_dup_c, r_dup_c, f_dup_c, _ = precision_recall_fscore_support(y_true_all, y_pred_dup_all, labels=range(C), zero_division=0)

    df_per_class = []
    for i, cname in enumerate(class_names):
        df_per_class.append({
            "Classe": cname,
            "Recall_Top": r_top_c[i],
            "Recall_Vis": r_vis_c[i],
            "Recall_Dupla": r_dup_c[i],
            "Prec_Top": p_top_c[i],
            "Prec_Vis": p_vis_c[i],
            "Prec_Dupla": p_dup_c[i],
            "F1_Top": f_top_c[i],
            "F1_Vis": f_vis_c[i],
            "F1_Dupla": f_dup_c[i],
        })
    df_cls = pd.DataFrame(df_per_class)

    print("\nDESEMPENHO DETALHADO POR CLASSE (Média agregada ao longo de 5 sementes):")
    print(df_cls.to_string(index=False))

    # Exporta tabelas LaTeX prontas para o capítulo
    caption_vis = f"Matriz de confusão agregada ao longo das 5 sementes para a via visual isolada ({MODEL_INFO[vis_cfg['model']]['label']}) no conjunto {ds}."
    caption_dup = f"Matriz de confusão agregada ao longo das 5 sementes para a arquitetura unificada de dupla via no conjunto {ds}."

    tex_vis = format_latex_matrix(cm_vis_total, class_names, caption_vis, f"tab:cm-vis-{ds.lower()}")
    tex_dup = format_latex_matrix(cm_dup_total, class_names, caption_dup, f"tab:cm-dupla-{ds.lower()}")

    out_tex = Path(a.out_dir) / f"matriz_confusao_{ds}.tex"
    with open(out_tex, "w") as f:
        f.write("% Matrizes de Confusao geradas por gerar_matrizes_confusao.py\n\n")
        f.write(tex_vis)
        f.write("\n\n")
        f.write(tex_dup)
        f.write("\n")
    print(f"\nTabelas LaTeX salvas em: {out_tex}")

    # Salva relatório em Markdown
    out_md = Path(a.out_dir) / f"relatorio_classes_{ds}.md"
    with open(out_md, "w") as f:
        f.write(f"# Análise de Classes e Matrizes de Confusão: {ds}\n\n")
        f.write("## 1. Matriz de Confusão Agregada - Via Visual Isolada\n\n```\n")
        f.write(str(cm_vis_total) + "\n```\n\n")
        f.write("## 2. Matriz de Confusão Agregada - Rede de Dupla Via\n\n```\n")
        f.write(str(cm_dup_total) + "\n```\n\n")
        f.write(df_cls.to_string(index=False) + "\n")
    print(f"Relatório Markdown salvo em: {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Geração de Matrizes de Confusão e Métricas por Classe")
    ap.add_argument("--dataset", default=None, choices=["NHL", "LA", "UCSB", "CR", "LG"], help="Executa um dataset específico")
    ap.add_argument("--datasets", nargs="+", default=["NHL", "LA"], help="Lista de datasets (padrão: NHL e LA)")
    ap.add_argument("--data_dir", default=str(ROOT / "data"), help="Diretório de dados (topological.npz e visual_256.npz)")
    ap.add_argument("--top_dir", default=str(ROOT / "results_topologica_v3"), help="Diretório de resultados da via topológica")
    ap.add_argument("--vis_dir", default=str(ROOT / "results_visual"), help="Diretório de resultados da via visual")
    ap.add_argument("--out_dir", default=str(ROOT / "results_dupla_via"), help="Diretório de saída")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Dispositivo PyTorch")
    ap.add_argument("--amp", action="store_true", help="Precisão mista bfloat16/float16")
    ap.add_argument("--dummy", action="store_true", help="Dados sintéticos para teste")
    ap.add_argument("--no_pretrained", action="store_true", help="Sem pesos pré-treinados")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Aviso: CUDA não disponível. Alternando automaticamente para CPU.")
        args.device = "cpu"

    target_datasets = [args.dataset] if args.dataset else args.datasets
    for ds in target_datasets:
        run_confusion_analysis(ds, args)


if __name__ == "__main__":
    main()
