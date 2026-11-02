#!/usr/bin/env bash
# Pipeline completo da ablação topológica: compila o C++, gera as sequências topológicas
# e roda os estágios enc -> rmax -> hp (escolhendo o melhor enc/rep por ranking médio).
#
# Uso (a partir de ~/tcc, dentro do tmux):
#     bash run_all.sh
# Variáveis opcionais:
#     DATASETS="UCSB CR LG LA NHL"  REPEATS=3  MAX_EPOCHS=200  PATIENCE=20  FORCE=1
# É seguro interromper e rodar de novo: o cache por imagem retoma a geração e os estágios
# já concluídos (results/*_summary.csv) são pulados, a menos que FORCE=1.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DATASETS="${DATASETS:-UCSB CR LG LA NHL}"   # NHL por último: é o mais lento
REPEATS="${REPEATS:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
FORCE="${FORCE:-0}"
RUNNER="${RUNNER:-uv run}"                  # RUNNER=python para ambientes sem uv

mkdir -p logs results
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a logs/run_all.log; }

# ---------------------------------------------------------------- 1. compilar
log "Compilando src/SSNModel.so"
g++ -O3 -shared -fPIC -fopenmp -o src/SSNModel.so src/SSNModel.cpp

# ---------------------------------------------------------------- 2. sequências topológicas
for ds in $DATASETS; do
    # pula só se o .npz existe E cobre todas as imagens (evita reaproveitar um teste com --limit)
    if [[ -f "data/$ds/topological.npz" && "$FORCE" != "1" ]] && $RUNNER - "$ds" <<'EOF2'
import sys, numpy as np
from pathlib import Path
ds = Path("data") / sys.argv[1]
ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
n_img = sum(1 for p in ds.rglob("*") if p.suffix.lower() in ext and ".topo_cache" not in str(p))
n_npz = len(np.load(ds / "topological.npz")["y"])
sys.exit(0 if n_img == n_npz else 1)
EOF2
    then
        log "[$ds] topological.npz completo, pulando (FORCE=1 para refazer)"
        continue
    fi
    log "[$ds] gerando sequências topológicas"
    $RUNNER src/build_topological.py --datasets "$ds" 2>&1 | tee -a "logs/build_${ds}.log"
done

# ---------------------------------------------------------------- 3. ablação
COMMON=(--datasets $DATASETS --repeats "$REPEATS" --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE")

run_stage() {  # run_stage <stage> [args...]
    local stage="$1"; shift
    if [[ -f "results/${stage}_summary.csv" && "$FORCE" != "1" ]]; then
        log "estágio $stage já concluído, pulando (FORCE=1 para refazer)"
        return
    fi
    log "estágio $stage $*"
    $RUNNER src/ablacao_topologica.py --stage "$stage" "$@" "${COMMON[@]}" 2>&1 | tee -a "logs/ablacao_${stage}.log"
}

run_stage enc

# melhor (enc, rep) = menor ranking médio de F1 entre os datasets
read -r BEST_ENC BEST_REP < <($RUNNER - <<'EOF'
import pandas as pd
s = pd.read_csv("results/enc_summary.csv")
s["rank"] = s.groupby("dataset")["f1_mean"].rank(ascending=False)
best = s.groupby(["enc", "rep"])["rank"].mean().sort_values().index[0]
print(best[0], best[1])
EOF
)
log "melhor codificador pelo ranking médio: enc=$BEST_ENC rep=$BEST_REP"

run_stage rmax --enc "$BEST_ENC" --rep "$BEST_REP"
run_stage hp   --enc "$BEST_ENC" --rep "$BEST_REP"

log "Concluído. Resultados em results/ e logs em logs/"
