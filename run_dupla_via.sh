#!/usr/bin/env bash
# ==============================================================================
# Treinamento e Avaliação da Rede de Dupla Via (Two-Stream Network)
# Protocolo Oficial (Abordagem 1 - Resultados e Discussão)
# ==============================================================================
# Executa a comparação pareada no conjunto de teste independente (20% holdout, 5 sementes):
#   1. Via Topológica Isolada (campeão da ablação topológica)
#   2. Via Visual Isolada (campeão da ablação visual)
#   3. Rede de Dupla Via (Two-Stream Network)
#
# Uso:
#   bash run_dupla_via.sh
#   DATASETS="UCSB CR" bash run_dupla_via.sh
#   AMP=1 bash run_dupla_via.sh
#   QUICK=1 DUMMY=1 NO_PRETRAINED=1 DEVICE=cpu bash run_dupla_via.sh
# ==============================================================================
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
cd "$here"

DATASETS="${DATASETS:-UCSB CR LA LG NHL}"
AMP="${AMP:-0}"
QUICK="${QUICK:-0}"
DUMMY="${DUMMY:-0}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"

# Detecção automática de dispositivo e GPUs
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    DETECTED_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
    GPUS="${GPUS:-$DETECTED_GPUS}"
    DEVICE="${DEVICE:-cuda}"
else
    GPUS="${GPUS:-}"
    DEVICE="${DEVICE:-cpu}"
fi

RUNNER="${RUNNER:-uv run --with timm,torch,torchvision,pandas,scikit-learn,opencv-python --no-sync}"

EXTRA_FLAGS=""
[ "$AMP" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --amp"
[ "$QUICK" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --quick"
[ "$DUMMY" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --dummy"
[ "$NO_PRETRAINED" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --no_pretrained"

LOG_DIR="logs_dupla_via"
mkdir -p "$LOG_DIR" results_dupla_via

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run_dupla_via.log"
}

log "Iniciando experimento da Rede de Dupla Via (Two-Stream Network)"
log "Datasets: $DATASETS"
log "Device:   $DEVICE (GPUs: ${GPUS:-nenhuma})"

for ds in $DATASETS; do
    log "Iniciando dataset $ds..."
    t0="$(date +%s)"
    if $RUNNER src/treino_dupla_via.py \
        --dataset "$ds" --device "$DEVICE" $EXTRA_FLAGS \
        >> "$LOG_DIR/${ds}.log" 2>&1; then
        log "OK: $ds ($(( $(date +%s) - t0 ))s)"
    else
        log "FALHA: $ds (verificar $LOG_DIR/${ds}.log)"
    fi
done

log "Gerando resumo consolidado e tabelas LaTeX..."
$RUNNER src/treino_dupla_via.py --summary | tee -a "$LOG_DIR/summary.log"

log "Experimento da Rede de Dupla Via concluído com sucesso! Resultados em results_dupla_via/."
