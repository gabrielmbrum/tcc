#!/usr/bin/env bash
# ==============================================================================
# Ablação da Via Visual Isolada (Protocolo exposto em tex/ablacao.tex)
# ==============================================================================
# Modelos (6): resnet50, efficientnet_b0, vit_tiny, deit_tiny, mobilevit_s, coatnet_0
# Datasets (5): UCSB, CR, LA, LG, NHL
# Grade: 108 combinações por modelo x 5 dobras = 540 treinos por (dataset, modelo)
#
# Uso (a partir da raiz do projeto ~/tcc):
#     bash run_visual.sh
#
# Variáveis de ambiente configuráveis (com valores padrão):
#   GPUS            Lista de GPUs a utilizar (ex.: "0" ou "0 1"; detectada via nvidia-smi)
#   PER_GPU         Número de tarefas simultâneas por GPU (padrão: 1; suba para 2 se VRAM permitir)
#   DEVICE          Dispositivo PyTorch ("cuda" ou "cpu"; padrão: cuda se GPU presente)
#   AMP             Ativa precisão mista bfloat16/float16 (padrão: 0; defina AMP=1 para ativar)
#   DATASETS        Conjuntos de dados (padrão: "UCSB CR LA LG NHL")
#   MODELS          Arquiteturas (padrão: "resnet50 efficientnet_b0 vit_tiny deit_tiny mobilevit_s coatnet_0")
#   RUNNER          Executor Python (padrão: uv run --with timm,torch,torchvision,pandas,scikit-learn,opencv-python --no-sync)
#   DRY             Se 1, apenas lista as tarefas e encerra (padrão: 0)
#   QUICK           Se 1, executa teste mínimo de fumaça (padrão: 0)
#   DUMMY           Se 1, gera dados sintéticos aleatórios (padrão: 0)
#   NO_PRETRAINED   Se 1, inicializa sem baixar pesos pré-treinados (padrão: 0)
#   FORCE           Se 1, reexecuta tarefas já concluídas (padrão: 0)
# ==============================================================================
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
cd "$here"

DATASETS="${DATASETS:-UCSB CR LA LG NHL}"
MODELS="${MODELS:-resnet50 efficientnet_b0 vit_tiny deit_tiny mobilevit_s coatnet_0}"
PER_GPU="${PER_GPU:-1}"
AMP="${AMP:-0}"
DRY="${DRY:-0}"
QUICK="${QUICK:-0}"
DUMMY="${DUMMY:-0}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
FORCE="${FORCE:-0}"

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
[ "$FORCE" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --force"

LOG_DIR="logs_visual"
LOCK_DIR="$LOG_DIR/gpu_slots"
mkdir -p "$LOG_DIR" results_visual/folds

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run_visual.log"
}

# Monta a fila de tarefas (dataset x modelo)
tasks=()
for ds in $DATASETS; do
    for m in $MODELS; do
        tasks+=("$ds $m")
    done
done

log "Iniciando pipeline de ablação visual: ${#tasks[@]} tarefas agendadas."
log "Datasets: $DATASETS"
log "Modelos:  $MODELS"
log "Device:   $DEVICE (GPUs: ${GPUS:-nenhuma})"

if [ "$DRY" = "1" ]; then
    printf '%s\n' "${tasks[@]}"
    exit 0
fi

if [ "$DEVICE" = "cuda" ] && [ -n "$GPUS" ]; then
    read -ra GARR <<< "$GPUS"
    NGPU="${#GARR[@]}"
    NSLOTS=$(( NGPU * PER_GPU ))
    rm -rf "$LOCK_DIR"
    mkdir -p "$LOCK_DIR"

    log "Escalonamento dinâmico GPU: $NGPU GPU(s) com $PER_GPU tarefa(s)/GPU = $NSLOTS tarefas concorrentes."

    run_gpu_task() {
        local ds="$1" m="$2" slot=-1 i t0
        read -ra G <<< "$GPUS"
        while :; do
            for (( i = 0; i < NSLOTS; i++ )); do
                if mkdir "$LOCK_DIR/slot$i" 2>/dev/null; then
                    slot=$i
                    break 2
                fi
            done
            sleep 2
        done

        local gpu="${G[$(( slot % ${#G[@]} ))]}"
        local job_name="${ds}__${m}"
        t0="$(date +%s)"
        log "Disparando $job_name na GPU $gpu (slot $slot)"

        if CUDA_VISIBLE_DEVICES="$gpu" $RUNNER src/ablacao_via_visual.py \
            --dataset "$ds" --model "$m" --device cuda $EXTRA_FLAGS \
            >> "$LOG_DIR/${job_name}.log" 2>&1; then
            log "OK: $job_name (GPU $gpu, $(( $(date +%s) - t0 ))s)"
            rmdir "$LOCK_DIR/slot$slot"
        else
            log "FALHA: $job_name (GPU $gpu; verificar $LOG_DIR/${job_name}.log)"
            rmdir "$LOCK_DIR/slot$slot"
            return 1
        fi
    }
    export -f run_gpu_task log
    export RUNNER GPUS NSLOTS LOCK_DIR LOG_DIR EXTRA_FLAGS

    FAIL=0
    printf '%s\n' "${tasks[@]}" | xargs -P "$NSLOTS" -L 1 bash -c 'run_gpu_task "$@"' _ || FAIL=1
    rm -rf "$LOCK_DIR"
else
    NCPU="$(nproc 2>/dev/null || echo 4)"
    JOBS="${JOBS:-$(( NCPU > 4 ? 4 : NCPU ))}"
    log "Execução em CPU com $JOBS tarefas simultâneas."

    run_cpu_task() {
        local ds="$1" m="$2" job_name="${ds}__${m}" t0
        t0="$(date +%s)"
        log "Disparando $job_name em CPU"
        if OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $RUNNER src/ablacao_via_visual.py \
            --dataset "$ds" --model "$m" --device cpu $EXTRA_FLAGS \
            >> "$LOG_DIR/${job_name}.log" 2>&1; then
            log "OK: $job_name ($(( $(date +%s) - t0 ))s)"
        else
            log "FALHA: $job_name (verificar $LOG_DIR/${job_name}.log)"
            return 1
        fi
    }
    export -f run_cpu_task log
    export RUNNER LOG_DIR EXTRA_FLAGS

    FAIL=0
    printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -L 1 bash -c 'run_cpu_task "$@"' _ || FAIL=1
fi

if [ "$FAIL" = "0" ]; then
    log "Todas as tarefas da validação cruzada concluídas com sucesso."
    log "Gerando resumo e tabelas LaTeX..."
    $RUNNER src/ablacao_via_visual.py --summary $EXTRA_FLAGS | tee -a "$LOG_DIR/summary.log"

    log "Executando avaliação confirmatória no teste cego de 20% (Holdout)..."
    $RUNNER src/ablacao_via_visual.py --holdout $EXTRA_FLAGS | tee -a "$LOG_DIR/holdout.log"
    log "Ablação da via visual concluída. Resultados em results_visual/."
else
    log "ATENÇÃO: Uma ou mais tarefas falharam. Corrija o problema e execute novamente; tarefas concluídas serão puladas automaticamente."
    exit 1
fi
