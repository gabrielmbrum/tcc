#!/usr/bin/env bash
# Ablação da via visual: arquiteturas x {frozen, ft com aumento, ft sem aumento} x 5 datasets, e análise.
# Usa todas as GPUs: cada GPU roda PER_GPU tarefas ao mesmo tempo, e cada tarefa livre pega o próximo
# trabalho da fila (escalonamento dinâmico: a GPU mais rápida processa mais tarefas).
# Uso (da raiz do projeto, dentro do tmux):   bash run_visual.sh
# Variáveis (padrão entre parênteses):
#   GPUS (todas as detectadas, ex.: "0 1")   PER_GPU (2; suba para 3 se a memória e a utilização sobrarem)
#   AMP (0; 1 liga a precisão mista bfloat16, mais rápida)   TRIALS (12)   REPEATS (3)
#   DATASETS ("NHL LA CR LG UCSB")   ARCHS ("mobilevit_s efficientnet_b0 deit_tiny")
#   AUG_ARCHS (arquiteturas que também rodam o ajuste fino SEM aumento; "mobilevit_s")
#   JOBS (força o nº total de tarefas simultâneas)   DEVICE (cuda/cpu)   DRY=1 (só lista)
# Os arquivos incluem __T<TRIALS>__R<REPEATS> (e __amp): resultados de outra configuração não são sobrescritos.
# É seguro interromper e rodar de novo: tarefas concluídas são puladas e as interrompidas retomam do último fold.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DATASETS="${DATASETS:-NHL LA CR LG UCSB}"
ARCHS="${ARCHS:-mobilevit_s efficientnet_b0 deit_tiny}"
AUG_ARCHS="${AUG_ARCHS-mobilevit_s}"
TRIALS="${TRIALS:-12}"; REPEATS="${REPEATS:-3}"; AMP="${AMP:-0}"; PER_GPU="${PER_GPU:-2}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
GPUS="${GPUS:-0}"
read -ra GARR <<< "$GPUS"; NGPU="${#GARR[@]}"
NSLOTS="${JOBS:-$(( NGPU * PER_GPU ))}"
NCPU="$(nproc 2>/dev/null || echo 8)"
THREADS="$(( NCPU / NSLOTS ))"; (( THREADS < 1 )) && THREADS=1; (( THREADS > 4 )) && THREADS=4
OUT="${OUT:-results_v2/visual}"; DRY="${DRY:-0}"
RUNNER="${RUNNER:-uv run --no-sync}"
LOCKDIR="logs_v2/gpu_slots"
mkdir -p logs_v2/jobs_visual "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a logs_v2/run_visual.log; }

# colunas: dataset arch modo aumento (as tarefas mais pesadas primeiro dentro de cada dataset)
JOBFILE=logs_v2/jobs_visual.txt; : > "$JOBFILE"
for ds in $DATASETS; do
    for arch in $ARCHS; do
        echo "$ds $arch ft flip_rot" >> "$JOBFILE"
        for aa in $AUG_ARCHS; do [[ "$aa" == "$arch" ]] && echo "$ds $arch ft none" >> "$JOBFILE"; done
    done
    for arch in $ARCHS; do echo "$ds $arch frozen none" >> "$JOBFILE"; done
done
log "$(wc -l < "$JOBFILE") tarefas; GPUs: $GPUS x $PER_GPU = $NSLOTS simultâneas; AMP=$AMP TRIALS=$TRIALS REPEATS=$REPEATS"
if [[ "$DRY" == "1" ]]; then cat "$JOBFILE"; exit 0; fi
rm -rf "$LOCKDIR"; mkdir -p "$LOCKDIR"

run_job() {
    local ds="$1" arch="$2" mode="$3" aug="$4" slot=-1 i t0
    read -ra G <<< "$GPUS"
    while :; do   # pega um espaço livre (mkdir é atômico); a GPU é o espaço módulo o nº de GPUs
        for (( i = 0; i < NSLOTS; i++ )); do
            if mkdir "$LOCKDIR/slot$i" 2>/dev/null; then slot=$i; break 2; fi
        done
        sleep 3
    done
    local gpu="${G[$(( slot % ${#G[@]} ))]}" name="${ds}__${arch}__${mode}__${aug}__T${TRIALS}__R${REPEATS}"
    local amp_flag=(); [[ "$AMP" == "1" ]] && amp_flag=(--amp)
    t0="$(date +%s)"
    if OMP_NUM_THREADS="$THREADS" CUDA_VISIBLE_DEVICES="$gpu" $RUNNER src/ablacao_visual.py \
        --dataset "$ds" --arch "$arch" --mode "$mode" --aug "$aug" --repeats "$REPEATS" --trials "$TRIALS" \
        --device "$DEVICE" --out_dir "$OUT" "${amp_flag[@]}" >> "logs_v2/jobs_visual/${name}.log" 2>&1; then
        echo "[$(date '+%F %T')] ok    $name (GPU $gpu, $(( $(date +%s) - t0 ))s)" >> logs_v2/run_visual.log
        rmdir "$LOCKDIR/slot$slot"
    else
        echo "[$(date '+%F %T')] FALHA $name (GPU $gpu; ver logs_v2/jobs_visual/${name}.log)" >> logs_v2/run_visual.log
        rmdir "$LOCKDIR/slot$slot"
        return 1
    fi
}
export -f run_job
export RUNNER REPEATS TRIALS DEVICE OUT GPUS AMP THREADS NSLOTS LOCKDIR
FAIL=0
xargs -P "$NSLOTS" -L 1 bash -c 'run_job "$@"' _ < "$JOBFILE" || FAIL=1

log "Análise"
$RUNNER src/analise_visual.py --repeats "$REPEATS" --trials "$TRIALS" --out_dir "$(dirname "$OUT")" 2>&1 \
    | tee -a logs_v2/analise_visual.log | tail -n 5
log "Concluído. Leia $OUT/report_visual.md"
[[ "$FAIL" == "0" ]] || { log "ATENÇÃO: algumas tarefas falharam; rode de novo para retomar"; exit 1; }
