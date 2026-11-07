#!/usr/bin/env bash
# Teste final: modelo de dupla via (topológica + visual) x 4 variantes x 5 datasets, controles de capacidade,
# avaliação no holdout e análise.
# Usa todas as GPUs: cada GPU roda PER_GPU tarefas ao mesmo tempo, e cada tarefa livre pega o próximo
# trabalho da fila (escalonamento dinâmico).
# Uso (da raiz do projeto, dentro do tmux):   bash run_final.sh
# Variáveis (padrão entre parênteses):
#   GPUS (todas as detectadas)   PER_GPU (2)   AMP (0)   REPEATS (3)   DEVICE (cuda)   DRY=1 (só lista)
#   DATASETS ("NHL LA CR LG UCSB")
#   VARIANTS  codificador:representação ("mlp:diff bilstm:diff mlp:cum bilstm:cum")
#   PRIMARY   variante primária, a única que recebe os controles ("mlp:diff")
#   CONTROLS  controles de capacidade ("permtrain permtest"; "" desliga)
#   HOLDOUT   1 roda também a avaliação única no holdout de cada variante (1)
#   EXTRA     argumentos extras do dupla_via.py (ex.: EXTRA="--lr_bb 1e-4 3e-4 1e-3 --max_epochs 50 --patience 10")
#   TAG       sufixo nos arquivos de saída (ex.: TAG=__piloto); passe o mesmo TAG na análise
#   JOBS (força o nº total de tarefas simultâneas)   OUT (results_v2/final)
# É seguro interromper e rodar de novo: tarefas concluídas são puladas e as interrompidas retomam do último fold.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DATASETS="${DATASETS:-NHL LA CR LG UCSB}"
VARIANTS="${VARIANTS:-mlp:diff bilstm:diff mlp:cum bilstm:cum}"
PRIMARY="${PRIMARY:-mlp:diff}"
CONTROLS="${CONTROLS-permtrain permtest}"
HOLDOUT="${HOLDOUT:-1}"
REPEATS="${REPEATS:-3}"; AMP="${AMP:-0}"; PER_GPU="${PER_GPU:-2}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"; TAG="${TAG:-}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}"
GPUS="${GPUS:-0}"
read -ra GARR <<< "$GPUS"; NGPU="${#GARR[@]}"
NSLOTS="${JOBS:-$(( NGPU * PER_GPU ))}"
NCPU="$(nproc 2>/dev/null || echo 8)"
THREADS="$(( NCPU / NSLOTS ))"; (( THREADS < 1 )) && THREADS=1; (( THREADS > 4 )) && THREADS=4
OUT="${OUT:-results_v2/final}"; DRY="${DRY:-0}"
RUNNER="${RUNNER:-uv run --no-sync}"
LOCKDIR="logs_v2/gpu_slots_final"
mkdir -p logs_v2/jobs_final "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a logs_v2/run_final.log; }

# colunas: dataset enc rep ctrl modo (a variante primária e a CV primeiro dentro de cada dataset)
JOBFILE=logs_v2/jobs_final.txt; : > "$JOBFILE"
PENC="${PRIMARY%%:*}"; PREP="${PRIMARY##*:}"
for ds in $DATASETS; do
    echo "$ds $PENC $PREP none cv" >> "$JOBFILE"
    for v in $VARIANTS; do
        enc="${v%%:*}"; rep="${v##*:}"
        if [[ "$enc" != "$PENC" || "$rep" != "$PREP" ]]; then echo "$ds $enc $rep none cv" >> "$JOBFILE"; fi
    done
    for c in $CONTROLS; do echo "$ds $PENC $PREP $c cv" >> "$JOBFILE"; done
    if [[ "$HOLDOUT" == "1" ]]; then
        for v in $VARIANTS; do echo "$ds ${v%%:*} ${v##*:} none holdout" >> "$JOBFILE"; done
    fi
done
log "$(wc -l < "$JOBFILE") tarefas; GPUs: $GPUS x $PER_GPU = $NSLOTS simultâneas; AMP=$AMP REPEATS=$REPEATS TAG='$TAG'"
if [[ "$DRY" == "1" ]]; then cat "$JOBFILE"; exit 0; fi
rm -rf "$LOCKDIR"; mkdir -p "$LOCKDIR"

run_job() {
    local ds="$1" enc="$2" rep="$3" ctrl="$4" mode="$5" slot=-1 i t0
    read -ra G <<< "$GPUS"
    while :; do   # pega um espaço livre (mkdir é atômico); a GPU é o espaço módulo o nº de GPUs
        for (( i = 0; i < NSLOTS; i++ )); do
            if mkdir "$LOCKDIR/slot$i" 2>/dev/null; then slot=$i; break 2; fi
        done
        sleep 3
    done
    local gpu="${G[$(( slot % ${#G[@]} ))]}" name="${ds}__${enc}__${rep}__${ctrl}__${mode}__R${REPEATS}${TAG}"
    local flags=()
    [[ "$AMP" == "1" ]] && flags+=(--amp)
    [[ "$mode" == "holdout" ]] && flags+=(--holdout)
    t0="$(date +%s)"
    if OMP_NUM_THREADS="$THREADS" CUDA_VISIBLE_DEVICES="$gpu" $RUNNER src/dupla_via.py \
        --dataset "$ds" --enc "$enc" --rep "$rep" --ctrl "$ctrl" --repeats "$REPEATS" --tag "$TAG" \
        --device "$DEVICE" --out_dir "$OUT" ${flags[@]+"${flags[@]}"} $EXTRA >> "logs_v2/jobs_final/${name}.log" 2>&1; then
        echo "[$(date '+%F %T')] ok    $name (GPU $gpu, $(( $(date +%s) - t0 ))s)" >> logs_v2/run_final.log
        rmdir "$LOCKDIR/slot$slot"
    else
        echo "[$(date '+%F %T')] FALHA $name (GPU $gpu; ver logs_v2/jobs_final/${name}.log)" >> logs_v2/run_final.log
        rmdir "$LOCKDIR/slot$slot"
        return 1
    fi
}
export -f run_job
export RUNNER REPEATS DEVICE OUT GPUS AMP THREADS NSLOTS LOCKDIR TAG EXTRA
FAIL=0
xargs -P "$NSLOTS" -L 1 bash -c 'run_job "$@"' _ < "$JOBFILE" || FAIL=1

log "Análise"
$RUNNER src/analise_final.py --repeats "$REPEATS" --tag "$TAG" --out_dir "$(dirname "$OUT")" 2>&1 \
    | tee -a logs_v2/analise_final.log | tail -n 5
log "Concluído. Leia $OUT/report_final.md"
[[ "$FAIL" == "0" ]] || { log "ATENÇÃO: algumas tarefas falharam; rode de novo para retomar"; exit 1; }
