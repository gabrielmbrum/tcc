#!/usr/bin/env bash
# Ablação robusta da via topológica (v2): CV aninhada, busca de hiperparâmetros por
# codificador, baselines, r_max, resolução (nativa vs 256x256), sanidade por permutação e
# análise estatística.
#
# Uso (a partir de ~/tcc, dentro do tmux):
#     bash run_all_v2.sh
# Variáveis opcionais (valores padrão entre parênteses):
#     JOBS         tarefas em paralelo, 1 thread cada     (nº de núcleos, no máximo 16)
#     BUILD_THREADS threads OpenMP só na geração dos .npz (nº de núcleos)
#     REPEATS      repetições da CV externa 5-fold         (3; entra no nome dos arquivos de saída)
#     TRIALS       configurações por codificador           (12; a 1ª é a padrão)
#     INNER_K      folds da CV interna                     (3)
#     DATASETS     "NHL LA CR LG UCSB"                     (maiores primeiro)
#     ENCS         "bilstm lstm gru mlp cnn1d logreg svm rf knn dummy"
#     REPS         "cum diff"
#     RMAX_ENCS    "bilstm mlp"   RMAXS "4 7 10"           (r_max=15 vem do estágio A; "" desliga o estágio B)
#     RES_ENCS     "bilstm mlp"                            (estágio D: resolução 256x256 vs nativa; "" desliga)
#     SANITY_ENCS  "bilstm mlp"                            (rótulos permutados; "" desliga o estágio C)
#     REF          bilstm                                  (referência dos testes pareados)
#     DRY=1        só lista as tarefas;  DUMMY=1  dados sintéticos (teste do pipeline)
# Estágios: A = codificador x representação (r_max=15, resolução nativa); B = r_max;
#           C = sanidade; D = mesma via topológica sobre a imagem 256x256 (data/<DS>/topological_256.npz).
# É seguro interromper e rodar de novo: tarefas concluídas são puladas e as interrompidas
# retomam do último fold salvo. Os arquivos de saída incluem __R<REPEATS>: mudar REPEATS cria
# arquivos novos em vez de pular/sobrescrever. Resultados em results_v2/, logs em logs_v2/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

NCPU="$(nproc 2>/dev/null || echo 4)"
JOBS="${JOBS:-$(( NCPU > 16 ? 16 : NCPU ))}"
BUILD_THREADS="${BUILD_THREADS:-$NCPU}"
REPEATS="${REPEATS:-3}"
TRIALS="${TRIALS:-12}"
INNER_K="${INNER_K:-3}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
DATASETS="${DATASETS:-NHL LA CR LG UCSB}"
ENCS="${ENCS:-bilstm lstm gru mlp cnn1d logreg svm rf knn dummy}"
REPS="${REPS:-cum diff}"
RMAX_ENCS="${RMAX_ENCS-bilstm mlp}"      # "-" (sem ":"): string vazia desliga o estágio
RMAXS="${RMAXS:-4 7 10}"
RES_ENCS="${RES_ENCS-bilstm mlp}"
SANITY_ENCS="${SANITY_ENCS-bilstm mlp}"
REF="${REF:-bilstm}"
OUT="${OUT:-results_v2}"
DEVICE="${DEVICE:-cpu}"
DUMMY="${DUMMY:-0}"
DRY="${DRY:-0}"
RUNNER="${RUNNER:-uv run --no-sync}"
# Treinos: 1 thread por tarefa. O build dos .npz (C++ com OpenMP) usa BUILD_THREADS (ver abaixo).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p logs_v2/jobs "$OUT"
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a logs_v2/run_all_v2.log; }

# ---------------------------------------------------------------- 1. sequências topológicas
# npz_complete <ds> <arquivo> <resize|0>: sucesso se o .npz existe e cobre todas as imagens
# (e, quando as chaves existem, bate com weight=ribas2024, r_max=15 e o resize pedido).
npz_complete() {
    [[ -f "data/$1/$2" ]] || return 1
    $RUNNER - "$1" "$2" "$3" <<'EOF2'
import sys, numpy as np
from pathlib import Path
ds, fname, resize = Path("data") / sys.argv[1], sys.argv[2], int(sys.argv[3])
ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
n_img = sum(1 for p in ds.rglob("*") if p.suffix.lower() in ext and ".topo_cache" not in str(p))
d = np.load(ds / fname)
ok = n_img == len(d["y"]) and d["X"].shape[1] == 15
if "weight" in d.files:
    ok = ok and str(d["weight"]) == "ribas2024"
if resize:
    ok = ok and "resize" in d.files and int(d["resize"]) == resize
sys.exit(0 if ok else 1)
EOF2
}

if [[ "$DUMMY" != "1" && "$DRY" != "1" ]]; then
    log "Compilando src/SSNModel.so"
    g++ -O3 -shared -fPIC -fopenmp -o src/SSNModel.so src/SSNModel.cpp
    for ds in $DATASETS; do
        if npz_complete "$ds" topological.npz 0; then
            log "[$ds] topological.npz completo, pulando"
        else
            log "[$ds] gerando sequências topológicas (nativa), $BUILD_THREADS threads"
            OMP_NUM_THREADS="$BUILD_THREADS" $RUNNER src/build_topological.py --datasets "$ds" \
                2>&1 | tee -a "logs_v2/build_${ds}.log"
        fi
        if [[ -n "$RES_ENCS" ]]; then
            if npz_complete "$ds" topological_256.npz 256; then
                log "[$ds] topological_256.npz completo, pulando"
            else
                log "[$ds] gerando sequências topológicas (256x256), $BUILD_THREADS threads"
                OMP_NUM_THREADS="$BUILD_THREADS" $RUNNER src/build_topological.py --datasets "$ds" \
                    --resize 256 2>&1 | tee -a "logs_v2/build_${ds}_256.log"
            fi
        fi
    done
fi

# ---------------------------------------------------------------- 2. lista de tarefas
# colunas: dataset enc rep rmax modo resolução
JOBFILE="logs_v2/jobs.txt"
: > "$JOBFILE"
for ds in $DATASETS; do   # estágio A: codificador x representação, r_max = 15
    for enc in $ENCS; do
        for rep in $REPS; do
            [[ "$enc" == "dummy" && "$rep" != "cum" ]] && continue
            echo "$ds $enc $rep 15 cv native" >> "$JOBFILE"
        done
    done
done
for ds in $DATASETS; do   # estágio B: sensibilidade a r_max
    for enc in $RMAX_ENCS; do
        for rep in $REPS; do
            for rm in $RMAXS; do echo "$ds $enc $rep $rm cv native" >> "$JOBFILE"; done
        done
    done
done
for ds in $DATASETS; do   # estágio C: sanidade (rótulos permutados)
    for enc in $SANITY_ENCS; do echo "$ds $enc cum 15 perm native" >> "$JOBFILE"; done
done
for ds in $DATASETS; do   # estágio D: resolução 256x256 (comparada à nativa do estágio A)
    for enc in $RES_ENCS; do
        for rep in $REPS; do echo "$ds $enc $rep 15 cv 256" >> "$JOBFILE"; done
    done
done
NJOBS="$(wc -l < "$JOBFILE")"
log "$NJOBS tarefas, $JOBS em paralelo; REPEATS=$REPEATS TRIALS=$TRIALS INNER_K=$INNER_K DEVICE=$DEVICE"
if [[ "$DRY" == "1" ]]; then cat "$JOBFILE"; exit 0; fi

# ---------------------------------------------------------------- 3. execução paralela
run_job() {
    local ds="$1" enc="$2" rep="$3" rmax="$4" mode="$5" res="$6"
    local name="${ds}__${enc}__${rep}__r${rmax}__${mode}__${res}__R${REPEATS}"
    local extra=(--repeats "$REPEATS" --trials "$TRIALS")
    [[ "$mode" == "perm" ]] && extra=(--permute --repeats 1 --trials 3)
    [[ "$DUMMY" == "1" ]] && extra+=(--dummy)
    local t0; t0="$(date +%s)"
    if $RUNNER src/ablacao_robusta.py --dataset "$ds" --enc "$enc" --rep "$rep" --rmax "$rmax" \
        --res "$res" --inner_k "$INNER_K" --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
        --out_dir "$OUT" --device "$DEVICE" "${extra[@]}" >> "logs_v2/jobs/${name}.log" 2>&1; then
        echo "[$(date '+%F %T')] ok    $name ($(( $(date +%s) - t0 ))s)" >> logs_v2/run_all_v2.log
    else
        echo "[$(date '+%F %T')] FALHA $name (ver logs_v2/jobs/${name}.log)" >> logs_v2/run_all_v2.log
        return 1
    fi
}
export -f run_job
export RUNNER REPEATS TRIALS INNER_K MAX_EPOCHS PATIENCE OUT DEVICE DUMMY

FAIL=0
xargs -P "$JOBS" -L 1 bash -c 'run_job "$@"' _ < "$JOBFILE" || FAIL=1

NDONE="$(( $(ls "$OUT"/folds/*.csv 2>/dev/null | grep -vc partial || true) ))"
log "estágios A/B/D concluídos: $NDONE arquivos em $OUT/folds (falhas: $FAIL)"

# ---------------------------------------------------------------- 4. análise
log "Análise estatística (REPEATS=$REPEATS)"
$RUNNER src/analise_ablacao.py --ref "$REF" --out_dir "$OUT" --repeats "$REPEATS" 2>&1 \
    | tee -a logs_v2/analise.log | tail -n 5
log "Concluído. Leia $OUT/report.md; tabela LaTeX em $OUT/tabela_ablacao.tex"
[[ "$FAIL" == "0" ]] || { log "ATENÇÃO: algumas tarefas falharam; rode de novo para tentar retomar"; exit 1; }
