#!/usr/bin/env bash
# Ablação topológica v3: 5 datasets x 6 modelos x 180 combinações x 5 folds, depois resumo + holdout.
# Uso (raiz do projeto): bash src/run_ablacao_topologica_v3.sh
# Variáveis: JOBS, DATASETS, MODELS, RUNNER, DRY=1, QUICK=1, DUMMY=1
# Seguro interromper e retomar.
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
if [ -f "$here/src/ablacao_via_topologica_v3.py" ]; then cd "$here"; else cd "$here/.."; fi

JOBS="${JOBS:-10}"
DATASETS="${DATASETS:-NHL LA LG CR UCSB}"
MODELS="${MODELS:-lstm bilstm gru bigru saclstm mlp}"
RUNNER="${RUNNER:-uv run --no-sync}"
EXTRA=""
[ "${DUMMY:-0}" = "1" ] && EXTRA="$EXTRA --dummy"
[ "${QUICK:-0}" = "1" ] && EXTRA="$EXTRA --quick"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p logs_topologica_v3
tasks=()
for ds in $DATASETS; do for m in $MODELS; do tasks+=("$ds $m"); done; done

if [ "${DRY:-0}" = "1" ]; then printf '%s\n' "${tasks[@]}"; exit 0; fi

$RUNNER src/ablacao_via_topologica_v3.py --check-leakage $EXTRA | tee logs_topologica_v3/leakage.log

printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -L 1 bash -c \
  "$RUNNER src/ablacao_via_topologica_v3.py --dataset \$0 --model \$1 $EXTRA >> logs_topologica_v3/\$0__\$1.log 2>&1"

$RUNNER src/ablacao_via_topologica_v3.py --summary $EXTRA
$RUNNER src/ablacao_via_topologica_v3.py --holdout $EXTRA
