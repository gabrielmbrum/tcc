#!/usr/bin/env bash
# Ablação da via topológica (protocolo de tex/ablacao.tex): 5 datasets x 6 modelos x 324 combinações
# (UCSB: lote {4,8,16}; demais: {16,32,64}) x 5 rodadas. Tarefas = (dataset, modelo).
#
# Uso (a partir da raiz do projeto):
#     bash src/run_ablacao_topologica.sh
# Variáveis: JOBS (nº de núcleos), DATASETS, MODELS, DUMMY=1 (dados sintéticos), QUICK=1 (fumaça),
#            RUNNER (padrão: "uv run --no-sync"), DRY=1 (só lista).
# É seguro interromper e rodar de novo: combinações concluídas são puladas.
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
if [ -f "$here/src/ablacao_via_topologica.py" ]; then cd "$here"; else cd "$here/.."; fi
[ -f src/ablacao_via_topologica.py ] || { echo "ERRO: src/ablacao_via_topologica.py não encontrado em $(pwd)"; exit 1; }

JOBS="${JOBS:-10}"
DATASETS="${DATASETS:-NHL LA LG CR UCSB}"
MODELS="${MODELS:-lstm bilstm gru bigru saclstm mlp}"
RUNNER="${RUNNER:-uv run --no-sync}"
EXTRA=""
[ "${DUMMY:-0}" = "1" ] && EXTRA="$EXTRA --dummy"
[ "${QUICK:-0}" = "1" ] && EXTRA="$EXTRA --quick"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p logs_topologica
tasks=()
for ds in $DATASETS; do for m in $MODELS; do tasks+=("$ds $m"); done; done

if [ "${DRY:-0}" = "1" ]; then printf '%s\n' "${tasks[@]}"; exit 0; fi

printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -L 1 bash -c \
  "$RUNNER src/ablacao_via_topologica.py --dataset \$0 --model \$1 $EXTRA >> logs_topologica/\$0__\$1.log 2>&1"

$RUNNER src/ablacao_via_topologica.py --summary
