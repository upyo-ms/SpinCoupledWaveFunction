#!/usr/bin/env bash
# Launch inside an allocated node, from an activated Python environment.
# Scheduler GPU visibility and CPU allocation are respected; no module names assumed.
set -euo pipefail
SCGVB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCGVB_ROOT"
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
SCGVB_MODE="${1:-benchmark}"
SCGVB_OUT="${SCGVB_OUTPUT_DIR:-results/12pi-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$SCGVB_OUT"
python hpc/check_environment.py --device gpu | tee "$SCGVB_OUT/environment.json"
python -m pip freeze > "$SCGVB_OUT/pip-freeze.txt"
case "$SCGVB_MODE" in
  benchmark)
    python benchmark_active_space.py --device gpu --verify --repeats 5 \
      --output "$SCGVB_OUT/synthetic12.json" > "$SCGVB_OUT/synthetic12.log" 2>&1
    python benchmark_active_space.py --device gpu --xyz examples/polyene12.xyz \
      --pi-electrons 12 --verify --repeats 5 \
      --output "$SCGVB_OUT/polyene12.json" > "$SCGVB_OUT/polyene12.log" 2>&1
    ;;
  optimize)
    python calc_molecule.py --xyz examples/polyene12.xyz --active-space pi --pi-electrons 12 \
      --energy-evaluation determinant --backend jax --device gpu --initial-guess hf \
      --verbose --output "$SCGVB_OUT/polyene12_result.json" > "$SCGVB_OUT/optimization.log" 2>&1
    ;;
  verify)
    SCGVB_RUN_12E=1 SCGVB_TEST_DEVICE=gpu python -m unittest test_memory_evaluator -v \
      > "$SCGVB_OUT/verification.log" 2>&1
    ;;
  *) printf '%s\n' 'Usage: bash hpc/run_12pi.sh [benchmark|optimize|verify]' >&2; exit 2 ;;
esac
printf 'Results: %s\n' "$SCGVB_OUT"
