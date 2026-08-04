#!/usr/bin/env bash
# Profile the quack mxfp8 GEMM kernel with Nsight Compute.
#
# Cute DSL mangles the kernel symbol from the Python class path. Match the
# dGated class name specifically so setup quantization is never profiled.
#
# Usage:
#   ./benchmark.sh                # profile, default skip=0
#   ./benchmark.sh --list         # list every kernel launched (no metrics, fast)
#   SKIP=2 ./benchmark.sh out2    # custom skip count, custom output name
#   DRIVER=ncu_drive_down_proj.py ./benchmark.sh down_proj   # profile a different driver script

set -e

NCU="/opt/nvidia/nsight-compute/2025.2.1/ncu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$SCRIPT_DIR/.venv-quack-bench/bin/python}"
DRIVER="${DRIVER:-$SCRIPT_DIR/ncu_drive_mxfp8_dgated.py}"
read -r -a EXTRA_DRIVER_ARGS <<< "${DRIVER_ARGS:-}"
BENCH=("$DRIVER" --warmup 5 --target 1 "${EXTRA_DRIVER_ARGS[@]}")
# SUDO_ENV=(sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64")
SUDO_ENV=(sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64")

if [ "$1" = "--list" ]; then
  # Enumerate kernels with minimal section work. mktemp -u so ncu can create the
  # log file (it refuses to overwrite an existing one).
  LOG=$(mktemp -u --suffix=.csv)
  "${SUDO_ENV[@]}" "$NCU" \
    --target-processes all \
    --launch-count 50 \
    --print-summary per-kernel \
    --csv \
    --log-file "$LOG" \
    "$PY" "${BENCH[@]}" > /dev/null 2>&1 || true
  # Find which CSV column is "Kernel Name" from the header, then print unique values.
  python3 - "$LOG" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.reader(f))
hdr_idx = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
if hdr_idx is None:
    print(f"(no kernel rows found — open {path} to inspect)"); sys.exit(0)
col = rows[hdr_idx].index("Kernel Name")
seen = set()
for r in rows[hdr_idx + 1:]:
    if len(r) > col and r[col] and r[col] not in seen:
        seen.add(r[col]); print(r[col])
PY
  sudo rm -f "$LOG"
  exit 0
fi

OUT="${1:-ncu_profile}"
SKIP="${SKIP:-0}"
KERNEL_REGEX="${KERNEL_REGEX:-^kernel_cutlass_kernel_quackgemm_dactGemmDGatedSm90}"

"${SUDO_ENV[@]}" \
  "$NCU" \
    --target-processes all \
    --kernel-name "regex:${KERNEL_REGEX}" \
    --launch-skip "$SKIP" \
    --launch-count 1 \
    --set full \
    --export "$OUT" \
    --force-overwrite \
    "$PY" "${BENCH[@]}"

echo
echo "Profile written to ${OUT}.ncu-rep"
echo "View:    ncu-ui ${OUT}.ncu-rep"
echo "Summary: ncu --import ${OUT}.ncu-rep --print-summary per-kernel"
