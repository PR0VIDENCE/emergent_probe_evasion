#!/bin/bash
#
# Collect probe-evasion eval results from three model runs into one directory,
# excluding the large activation tensors (*.pt).
#
# Sources (read-only):
#   gptoss120b_full  /mnt/d2/acp23ajh/probe_evasion/gptoss120b_full
#   qwq32b_full      /home/acp23ajh/emergent_probe_evasion/data/eval_runs/qwq32b_full
#   olmo3_32b_full   /mnt/d2/acp23ajh/probe_evasion/olmo3_32b_full
#
# Everything inside each source is copied EXCEPT *.pt files (activations).
#
# Usage:
#   scripts/collect_multimodel_results.sh [DEST_DIR]
#
# Args:
#   DEST_DIR   Output directory (default: ./multimodel_results). Each source is
#              placed in a named subdirectory under DEST_DIR.
#
# Env:
#   TARBALL=1  Also produce DEST_DIR.tar.gz for easy transfer (default: 0).

set -euo pipefail

DEST_DIR="${1:-./multimodel_results}"
TARBALL="${TARBALL:-0}"

# subdir_name -> source path
declare -A SOURCES=(
    [gptoss120b_full]="/mnt/d2/acp23ajh/probe_evasion/gptoss120b_full"
    [qwq32b_full]="/home/acp23ajh/emergent_probe_evasion/data/eval_runs/qwq32b_full"
    [olmo3_32b_full]="/mnt/d2/acp23ajh/probe_evasion/olmo3_32b_full"
)

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required but not found on PATH." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

echo "Collecting results into: $DEST_DIR"
echo "Excluding: *.pt (activation tensors)"
echo

missing=0
for name in "${!SOURCES[@]}"; do
    src="${SOURCES[$name]}"
    if [ ! -d "$src" ]; then
        echo "WARNING: source not found, skipping: $src" >&2
        missing=$((missing + 1))
        continue
    fi

    n_pt=$(find "$src" -type f -name '*.pt' | wc -l | tr -d ' ')
    echo ">> $name"
    echo "   from: $src"
    echo "   skipping $n_pt .pt file(s)"

    # Trailing slash on src copies its *contents* into DEST_DIR/$name.
    rsync -a --exclude='*.pt' "$src/" "$DEST_DIR/$name/"
done

echo
echo "Done. Collected layout:"
du -sh "$DEST_DIR"/*/ 2>/dev/null || true

if [ "$missing" -gt 0 ]; then
    echo
    echo "NOTE: $missing source dir(s) were missing and skipped (see warnings above)."
fi

if [ "$TARBALL" = "1" ]; then
    tarball="${DEST_DIR%/}.tar.gz"
    echo
    echo "Creating tarball: $tarball"
    tar -czf "$tarball" -C "$(dirname "$DEST_DIR")" "$(basename "$DEST_DIR")"
    echo "Tarball size: $(du -sh "$tarball" | cut -f1)"
fi
