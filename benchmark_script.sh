#!/bin/bash

set -e

LOG_DIR="benchmark_logs"
mkdir -p "$LOG_DIR"

TOTAL=$(wc -l < batch_commands.txt)
COUNT=0
PASSED=0
FAILED=0

while IFS= read -r cmd; do
    # Skip empty lines
    [[ -z "$cmd" ]] && continue

    COUNT=$((COUNT + 1))

    # Extract repo name for logging
    REPO=$(echo "$cmd" | grep -oP '(?<=--full_name ")[^"]+')
    SAFE_NAME=$(echo "$REPO" | tr '/' '_')

    echo "============================================"
    echo "[$COUNT/$TOTAL] Processing: $REPO"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    if eval "$cmd" 2>&1 | tee "$LOG_DIR/${SAFE_NAME}.log"; then
        echo "[$COUNT/$TOTAL] SUCCESS: $REPO"
        PASSED=$((PASSED + 1))
    else
        echo "[$COUNT/$TOTAL] FAILED: $REPO"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done < batch_commands.txt

echo "============================================"
echo "BENCHMARK COMPLETE"
echo "Total: $TOTAL | Passed: $PASSED | Failed: $FAILED"
echo "============================================"