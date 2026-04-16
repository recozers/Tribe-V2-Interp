#!/bin/bash
cd ~/tribe_v2_interp
# HF_TOKEN lives in ~/.hf_secrets (not tracked in git); chmod 600
source ~/.hf_secrets
export HF_HOME=~/tribe_v2_interp/cache/hf

PY=".venv/bin/python -u feature_viz.py"
COMMON="--target-roi FFA --single-frame --skip-sweep --n-restarts 1 --full-steps 3000 --stages-preset extra-slow-coarse --suppress-rois V4,MT"

run_exp () {
    local tag="$1"; shift
    echo ""
    echo "=========================================="
    echo "[$tag]  start $(date)"
    echo "=========================================="
    $PY $COMMON "$@" 2>&1
    echo "[$tag]  end $(date)"
}

run_exp "1/3 beta=0.3" --suppress-beta 0.3 --out-dir ./outputs/FFA_supp_b0.3
run_exp "2/3 beta=1.0" --suppress-beta 1.0 --out-dir ./outputs/FFA_supp_b1.0
run_exp "3/3 beta=3.0" --suppress-beta 3.0 --out-dir ./outputs/FFA_supp_b3.0

echo ""
echo "=========================================="
echo "BETA SWEEP COMPLETE at $(date)"
echo "=========================================="
