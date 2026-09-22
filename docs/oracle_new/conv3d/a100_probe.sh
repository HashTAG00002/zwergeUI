#!/bin/bash
# A100 Conv3d dispatch probe: run conv3d_probe.py under each relevant conda env.
# Submit via: hope run probe.hope   (from this directory)
set -x
nvidia-smi
OUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/conv3d_probe"
mkdir -p "${OUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS="qwen3 qwen3-verl qwen35"
for E in ${ENVS}; do
    PY="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs/${E}/bin/python"
    echo "===== ENV ${E} : $(${PY} -c 'import torch; print(torch.__version__)') ====="
    "${PY}" "${SCRIPT_DIR}/conv3d_probe.py" "${OUT_DIR}/a100_${E}.json"
done
echo "ALL_DONE"
ls -la "${OUT_DIR}"
