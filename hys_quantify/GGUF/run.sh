#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/ddc/新加卷/hys/hysnew3/model/Qwen3-VL-4B-GGUF}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/media/ddc/新加卷/hys/hysnew3/llama.cpp}"
OUTTYPE="${OUTTYPE:-f16}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
JOBS="${JOBS:-$(nproc)}"

F16_GGUF="${OUTPUT_DIR}/model-${OUTTYPE}.gguf"
QUANT_GGUF="${OUTPUT_DIR}/model-${QUANT_TYPE}.gguf"

echo "GGUF 量化 - Qwen3-VL"
echo "MODEL=${MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LLAMA_CPP_DIR=${LLAMA_CPP_DIR}"
echo "OUTTYPE=${OUTTYPE}"
echo "QUANT_TYPE=${QUANT_TYPE}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${LLAMA_CPP_DIR}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
else
    git -C "${LLAMA_CPP_DIR}" pull --ff-only
fi

cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DLLAMA_CURL=ON
cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j "${JOBS}"

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "未找到转换脚本: ${CONVERT_SCRIPT}" >&2
    exit 1
fi

if [ ! -x "${QUANTIZE_BIN}" ]; then
    echo "未找到量化程序: ${QUANTIZE_BIN}" >&2
    exit 1
fi

python "${CONVERT_SCRIPT}" \
    "${MODEL}" \
    --outfile "${F16_GGUF}" \
    --outtype "${OUTTYPE}"

"${QUANTIZE_BIN}" "${F16_GGUF}" "${QUANT_GGUF}" "${QUANT_TYPE}"

echo "完成: ${QUANT_GGUF}"
