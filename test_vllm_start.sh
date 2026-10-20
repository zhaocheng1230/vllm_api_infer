#!/bin/bash
# test_vllm_start.sh - 快速测试 vLLM Ascend 启动及 torch.compile 状态
# 用法: bash test_vllm_start.sh

set -euo pipefail

# ==================== 环境变量 ====================
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=ascend

export VLLM_NO_USAGE_STATS=1
export OTEL_SDK_DISABLED=true

export ASCEND_RT_VISIBLE_DEVICES="0"

# ==================== 模型配置 ====================
MODEL_PATH="/home/ma-user/work/dataset/****/common_opensource_models/Qwen/Qwen3.5-4B"
SERVED_MODEL_NAME="Qwen3.5-4B"
PORT=8000
TP_SIZE=1

# ==================== 启动服务 ====================
echo "========================================="
echo "  模型路径: ${MODEL_PATH}"
echo "  端口:     ${PORT}"
echo "  TP:       ${TP_SIZE}"
echo "  NPU设备:  ${ASCEND_RT_VISIBLE_DEVICES}"
echo "========================================="

python -u -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --async-scheduling \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
