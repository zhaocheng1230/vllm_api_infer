#!/bin/bash
# run_pipeline.sh - 启动 vLLM + Nginx 服务，并批量推理指定目录下的 jsonl 文件
#
# 用法:
#   bash run_pipeline.sh <input_dir> [output_dir] [nginx_port] [extra_args...]
#
# 环境变量:
#   MODEL_PATH      模型路径（必填）
#   MODEL_NAME      模型名，需与 --served-model-name 一致（必填）
#   TP_SIZE         每实例卡数，默认 2
#   MAX_MODEL_LEN   默认 131072
#   VLLM_BASE_PORT  vLLM 实例起始端口，默认 8000
#   NGINX_PORT      Nginx 监听端口，默认 1358
#   DATE_STR        任务日期目录，默认当天 $(date +%Y%m%d)
#   STARTUP_TIMEOUT 单实例启动等待时间（秒），默认 900
#   READY_TIMEOUT   等待 Nginx 入口就绪的时间（秒），默认 120
#   CONCURRENCY     推理并发数，默认 64
#   STOP_AFTER      推理结束后是否停止服务，0/1，默认 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析位置参数并保留额外参数
shift_count=0
INPUT_DIR="${1:-${SCRIPT_DIR}/datasets}"
[ $# -ge 1 ] && shift_count=1
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"
[ $# -ge 2 ] && shift_count=2
NGINX_PORT="${3:-${NGINX_PORT:-1358}}"
[ $# -ge 3 ] && shift_count=3
shift "$shift_count" || true

# 默认环境变量
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
DATE_STR="${DATE_STR:-$(date +%Y%m%d)}"
CONCURRENCY="${CONCURRENCY:-64}"
STOP_AFTER="${STOP_AFTER:-0}"
PYTHON="${PYTHON:-python}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"

SERVICE_READY=0
cleanup() {
    if [ "$STOP_AFTER" = "1" ] && [ "$SERVICE_READY" = "1" ]; then
        echo "[INFO] 停止 vLLM + Nginx 服务"
        bash "${SCRIPT_DIR}/start_server.sh" stop "${MODEL_NAME}" "${DATE_STR}"
    fi
}
trap cleanup EXIT

if [ -z "$INPUT_DIR" ]; then
    echo "[ERROR] 用法: bash $0 <input_dir> [output_dir] [nginx_port]"
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] 输入目录不存在: $INPUT_DIR"
    exit 1
fi
if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "[ERROR] 请设置环境变量 MODEL_PATH 和 MODEL_NAME"
    echo "  export MODEL_PATH=/path/to/model"
    echo "  export MODEL_NAME=MyModel"
    exit 1
fi

HEALTH_URL="http://localhost:${NGINX_PORT}/v1/models"
if curl -s --noproxy '*' "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[INFO] 检测到服务已在 ${NGINX_PORT} 运行，跳过启动"
    MODELS_JSON=$(curl -s --noproxy '*' "$HEALTH_URL" || true)
    if ! echo "$MODELS_JSON" | grep -qF "\"${MODEL_NAME}\""; then
        echo "[WARN] 服务已运行，但 /v1/models 中未找到模型 ${MODEL_NAME}，请确认 MODEL_NAME 是否匹配"
    fi
else
    echo "[INFO] 启动 vLLM 实例: model=${MODEL_NAME}, tp=${TP_SIZE}, date=${DATE_STR}"
    if ! bash "${SCRIPT_DIR}/start_server.sh" start "${MODEL_PATH}" "${MODEL_NAME}" "${TP_SIZE}" "${DATE_STR}" "${MAX_MODEL_LEN}"; then
        echo "[ERROR] vLLM 启动失败，请查看 logs/${DATE_STR}/${MODEL_NAME}/vllm_*.log"
        exit 1
    fi

    echo "[INFO] 启动 Nginx 反向代理: port=${NGINX_PORT}"
    bash "${SCRIPT_DIR}/start_server.sh" nginx "${NGINX_PORT}" "${MODEL_NAME}" "${DATE_STR}"
fi

echo "[INFO] 等待服务就绪: ${HEALTH_URL}"
for _ in $(seq 1 "$READY_TIMEOUT"); do
    if curl -s --noproxy '*' "$HEALTH_URL" >/dev/null 2>&1; then
        SERVICE_READY=1
        break
    fi
    sleep 1
done
if [ "$SERVICE_READY" != "1" ]; then
    echo "[ERROR] 服务在 ${READY_TIMEOUT}s 内未就绪，请检查 logs/${DATE_STR}/${MODEL_NAME}/ 与 nginx_error.log"
    exit 1
fi
echo "[INFO] 服务已就绪"

echo "[INFO] 开始推理: input=${INPUT_DIR}, output=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

"$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --host "http://localhost:${NGINX_PORT}/v1" \
    --model "$MODEL_NAME" \
    --concurrency "$CONCURRENCY" \
    "$@"
