#!/bin/bash
# run_pipeline.sh - 启动 vLLM + Nginx 服务，并批量推理 benchmark
#
# 用法:
#   bash run_pipeline.sh [output_dir] [max_model_len] [选项]
#
# 位置参数:
#   output_dir     结果输出目录，默认 <项目>/output/<日期>/<模型名>
#   max_model_len  服务端 max_model_len，默认 32768
#
# 选项:
#   --benchmarks <a,b,c>   跑哪些 benchmark。输入文件和客户端参数都从
#                          configs/benchmarks.json 读，不用在这里传
#   --input <file>         单文件模式（不走配置）
#   其余参数原样透传给 vllm_infer.py
#
# 任务目录: logs/<date>/<model>/<benchmarks 原始值>/
#   第三层直接用 --benchmarks 的原始值（比如 "gpqa,gsm8k_zh"）：
#   nginx.pid / nginx.conf / nginx_access.log 这些文件名都一样，
#   按原始值建目录，不同任务之间才不会互相覆盖、混在一起。
# 输出目录: output/<date>/<model>/             （产出文件名带 benchmark 名，不会重名）
#
# 环境变量:
#   MODEL_PATH      模型路径（必填）
#   MODEL_NAME      模型名，需与 --served-model-name 一致（必填）
#   TP_SIZE         每实例卡数，默认 2
#   VLLM_BASE_PORT  vLLM 实例起始端口，默认 8000
#   NGINX_PORT      Nginx 监听端口，默认 1358
#   DATE_STR        任务日期目录，默认当天 $(date +%Y%m%d)
#   STARTUP_TIMEOUT 单实例启动等待时间（秒），默认 900
#   READY_TIMEOUT   等待 Nginx 入口就绪的时间（秒），默认 120
#   CONCURRENCY     推理并发数，默认 64
#   STOP_AFTER      推理结束后是否停止服务，0/1，默认 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 位置参数：只有不以 - 开头的参数才算位置参数，--benchmarks 这类选项不会被误吃。
# max_model_len 必须是纯数字，避免把某个选项的值当成位置参数。
# ---------------------------------------------------------------------------
OUTPUT_DIR=""
MAX_MODEL_LEN=""
shift_count=0
if [ $# -gt 0 ] && [[ "${1}" != -* ]]; then
    OUTPUT_DIR="$1"
    shift_count=1
    if [ $# -ge 2 ] && [[ "${2}" != -* ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
        MAX_MODEL_LEN="$2"
        shift_count=2
    fi
fi
shift "$shift_count" || true

# 默认环境变量
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
TP_SIZE="${TP_SIZE:-2}"
DATE_STR="${DATE_STR:-$(date +%Y%m%d)}"
CONCURRENCY="${CONCURRENCY:-64}"
STOP_AFTER="${STOP_AFTER:-0}"
PYTHON="${PYTHON:-python}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"
NGINX_PORT="${NGINX_PORT:-1358}"

# 输出目录默认 <项目>/output/<日期>/<模型名>
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${DATE_STR}/${MODEL_NAME}}"
# 服务端 max_model_len 默认 32768
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# 从额外参数里摘出 --benchmarks / --input，其余原样透传给 vllm_infer.py
BENCHMARKS=""
INPUT_FILE=""
CLIENT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --benchmarks=*) BENCHMARKS="${1#*=}"; shift ;;
        --benchmarks)   shift; BENCHMARKS="${1:-}"; shift ;;
        --input=*)      INPUT_FILE="${1#*=}"; shift ;;
        --input)        shift; INPUT_FILE="${1:-}"; shift ;;
        *)              CLIENT_ARGS+=("$1"); shift ;;
    esac
done

SERVICE_READY=0
SERVING_BENCH=""      # 当前正在服务的任务目录名（第三层）

stop_service() {
    [ -n "$SERVING_BENCH" ] || return 0
    echo "[INFO] 停止服务: ${SERVING_BENCH}"
    bash "${SCRIPT_DIR}/start_server.sh" stop "${MODEL_NAME}" "${DATE_STR}" "${SERVING_BENCH}" || true
    SERVING_BENCH=""
    SERVICE_READY=0
    sleep 3
}

cleanup() {
    if [ "$STOP_AFTER" = "1" ] && [ "$SERVICE_READY" = "1" ]; then
        echo "[INFO] 停止 vLLM + Nginx 服务"
        stop_service
    fi
}
trap cleanup EXIT

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "[ERROR] 请设置环境变量 MODEL_PATH 和 MODEL_NAME"
    echo "  export MODEL_PATH=/path/to/model"
    echo "  export MODEL_NAME=MyModel"
    exit 1
fi

HEALTH_URL="http://localhost:${NGINX_PORT}/v1/models"

wait_ready() {
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
}

# $1 = 任务目录第三层（benchmark 模式用 --benchmarks 原始值，单文件模式用输入文件名）
start_service() {
    local bench="$1"
    echo "[INFO] 启动 vLLM 实例: model=${MODEL_NAME}, tp=${TP_SIZE}, max_model_len=${MAX_MODEL_LEN}, task=${bench}"
    if ! bash "${SCRIPT_DIR}/start_server.sh" start "${MODEL_PATH}" "${MODEL_NAME}" "${TP_SIZE}" \
            "${DATE_STR}" "${MAX_MODEL_LEN}" "${bench}"; then
        echo "[ERROR] vLLM 启动失败，请查看 logs/${DATE_STR}/${MODEL_NAME}/${bench}/vllm_*.log"
        exit 1
    fi

    echo "[INFO] 启动 Nginx 反向代理: port=${NGINX_PORT}"
    bash "${SCRIPT_DIR}/start_server.sh" nginx "${NGINX_PORT}" "${MODEL_NAME}" "${DATE_STR}" "${bench}"
    SERVING_BENCH="$bench"
    wait_ready
}

# 客户端能力检查：用源码里有没有这个参数来判断，不去执行客户端
# （避免依赖缺失时报错误导）。--input 必须单独查：旧版只有 --input_dir，
# 而 --input 是 --input_dir 的前缀，argparse 会把它静默绑过去。
for opt in --input --benchmarks; do
    if ! grep -q -- "\"${opt}\"" "${SCRIPT_DIR}/vllm_infer.py" 2>/dev/null; then
        echo "[ERROR] vllm_infer.py 版本过旧：不支持 ${opt} 参数"
        echo "       请用最新版 vllm_infer.py 覆盖后再跑。"
        exit 1
    fi
done

# ============================================================================
# benchmark 模式：一次起服务，逐个 benchmark 跑客户端
# ============================================================================
if [ -n "$BENCHMARKS" ]; then
    echo "================================================"
    echo " benchmark 模式"
    echo " 输出目录:     $OUTPUT_DIR"
    echo " max_model_len: $MAX_MODEL_LEN"
    echo " benchmark:     $BENCHMARKS"
    echo " 任务目录:     logs/${DATE_STR}/${MODEL_NAME}/${BENCHMARKS}"
    echo "================================================"

    # 任务目录第三层 = --benchmarks 的原始值，整串都用，不拆。
    # 服务只起一次；客户端拿到这串后按顺序逐个 benchmark 跑推理。
    start_service "$BENCHMARKS"

    if ! "$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" \
        --benchmarks "$BENCHMARKS" \
        --output_dir "$OUTPUT_DIR" \
        --max_model_len "$MAX_MODEL_LEN" \
        --host "http://localhost:${NGINX_PORT}/v1" \
        --model "$MODEL_NAME" \
        --concurrency "$CONCURRENCY" \
        ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}; then
        # 客户端失败必须中止：「部分跑完」很容易被误读成「全部完成」。
        echo "[ERROR] 推理失败，已中止（上面的日志里有原因）"
        exit 1
    fi

    echo "[INFO] 全部 benchmark 完成"
    exit 0
fi

# ============================================================================
# 单文件模式
# ============================================================================
if [ -z "$INPUT_FILE" ]; then
    echo "[ERROR] 用法: bash $0 [output_dir] [max_model_len] --benchmarks <a,b,c>"
    echo "       或:   bash $0 [output_dir] [max_model_len] --input <file.jsonl>"
    exit 1
fi
if [ ! -e "$INPUT_FILE" ]; then
    echo "[ERROR] 输入路径不存在: $INPUT_FILE"
    exit 1
fi

if curl -s --noproxy '*' "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[INFO] 检测到服务已在 ${NGINX_PORT} 运行，跳过启动"
    MODELS_JSON=$(curl -s --noproxy '*' "$HEALTH_URL" || true)
    if ! echo "$MODELS_JSON" | grep -qF "\"${MODEL_NAME}\""; then
        echo "[WARN] 服务已运行，但 /v1/models 中未找到模型 ${MODEL_NAME}，请确认 MODEL_NAME 是否匹配"
    fi
else
    # 单文件模式没有 benchmark 名，用输入文件名当任务目录的第三层，
    # 这样不同任务（不同输入文件）也不会互相覆盖日志。
    start_service "$(basename "${INPUT_FILE}" | sed 's/\.[^.]*$//')"
fi

echo "[INFO] 开始推理: input=${INPUT_FILE}, output=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

if ! "$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_model_len "$MAX_MODEL_LEN" \
    --host "http://localhost:${NGINX_PORT}/v1" \
    --model "$MODEL_NAME" \
    --concurrency "$CONCURRENCY" \
    ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}; then
    echo "[ERROR] 推理失败，已中止（上面的日志里有原因）"
    exit 1
fi
