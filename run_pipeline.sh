#!/bin/bash
# run_pipeline.sh - 启动 vLLM + Nginx 服务，并批量推理指定目录下的 jsonl 文件
#
# 用法:
#   bash run_pipeline.sh <input_path> [output_dir] [nginx_port] [extra_args...]
#   bash run_pipeline.sh --benchmarks gpqa,gsm8k_zh [extra_args...]
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
#   TASK_ID         可选，同一 benchmark 多任务并发时的目录消歧后缀
#
# 命令行参数:
#   --benchmarks <a,b,c>   按顺序跑哪些 benchmark（参数来自 configs/benchmarks.json）
#   --conf <file>          benchmark 配置文件，默认 <脚本目录>/configs/benchmarks.json
#   其余参数透传给 vllm_infer.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析位置参数并保留额外参数
# 只有不以 - 开头的参数才当位置参数，这样 --benchmarks 这种选项不会被误当成目录
shift_count=0
if [ $# -gt 0 ] && [[ "${1}" != -* ]]; then
    INPUT_PATH="$1"
    shift_count=1
    if [ $# -ge 2 ] && [[ "${2}" != -* ]]; then
        OUTPUT_DIR="$2"
        shift_count=2
        if [ $# -ge 3 ] && [[ "${3}" != -* ]]; then
            NGINX_PORT="$3"
            shift_count=3
        fi
    fi
fi
shift "$shift_count" || true

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
NGINX_PORT="${NGINX_PORT:-1358}"

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

# 从额外参数里摘出 --benchmarks / --conf，其余原样透传给 vllm_infer.py
BENCHMARKS=""
CONF_FILE="${SCRIPT_DIR}/configs/benchmarks.json"
CLIENT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --benchmarks=*) BENCHMARKS="${1#*=}"; shift ;;
        --benchmarks)   shift; BENCHMARKS="${1:-}"; shift ;;
        --conf=*)       CONF_FILE="${1#*=}"; shift ;;
        --conf)         shift; CONF_FILE="${1:-$CONF_FILE}"; shift ;;
        *)              CLIENT_ARGS+=("$1"); shift ;;
    esac
done

SERVICE_READY=0
SERVING_BENCH=""      # 当前正在提供服务实例的 benchmark（空 = 非 benchmark 模式）
cleanup() {
    if [ "$STOP_AFTER" = "1" ] && [ "$SERVICE_READY" = "1" ]; then
        echo "[INFO] 停止 vLLM + Nginx 服务"
        if [ -n "$SERVING_BENCH" ]; then
            bash "${SCRIPT_DIR}/start_server.sh" stop "${MODEL_NAME}" "${DATE_STR}" "${SERVING_BENCH}"
        else
            bash "${SCRIPT_DIR}/start_server.sh" stop "${MODEL_NAME}" "${DATE_STR}"
        fi
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
        echo "[ERROR] 服务在 ${READY_TIMEOUT}s 内未就绪，请检查 ${TASK_LOG_HINT:-logs/} 与 nginx_error.log"
        exit 1
    fi
    echo "[INFO] 服务已就绪"
}

start_service() {
    # $1 = benchmark 名（空 = 不分层）
    local bench="$1"
    local log_hint="logs/${DATE_STR}/${MODEL_NAME}"
    [ -n "$bench" ] && log_hint="${log_hint}/${bench}"

    echo "[INFO] 启动 vLLM 实例: model=${MODEL_NAME}, tp=${TP_SIZE}, max_model_len=${MAX_MODEL_LEN}, bench=${bench:-<none>}"
    if ! bash "${SCRIPT_DIR}/start_server.sh" start "${MODEL_PATH}" "${MODEL_NAME}" "${TP_SIZE}" \
            "${DATE_STR}" "${MAX_MODEL_LEN}" "${bench}"; then
        echo "[ERROR] vLLM 启动失败，请查看 ${log_hint}/vllm_*.log"
        exit 1
    fi

    echo "[INFO] 启动 Nginx 反向代理: port=${NGINX_PORT}"
    bash "${SCRIPT_DIR}/start_server.sh" nginx "${NGINX_PORT}" "${MODEL_NAME}" "${DATE_STR}" "${bench}"
    TASK_LOG_HINT="${log_hint}"
    wait_ready
}

# ============================================================================
# benchmark 模式：按顺序跑多个 benchmark，每个 benchmark 的参数来自配置文档
# ============================================================================
run_benchmarks() {
    local list="$1"
    [ -f "$CONF_FILE" ] || { echo "[ERROR] 配置文件不存在: $CONF_FILE"; exit 1; }

    # 客户端能力检查：vllm_infer.py 必须支持 --input。
    # 旧版只有 --input_dir，而 --input 是 --input_dir 的前缀，argparse 会把
    # --input 静默绑到 --input_dir 上 -> 拿文件路径去 glob -> 找不到文件却退出码 0。
    # 这���必须提前拦下来，否则会误报成「全部 benchmark 完成」。
    if ! "$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" --help 2>&1 | grep -qE '^[[:space:]]+--input[[:space:]]'; then
        echo "[ERROR] vllm_infer.py 版本过旧：不支持 --input 参数"
        echo "       旧版只有 --input_dir，argparse 会把 --input 静默绑过去，"
        echo "       导致「找不到文件」但退出码 0（假成功）。"
        echo "       请用最新版 vllm_infer.py 覆盖后再跑。"
        exit 1
    fi

    echo "================================================================"
    echo "  benchmark 模式"
    echo "  配置文件: $CONF_FILE"
    echo "  benchmark 列表: $list"
    echo "================================================================"

    # 调度计划：服务端参数指纹相同 -> 复用同一个服务实例，不重启
    local plan
    plan="$("$PYTHON" "${SCRIPT_DIR}/tools/conf2sh.py" "$CONF_FILE" --plan "$list")" || exit 1
    echo "[INFO] 调度计划:"
    echo "$plan" | sed 's/^/    /'
    echo ""

    local bench restart
    while read -r bench restart; do
        [ -n "$bench" ] || continue

        # 读取该 benchmark 的配置
        eval "$("$PYTHON" "${SCRIPT_DIR}/tools/conf2sh.py" "$CONF_FILE" "$bench")" || exit 1

        echo "----------------------------------------------------------------"
        echo "[BENCH] $bench"
        echo "  数据集: $INPUT_FILE"
        echo "  产出:   $OUTPUT_DIR"
        echo "  服务端: max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        echo "----------------------------------------------------------------"

        if [ "$restart" = "1" ] || [ -z "$SERVING_BENCH" ]; then
            # 服务端参数变了（或第一个）-> 停掉旧实例，起新实例
            if [ -n "$SERVING_BENCH" ]; then
                echo "[INFO] 停止上一个 benchmark 的服务: $SERVING_BENCH"
                bash "${SCRIPT_DIR}/start_server.sh" stop "${MODEL_NAME}" "${DATE_STR}" "${SERVING_BENCH}" || true
                SERVICE_READY=0
                sleep 3
            fi
            # 把配置里的服务端参数传给 start_server.sh（覆盖 TP 档位表）
            export MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
            start_service "$bench"
            SERVING_BENCH="$bench"
        else
            echo "[INFO] 服务端参数与上一个相同，复用当前服务实例（不重启）"
        fi

        if [ ! -f "$INPUT_FILE" ]; then
            echo "[ERROR] benchmark '$bench' 的数据集文件不存在: $INPUT_FILE"
            exit 1
        fi
        mkdir -p "$OUTPUT_DIR"

        # system_prompt_file -> --system_prompt
        SP_ARGS=()
        if [ -n "$SYSTEM_PROMPT_FILE" ]; then
            if [ ! -f "$SYSTEM_PROMPT_FILE" ]; then
                echo "[ERROR] benchmark '$bench' 的 system_prompt_file 不存在: $SYSTEM_PROMPT_FILE"
                exit 1
            fi
            SP_ARGS=(--system_prompt "$(cat "$SYSTEM_PROMPT_FILE")")
        fi

        echo "[INFO] 开始推理: bench=${bench}, input=${INPUT_FILE}, output=${OUTPUT_DIR}"
        if ! "$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" \
            --input "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --host "http://localhost:${NGINX_PORT}/v1" \
            --model "$MODEL_NAME" \
            --concurrency "$CONCURRENCY" \
            ${SP_ARGS[@]+"${SP_ARGS[@]}"} \
            ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}; then
            # 客户端失败必须中止：继续跑下一个 benchmark 没有意义，
            # 而且「部分跑完」很容易被误读成「全部完成」。
            echo "[ERROR] benchmark '$bench' 推理失败，已中止（上面的日志里有原因）"
            exit 1
        fi
        echo ""
    done <<< "$plan"

    echo "[INFO] 全部 benchmark 完成"
}

if [ -n "$BENCHMARKS" ]; then
    run_benchmarks "$BENCHMARKS"
    exit 0
fi

# ============================================================================
# 单任务模式（原有流程）
# ============================================================================
if [ -z "$INPUT_PATH" ]; then
    echo "[ERROR] 用法: bash $0 <input_path> [output_dir] [nginx_port]"
    echo "       <input_path> 可以是具体的 jsonl 文件，也可以是目录（目录扫描已弃用）"
    exit 1
fi
if [ ! -e "$INPUT_PATH" ]; then
    echo "[ERROR] 输入路径不存在: $INPUT_PATH"
    exit 1
fi

if curl -s --noproxy '*' "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[INFO] 检测到服务已在 ${NGINX_PORT} 运行，跳过启动"
    MODELS_JSON=$(curl -s --noproxy '*' "$HEALTH_URL" || true)
    if ! echo "$MODELS_JSON" | grep -qF "\"${MODEL_NAME}\""; then
        echo "[WARN] 服务已运行，但 /v1/models 中未找到模型 ${MODEL_NAME}，请确认 MODEL_NAME 是否匹配"
    fi
else
    start_service ""
fi

echo "[INFO] 开始推理: input=${INPUT_PATH}, output=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

"$PYTHON" "${SCRIPT_DIR}/vllm_infer.py" \
    --input "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --host "http://localhost:${NGINX_PORT}/v1" \
    --model "$MODEL_NAME" \
    --concurrency "$CONCURRENCY" \
    ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}
