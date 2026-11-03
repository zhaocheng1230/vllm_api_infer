#!/bin/bash
# start_server.sh - vLLM 多实例启动 + Nginx 反向代理管理（Ascend NPU / NVIDIA GPU 通用）
# 用法:
#   bash start_server.sh start <model_path> <model_name> <tp_size> [date_str] [max_model_len]
#   bash start_server.sh nginx [listen_port] [model_name] [date_str]
#   bash start_server.sh status [model_name] [date_str]
#   bash start_server.sh stop   <model_name> [date_str]
#
# 每个任务独立存放运行时状态: logs/<date_str>/<model_name>/
#   date_str 默认当天日期 $(date +%Y%m%d)，可通过第 4 个参数或 DATE_STR 覆盖
#
# 可选环境变量:
#   DATE_STR=YYYYMMDD             # 默认任务日期
#   DEVICE_BACKEND=auto|npu|gpu   # 默认 auto，按 npu-smi / nvidia-smi 自动识别
#   DEVICE_COUNT=N                # 覆盖自动检测到的设备数量
#   VLLM_GPU_MEM_UTIL=0.90        # GPU 显存利用率（NPU 使用 TP 档位内置值）
#   VLLM_DTYPE=auto|bfloat16      # 覆盖 dtype（GPU 默认 auto，NPU 默认 bfloat16）
#   STARTUP_TIMEOUT=900           # 单实例启动等待时间（秒）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LOG_DIR="${SCRIPT_DIR}/logs"
DEFAULT_DATE_STR="${DATE_STR:-$(date +%Y%m%d)}"

# 任务相关路径，由 set_task_paths() 填充
DATE_STR="$DEFAULT_DATE_STR"
MODEL_DIR=""
BENCH_DIR=""
TASK_LOG_DIR=""
PID_FILE=""
NGINX_PID=""
NGINX_CONF=""
NGINX_ERROR_LOG=""
NGINX_ACCESS_LOG=""

# 默认后端：优先读取 DEVICE_BACKEND，未设置则自动检测，默认值为 auto，可选项npu|gpu
DEFAULT_BACKEND="${DEVICE_BACKEND:-auto}"
# 单个实例启动等待时间（秒）；冷启动加载大模型可能超过 5 分钟
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

# ========================== 固定的服务端参数 ==========================
# 全部来自 910B4 + Qwen3.5-4B 实测，固化在这里。
# 不要做成可调项，也不要挪到 benchmark 配置里 —— 实测依据见 docs/性能调优结论.md
#
#   FULL_DECODE_ONLY   吞吐 +1.2%、TTFT p95 −34%、e2e p95 −6.5%
#   max_num_seqs=128   2 卡实测峰值并发 256 = 2 实例 × 128
#   关闭编译缓存       Qwen3.5 的 GDN 门控 layernorm 是 Triton kernel，
#                      编译产物不可跨进程序列化，换进程加载必崩
#                      （vLLM 已知问题：它的编译器缓存并不总是完全正确）
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
export VLLM_DISABLE_COMPILE_CACHE=1

# ========================== 工具函数 ==========================

log_info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
log_ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
log_err()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

set_task_paths() {
    local date_str="${1:-$DEFAULT_DATE_STR}"
    local model_name="${2:-}"
    local bench_name="${3:-${BENCHMARK:-}}"
    [ -n "$model_name" ] || { log_err "缺少 model_name"; exit 1; }

    DATE_STR="$date_str"
    MODEL_DIR="${model_name//\//_}"
    BENCH_DIR="${bench_name//\//_}"

    # 任务目录：logs/<date>/<model>/[<benchmark>][__<TASK_ID>]
    # 第三层是 benchmark 名（单文件模式用输入文件名）。
    # 必须分这一层：nginx.pid / nginx.conf / nginx_access.log / vllm_*.log
    # 这些文件名全都一样，放同一个目录里互相覆盖，根本分不清是哪个任务的。
    # TASK_ID 用于「同一个 benchmark 跑多次」的消歧。
    local sub=""
    [ -n "$BENCH_DIR" ] && sub="$BENCH_DIR"
    if [ -n "${TASK_ID:-}" ]; then
        sub="${sub:+${sub}__}${TASK_ID//\//_}"
    fi
    if [ -n "$sub" ]; then
        TASK_LOG_DIR="${BASE_LOG_DIR}/${DATE_STR}/${MODEL_DIR}/${sub}"
    else
        TASK_LOG_DIR="${BASE_LOG_DIR}/${DATE_STR}/${MODEL_DIR}"
    fi

    PID_FILE="${TASK_LOG_DIR}/vllm.pids"
    NGINX_PID="${TASK_LOG_DIR}/nginx.pid"
    NGINX_CONF="${TASK_LOG_DIR}/nginx.conf"
    NGINX_ERROR_LOG="${TASK_LOG_DIR}/nginx_error.log"
    NGINX_ACCESS_LOG="${TASK_LOG_DIR}/nginx_access.log"
}

# 把 <BASE_LOG_DIR>/<date>/<model>[/<benchmark>] 拆成三段
parse_task_rel() {
    local rel="${1#$BASE_LOG_DIR/}"
    local d="" m="" b=""
    IFS='/' read -r d m b <<< "$rel"
    echo "$d $m $b"
}

latest_task_dir() {
    [ -d "$BASE_LOG_DIR" ] || return 1
    # 优先带 benchmark 的任务目录（深度 3），没有再退回旧结构（深度 2）
    local d
    d=$(find "$BASE_LOG_DIR" -mindepth 3 -maxdepth 3 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$d" ] && { echo "$d"; return 0; }
    find "$BASE_LOG_DIR" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

check_port_available() {
    ! ss -tlnp 2>/dev/null | grep -q ":${1} " && \
    ! netstat -tlnp 2>/dev/null | grep -q ":${1} "
}

wait_for_port() {
    local port=$1 timeout=${2:-300} elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -s --noproxy '*' "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

detect_vllm_ports() {
    local pid_file="${1:-$PID_FILE}"
    local ports=""

    if [ -f "$pid_file" ]; then
        while read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                # 用 ss + awk 提取端口，避免 ps 截断和 grep -P 兼容性问题
                local p
                p=$(ss -tlnp 2>/dev/null | awk -v pid="pid=${pid}," '$0 ~ pid {split($4, a, ":"); print a[length(a)]}' | head -1)
                # ss 不可用 / 看不到别的用户的进程时，回退到 ps
                if [ -z "$p" ]; then
                    p=$(ps -p "$pid" -o args= 2>/dev/null | grep -oP '(?<=--port )\d+' | head -1 || true)
                fi
                [ -n "$p" ] && ports="${ports}${p}"$'\n'
            fi
        done < "$pid_file"
    fi

    printf "%s" "$ports" | sort -n | uniq
}

# 自动检测或显式校验后端，stdout 输出 npu|gpu
resolve_backend() {
    local backend
    backend=$(echo "$DEFAULT_BACKEND" | tr '[:upper:]' '[:lower:]')

    if [ "$backend" = "auto" ]; then
        if command -v npu-smi >/dev/null 2>&1 && npu-smi info -l 2>/dev/null | grep -q "NPU ID"; then
            backend="npu"
        elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q 'GPU [0-9]'; then
            backend="gpu"
        else
            log_err "未检测到 NPU 或 GPU，请设置 DEVICE_BACKEND=npu|gpu"
            exit 1
        fi
    elif [ "$backend" != "npu" ] && [ "$backend" != "gpu" ]; then
        log_err "DEVICE_BACKEND 必须是 auto|npu|gpu，当前: $DEFAULT_BACKEND"
        exit 1
    fi

    echo "$backend"
}

# 根据后端统计可用设备数
backend_device_count() {
    local backend="$1"
    case "$backend" in
        npu) npu-smi info -l 2>/dev/null | grep -c "NPU ID" || true ;;
        gpu) nvidia-smi -L 2>/dev/null | grep -c 'GPU [0-9]' || true ;;
        *)   echo 0 ;;
    esac
}

# 根据后端和可见性环境变量生成设备 ID 列表
# 结果写入全局变量 DEVICE_TOTAL 和 DEVICE_IDS
device_ids_for_backend() {
    local backend="$1"
    local detected
    detected=$(backend_device_count "$backend")

    if [ -n "${DEVICE_COUNT:-}" ] && ! [[ "$DEVICE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        log_err "DEVICE_COUNT 必须是正整数: $DEVICE_COUNT"
        exit 1
    fi
    if [ -z "${DEVICE_COUNT:-}" ] && { ! [[ "$detected" =~ ^[0-9]+$ ]] || [ "$detected" -le 0 ]; }; then
        log_err "设备数量无效: $detected (backend=$backend)，可通过 DEVICE_COUNT 覆盖"
        exit 1
    fi

    local -a ids=()
    local visible=""
    case "$backend" in
        gpu) visible="${CUDA_VISIBLE_DEVICES:-}" ;;
        npu) visible="${ASCEND_RT_VISIBLE_DEVICES:-}" ;;
    esac
    visible=$(echo "$visible" | tr -d ' ')

    local d
    if [ -n "$visible" ] && [ "$visible" != "all" ]; then
        IFS=',' read -ra ids <<< "$visible"
    else
        local fallback="$detected"
        if ! [[ "$fallback" =~ ^[0-9]+$ ]] || [ "$fallback" -le 0 ]; then
            fallback="${DEVICE_COUNT:-0}"
        fi
        for ((d = 0; d < fallback; d++)); do
            ids+=("$d")
        done
    fi

    local total=${#ids[@]}
    if [ -n "${DEVICE_COUNT:-}" ]; then
        if [ "$DEVICE_COUNT" -gt "$total" ]; then
            log_err "DEVICE_COUNT=$DEVICE_COUNT 超过可用设备数 $total"
            exit 1
        fi
        total="$DEVICE_COUNT"
        ids=("${ids[@]:0:$total}")
    fi

    if [ "$total" -le 0 ]; then
        log_err "无可用设备 (backend=$backend)"
        exit 1
    fi

    DEVICE_TOTAL=$total
    DEVICE_IDS=("${ids[@]}")
}

# 设置后端相关的进程级环境变量
export_backend_env() {
    local backend="$1"
    local devices="$2"

    export VLLM_NO_USAGE_STATS=1
    export OTEL_SDK_DISABLED=true
    # INFO 才会输出周期性吞吐 / KV cache / prefix cache 统计；WARNING 可让日志更安静
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

    case "$backend" in
        npu)
            export ASCEND_RT_VISIBLE_DEVICES="${devices}"
            
            # 👇 新增：从官方脚本借鉴的通用优化
            export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
            export OMP_NUM_THREADS=1
            export TASK_QUEUE_ENABLE=1

            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export VLLM_TARGET_DEVICE=ascend
            ;;
        gpu)
            export CUDA_VISIBLE_DEVICES="${devices}"
            ;;
    esac
}

# 根据后端和 TP 返回显存利用率
backend_memory_util() {
    local backend="$1"
    local tp="$2"

    if [ "$backend" = "gpu" ]; then
        echo "${VLLM_GPU_MEM_UTIL:-0.90}"
        return
    fi

    # 显存利用率按 TP 收窄：
    # max_num_batched_tokens 从 16384 提到 32768 后，batch buffer 占用翻倍，
    # 显存利用率必须相应降下来，否则容易 OOM（启动阶段就起不来）。
    case "$tp" in
        1) echo 0.85 ;;
        2) echo 0.85 ;;
        4) echo 0.82 ;;
        8) echo 0.80 ;;
        *) echo 0.80 ;;
    esac
}

# 根据后端返回默认 dtype
backend_dtype() {
    local backend="$1"

    if [ -n "${VLLM_DTYPE:-}" ]; then
        echo "$VLLM_DTYPE"
    elif [ "$backend" = "gpu" ]; then
        echo "auto"
    else
        echo "bfloat16"
    fi
}

# --- 服务端档位（按 TP 自动选，不是给人调的开关）---
# TP=1 是实测配置（2 卡 → 2 实例 → 峰值并发 256）。
# TP>1 单实例吃更多显存，序列数相应收、partial prefill 关掉。
#
# TP=2 的 max_num_seqs 从 128 提到 256 的依据（2026-11-02 实测，4 实例 / 8 卡）：
#   并发 256 -> 每实例 64  = max_num_seqs 的 50% -> 有余量，TTFT p50 11.8s
#   并发 512 -> 每实例 128 = max_num_seqs 的 100% -> 打满，新请求排队，
#                                     TTFT p50 升到 89.6s、端到端 p50 237s->403s
#   而实测 KV cache 峰值只有 22~32% -> 显存根本不是限制，限制的是这个数。
#   提到 256 后并发 512 时每实例占 50%，排队消失。
#
# max_num_batched_tokens 试过 32768，结论是**负收益，已退回 16384**（2026-11-03 实测）：
#   16384: 墙钟 1413s | 平均吞吐 2730 | 峰值/实例 2000 | TTFT p50 90.6s
#   32768: 墙钟 1515s | 平均吞吐 2564 | 峰值/实例 2430 | TTFT p50 93.6s
#   峰值涨了 21% 但平均吞吐跌了 6%、墙钟长了 7%、TTFT 一点没降 ——
#   batched_tokens 越大，prefill 越能抢 decode 的算力（vLLM 经典现象）。
#
# TTFT ~90 秒的本质是「prefill 总量 ÷ prefill 吞吐」，与 batched_tokens 无关：
#   每实例 128 个请求 × ~2000 prompt token = 256K token prefill
#   ÷ ~3000 tok/s/实例 ≈ 85 秒 ✓
# 要降 TTFT 只有三条路：降并发 / prefix caching / 缩短 prompt。
set_server_params() {
    # 档位依据（910B4 32GB 实测）：
    #   TP=1 实测：2 卡 2 实例，并发 256 是吞吐拐点（2061 tok/s），512 掉头
    #   TP=2 实测：8 卡 4 实例，并发 256 最优（TTFT p50 11.8s / 端到端 p50 237s）
    #
    # 客户端并发的取法：**每实例 64 个**（TP=1 因为 batched_tokens 更大，可以到 128）
    #   每实例只占 max_num_seqs 的 50% -> 新请求能立刻进来，不会排队
    #   （并发占满 max_num_seqs 时 TTFT 会从 11.8s 飙到 90s）
    case "$1" in
        1) MAX_NUM_SEQS=128; MAX_NUM_BATCHED_TOKENS=32768; TP_EXTRA_ARGS=() ;;
        2) MAX_NUM_SEQS=256; MAX_NUM_BATCHED_TOKENS=16384; TP_EXTRA_ARGS=(--max-num-partial-prefills 1) ;;
        4) MAX_NUM_SEQS=128; MAX_NUM_BATCHED_TOKENS=32768; TP_EXTRA_ARGS=(--max-num-partial-prefills 1) ;;
        8) MAX_NUM_SEQS=128; MAX_NUM_BATCHED_TOKENS=65536; TP_EXTRA_ARGS=(--max-num-partial-prefills 1) ;;
        *) MAX_NUM_SEQS=128; MAX_NUM_BATCHED_TOKENS=16384; TP_EXTRA_ARGS=() ;;
    esac
}

# ========================== start 命令 ==========================

do_start() {
    local MODEL_PATH="${1:-}"
    local MODEL_NAME="${2:-}"
    local TP_SIZE="${3:-2}"
    local DATE_STR_ARG="${4:-$DEFAULT_DATE_STR}"
    local MAX_MODEL_LEN="${5:-${MAX_MODEL_LEN:-32768}}"
    local BENCHMARK_ARG="${6:-${BENCHMARK:-}}"
    local BASE_PORT="${VLLM_BASE_PORT:-8000}"

    [ -z "$MODEL_PATH" ] && { log_err "缺少 model_path"; exit 1; }
    [ -z "$MODEL_NAME" ] && { log_err "缺少 model_name"; exit 1; }
    if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        log_err "tp_size 必须是正整数: $TP_SIZE"
        exit 1
    fi

    set_task_paths "$DATE_STR_ARG" "$MODEL_NAME" "$BENCHMARK_ARG"

    # 1) 后端识别
    local BACKEND
    BACKEND=$(resolve_backend)

    # 2) 设备列表解析
    local DEVICE_TOTAL
    local -a DEVICE_IDS=()
    device_ids_for_backend "$BACKEND"

    # 每实例推荐的客户端并发（实测：每实例占 max_num_seqs 的 50% 时不排队，
    # 占满时 TTFT 从 11.8s 飙到 90s）。TP=1 的 batched_tokens 更大，可以到 128。
    local RECOMMENDED_CONCURRENCY_PER_INSTANCE=64
    [ "$TP_SIZE" = "1" ] && RECOMMENDED_CONCURRENCY_PER_INSTANCE=128

    local LOOP_COUNT=$((DEVICE_TOTAL / TP_SIZE))
    [ "$LOOP_COUNT" -le 0 ] && { log_err "设备不足: total=$DEVICE_TOTAL, per_model=$TP_SIZE"; exit 1; }

    mkdir -p "$TASK_LOG_DIR"

    if [ -f "$PID_FILE" ]; then
        local alive_pids=()
        while read -r pid; do
            [ -z "$pid" ] && continue
            kill -0 "$pid" 2>/dev/null && alive_pids+=("$pid")
        done < "$PID_FILE"
        if [ ${#alive_pids[@]} -gt 0 ]; then
            log_err "同一任务已有 vLLM 实例在运行 (PID: ${alive_pids[*]})"
            log_err "请先停止: bash $0 stop \"${MODEL_PATH}\" \"${MODEL_NAME}\" ${TP_SIZE} ${DATE_STR}"
            exit 1
        fi
    fi
    > "$PID_FILE"

    # 3) 服务端参数（固定值 + 按 TP 自动选档，不接受外部覆盖）
    local GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
    local -a TP_EXTRA_ARGS=()
    set_server_params "$TP_SIZE"
    GPU_MEM_UTIL=$(backend_memory_util "$BACKEND" "$TP_SIZE")

    local DTYPE
    DTYPE=$(backend_dtype "$BACKEND")

    # 4) 模型特殊参数
    #    注意：--language-model-only 不能关。实测（910B4 + Qwen3.5-4B）关掉后
    #    吞吐掉 31%、e2e p95 涨 217%，见 docs/性能调优结论.md
    local TOOL_CALL_PARSER="hermes"
    local -a MODEL_EXTRA_ARGS=()

    if echo "$MODEL_NAME" | grep -iqE "Qwen3[._][5-9]"; then
        TOOL_CALL_PARSER="qwen3_coder"
        MODEL_EXTRA_ARGS=(--language-model-only)
    elif echo "$MODEL_NAME" | grep -iqE "MiniCPM5|minicpm5"; then
        TOOL_CALL_PARSER="minicpm5"
        MODEL_EXTRA_ARGS=(--tool-call-parser-plugin "${MINICPM5_PLUGIN:-/path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py}")
    fi

    local TIMESTAMP
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local started_ports=()

    echo "========================================="
    echo "  模型:       $MODEL_NAME  ($MODEL_PATH)"
    echo "  任务目录:   $TASK_LOG_DIR"
    echo "  后端:       $BACKEND   设备 $DEVICE_TOTAL 张   每模型 $TP_SIZE 卡   实例 $LOOP_COUNT 个"
    echo "  设备列表:   $(IFS=,; echo "${DEVICE_IDS[*]:0:$DEVICE_TOTAL}")"
    echo "  端口:       $BASE_PORT ~ $((BASE_PORT + LOOP_COUNT - 1))"
    echo "  服务端:     max-model-len=$MAX_MODEL_LEN   max-num-seqs=$MAX_NUM_SEQS   batched-tokens=$MAX_NUM_BATCHED_TOKENS"
    echo "  显存/dtype: $GPU_MEM_UTIL / $DTYPE"
    # 服务端总容量（所有实例的 max_num_seqs 之和）只是理论上限，
    # 不是建议的客户端并发。实测：并发占满总容量时 TTFT 会从 11.8s 飙到 90s。
    echo "  并发建议:   $((LOOP_COUNT * RECOMMENDED_CONCURRENCY_PER_INSTANCE))"
    echo "             （每实例 $RECOMMENDED_CONCURRENCY_PER_INSTANCE 个；"
    echo "              服务端总容量 $((LOOP_COUNT * MAX_NUM_SEQS))，不建议占满）"
    echo "========================================="

    for i in $(seq 0 $((LOOP_COUNT - 1))); do
        local CURRENT_PORT=$((BASE_PORT + i))
        local DEVICE_START=$((i * TP_SIZE))
        local CURRENT_DEVICES
        CURRENT_DEVICES=$(IFS=,; echo "${DEVICE_IDS[*]:$DEVICE_START:$TP_SIZE}")

        if ! check_port_available "$CURRENT_PORT"; then
            log_warn "端口 $CURRENT_PORT 已被占用，跳过该实例"
            continue
        fi

        local LOG_FILE="${TASK_LOG_DIR}/vllm_${TIMESTAMP}_${CURRENT_PORT}.log"

        echo "正在启动任务 $i: 设备 [${CURRENT_DEVICES}], 端口 [${CURRENT_PORT}], 日志 [${LOG_FILE}]..."
        echo "  编译缓存: ${VLLM_CACHE_BASE:-$HOME/.cache/vllm_multi}/${MODEL_DIR}/${CURRENT_PORT}"

        # vllm serve 的完整命令行。
        # 固定参数直接写死；只有「按 TP / 按模型变化」的少数几项走 SERVE_OPT，
        # 最后一次性追加 —— 空参数会让 argparse 直接报 "unrecognized arguments: " 拒绝启动。
        local -a SERVE_CMD=(
            vllm serve "${MODEL_PATH}"
            --host 0.0.0.0
            --port "${CURRENT_PORT}"
            --served-model-name "${MODEL_NAME}"
            --tensor-parallel-size "${TP_SIZE}"
            --dtype "${DTYPE}"
            --gpu-memory-utilization "${GPU_MEM_UTIL}"
            --max-model-len "${MAX_MODEL_LEN}"
            --max-num-seqs "${MAX_NUM_SEQS}"
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
            --trust-remote-code
            --async-scheduling
            --enable-auto-tool-choice
            --tool-call-parser "${TOOL_CALL_PARSER}"
            # 不开 prefix caching：
            # 实测 vLLM 0.23.0 + vllm-ascend 默认就是 enable_prefix_caching=False。
            # 我们的场景是单轮批量推理，没有多轮对话那种大段公共上下文；
            # 共享前缀只有 chat template + system_prompt（约 50~75 token），
            # 而每条请求 prefill 约 2000 token —— 收益不到 1%，
            # 却要付 prefix caching 的记账开销，还有混合 Mamba 模型的支持风险。
            # 如果将来是多轮对话（公共上下文几千 token），再打开才有意义。
        )
        local -a SERVE_OPT=()
        if [ ${#MODEL_EXTRA_ARGS[@]} -gt 0 ]; then
            SERVE_OPT+=("${MODEL_EXTRA_ARGS[@]}")
        fi
        if [ ${#TP_EXTRA_ARGS[@]} -gt 0 ]; then
            SERVE_OPT+=("${TP_EXTRA_ARGS[@]}")
        fi
        if [ ${#SERVE_OPT[@]} -gt 0 ]; then
            SERVE_CMD+=("${SERVE_OPT[@]}")
        fi
        SERVE_CMD+=(--compilation-config "${COMPILATION_CONFIG}")
        # 打印完整命令行：vLLM 有些参数会被静默忽略，日志里一搜就知道传没传进去
        echo "  serve 命令: ${SERVE_CMD[*]}"

        # 在子 shell 中设置设备相关环境变量，exec 后 PID 即为 vllm 进程
        (
            export_backend_env "$BACKEND" "$CURRENT_DEVICES"

            # 每个实例用独立的编译缓存目录。
            # vLLM 的缓存路径是 <VLLM_CACHE_ROOT>/torch_compile_cache/<hash>/rank_{rank}_{dp_rank}，
            # 多实例同模型同配置时 rank 都是 0_0 -> 会写同一个文件。
            # 注：vLLM 自己的编译缓存已全局关闭（见文件头），这里只是隔离
            # torch.compile / Triton 自己的缓存目录，避免多实例并发写冲突。
            export VLLM_CACHE_ROOT="${VLLM_CACHE_BASE:-$HOME/.cache/vllm_multi}/${MODEL_DIR}/${CURRENT_PORT}"
            export TRITON_CACHE_DIR="${VLLM_CACHE_ROOT}/triton"
            mkdir -p "${VLLM_CACHE_ROOT}"

            exec "${SERVE_CMD[@]}" \
                > "${LOG_FILE}" 2>&1
        ) &

        echo $! >> "$PID_FILE"
        started_ports+=("$CURRENT_PORT")
    done

    [ ${#started_ports[@]} -eq 0 ] && { log_err "没有成功启动任何实例"; exit 1; }

    echo ""
    echo "所有实例已发起启动，开始健康检查 (${#started_ports[@]} 个端口)..."
    echo "-----------------------------------------"

    local ready_count=0
    for port in "${started_ports[@]}"; do
        if wait_for_port "$port" "$STARTUP_TIMEOUT"; then
            echo "  ✅ 端口 $port 已就绪"
            ready_count=$((ready_count + 1))
        else
            echo "  ❌ 端口 $port 超时未就绪"
        fi
    done

    echo ""
    if [ $ready_count -eq ${#started_ports[@]} ]; then
        echo "========================================="
        log_ok "启动完成！${ready_count}/${#started_ports[@]} 个实例就绪"
        echo "  任务目录: ${TASK_LOG_DIR}"
        echo "  查看日志: tail -f ${TASK_LOG_DIR}/vllm_*.log"
        echo "  查看状态: bash $0 status \"${MODEL_NAME}\" ${DATE_STR}"
        echo "  停止服务: bash $0 stop \"${MODEL_NAME}\" ${DATE_STR}"
        echo "  健康检查: curl --noproxy '*' http://localhost:${started_ports[0]}/health"
        echo "========================================="
    else
        log_err "部分实例未就绪: ${ready_count}/${#started_ports[@]}（等待 ${STARTUP_TIMEOUT}s）"
        log_err "请查看 ${TASK_LOG_DIR}/vllm_*.log"
        log_err "确认失败后可执行: bash $0 stop \"${MODEL_NAME}\" ${DATE_STR}"
        exit 1
    fi

    # 输出端口列表供外部脚本使用
    echo "检测到 ${#started_ports[@]} 个 vLLM 实例端口: ${started_ports[*]}"
}

# ========================== NGINX 配置生成 ==========================
do_nginx() {
    local listen_port="${1:-1358}"
    local model_arg="${2:-}"
    local date_arg="${3:-}"
    local bench_arg="${4:-${BENCHMARK:-}}"
    local template_file="${SCRIPT_DIR}/nginx_vllm.conf.template"

    if ! command -v nginx >/dev/null 2>&1; then
        log_err "未找到 nginx 命令，请先安装 Nginx 或将其加入 PATH"
        log_err "  Debian/Ubuntu: sudo apt-get install -y nginx"
        log_err "  RHEL/CentOS:   sudo yum install -y nginx"
        exit 1
    fi

    [ -f "$template_file" ] || { log_err "模板文件不存在: ${template_file}"; exit 1; }

    # 定位任务：显式参数优先，否则取最近创建的任务目录
    if [ -z "$model_arg" ] && [ -n "${MODEL_NAME:-}" ]; then
        model_arg="$MODEL_NAME"
    fi

    if [ -n "$model_arg" ]; then
        set_task_paths "${date_arg:-$DEFAULT_DATE_STR}" "$model_arg" "$bench_arg"
    else
        local task_dir
        task_dir=$(latest_task_dir) || { log_err "未找到任务，请先执行 start 或指定 model_name"; exit 1; }
        local parts d m b
        parts=$(parse_task_rel "$task_dir")
        read -r d m b <<< "$parts"
        set_task_paths "$d" "$m" "$b"
        log_info "自动识别任务: ${DATE_STR}/${MODEL_DIR}${BENCH_DIR:+/${BENCH_DIR}}"
    fi

    mkdir -p "$TASK_LOG_DIR"

    # 从该任务自己的 PID 文件解析 vLLM 端口
    local ports
    ports=$(detect_vllm_ports "$PID_FILE")
    if [ -z "$ports" ]; then
        log_err "任务 ${DATE_STR}/${MODEL_DIR} 没有运行中的 vLLM 实例，请先执行 start"
        exit 1
    fi

    local port_count
    port_count=$(echo "$ports" | wc -l)
    echo "任务 ${DATE_STR}/${MODEL_DIR}: 检测到 ${port_count} 个 vLLM 实例端口: $(echo $ports | tr '\n' ' ')"

    # 生成 upstream server 块
    local upstream_servers=""
    for port in $ports; do
        upstream_servers="${upstream_servers}        server 127.0.0.1:${port} max_fails=3 fail_timeout=30s;"$'\n'
    done

    # 模板替换（用 awk 处理多行 upstream 块，避免 sed 多行插入问题）
    sed -e "s|{{NGINX_ERROR_LOG}}|${NGINX_ERROR_LOG}|g" \
        -e "s|{{NGINX_PID}}|${NGINX_PID}|g" \
        -e "s|{{NGINX_ACCESS_LOG}}|${NGINX_ACCESS_LOG}|g" \
        -e "s|{{LISTEN_PORT}}|${listen_port}|g" \
        "$template_file" > "$NGINX_CONF"

    # 临时文件必须和目标同目录：mktemp 默认在 /tmp，跨设备 mv 会报
    # "preserving permissions ... Operation not permitted"，而且不是原子替换。
    local tmp_conf
    tmp_conf=$(mktemp "${NGINX_CONF}.tmp.XXXXXX")
    awk -v servers="$upstream_servers" '
        /\{\{UPSTREAM_SERVERS\}\}/ { printf "%s", servers; next }
        { print }
    ' "$NGINX_CONF" > "$tmp_conf" && mv "$tmp_conf" "$NGINX_CONF"

    echo "已生成 ${NGINX_CONF} (监听端口: ${listen_port}, 后端实例: ${port_count})"
    log_info "Nginx 使用 least_conn 负载均衡策略"

    # 停止旧进程 → 验证 → 启动/重载
    if [ -f "$NGINX_PID" ] && kill -0 "$(cat "$NGINX_PID" 2>/dev/null)" 2>/dev/null; then
        echo "停止旧的 nginx 进程..."
        nginx -s stop -c "${NGINX_CONF}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$(cat "$NGINX_PID" 2>/dev/null)" 2>/dev/null || break
            sleep 1
        done
    fi

    if nginx -t -c "${NGINX_CONF}" 2>&1; then
        if [ -f "$NGINX_PID" ] && kill -0 "$(cat "$NGINX_PID" 2>/dev/null)" 2>/dev/null; then
            nginx -s reload -c "${NGINX_CONF}"
            echo "[OK] Nginx 已重载"
        else
            nginx -c "${NGINX_CONF}"
            echo "[OK] Nginx 已启动"
        fi
        echo "统一访问入口: http://localhost:${listen_port}"

        for _ in $(seq 1 10); do
            if curl -s --noproxy '*' "http://localhost:${listen_port}/health" >/dev/null 2>&1; then
                log_ok "Nginx 健康检查通过"
                return 0
            fi
            sleep 1
        done
        log_warn "Nginx 已启动，但健康检查未通过（后端实例可能尚未就绪）"
    else
        log_err "Nginx 配置验证失败，请检查 ${NGINX_CONF}"
        exit 1
    fi
}

# ========================== status 命令 ==========================

print_task_status() {
    echo "=== 任务 ${DATE_STR}/${MODEL_DIR} ==="
    echo "  目录: ${TASK_LOG_DIR}"
    echo "--- vLLM 进程 ---"
    if [ -f "$PID_FILE" ]; then
        local alive=0 dead=0
        while read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                local port
                port=$(ps -p "$pid" -o args= 2>/dev/null | grep -oP '(?<=--port )\d+' || echo "?")
                echo "  ✅ PID $pid (port $port) running"
                alive=$((alive + 1))
            else
                echo "  ❌ PID $pid dead"
                dead=$((dead + 1))
            fi
        done < "$PID_FILE"
        echo "  存活: $alive | 死亡: $dead"
    else
        echo "  无 PID 文件"
    fi

    echo ""
    echo "--- Nginx ---"
    if [ -f "$NGINX_PID" ]; then
        local npid
        npid=$(cat "$NGINX_PID" 2>/dev/null || true)
        if [ -n "$npid" ] && kill -0 "$npid" 2>/dev/null; then
            echo "  ✅ Nginx running (PID $npid)"
        else
            echo "  ❌ Nginx not running"
        fi
    else
        echo "  无 Nginx PID 文件"
    fi
}

do_status() {
    if [ "$#" -gt 0 ]; then
        local MODEL_NAME="$1"
        local DATE_STR_ARG="${2:-$DEFAULT_DATE_STR}"
        local BENCH_ARG="${3:-${BENCHMARK:-}}"
        set_task_paths "$DATE_STR_ARG" "$MODEL_NAME" "$BENCH_ARG"
        print_task_status
        return 0
    fi

    local found=0
    local dir parts d m b
    # 优先列出带 benchmark 的任务目录（深度 3）
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        parts=$(parse_task_rel "$dir")
        read -r d m b <<< "$parts"
        set_task_paths "$d" "$m" "$b"
        print_task_status
        echo ""
        found=1
    done < <(find "$BASE_LOG_DIR" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | sort)

    # 没有 benchmark 层的老任务（深度 2）
    if [ "$found" -eq 0 ]; then
        while IFS= read -r dir; do
            [ -n "$dir" ] || continue
            parts=$(parse_task_rel "$dir")
            read -r d m b <<< "$parts"
            set_task_paths "$d" "$m" "$b"
            print_task_status
            echo ""
            found=1
        done < <(find "$BASE_LOG_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort)
    fi

    if [ "$found" -eq 0 ]; then
        echo "未找到任何任务目录: ${BASE_LOG_DIR}/<date_str>/<model_name>[/<benchmark>]"
    fi
}

# ========================== stop 命令 ==========================

do_stop() {
    local MODEL_NAME="${1:-}"
    local DATE_STR_ARG="${2:-$DEFAULT_DATE_STR}"
    local BENCH_ARG="${3:-${BENCHMARK:-}}"
    [ -z "$MODEL_NAME" ] && { log_err "缺少 model_name"; exit 1; }

    set_task_paths "$DATE_STR_ARG" "$MODEL_NAME" "$BENCH_ARG"

    if [ ! -d "$TASK_LOG_DIR" ]; then
        log_warn "任务目录不存在: $TASK_LOG_DIR"
        log_warn "该模型下已有的任务目录:"
        find "${BASE_LOG_DIR}/${DATE_STR}/${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sed 's/^/    /' || true
        exit 1
    fi

    echo "任务目录: ${TASK_LOG_DIR}"

    local count=0
    local pids=()

    # 停止 vLLM
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                echo "  正在停止 PID $pid ..."
                kill "$pid" 2>/dev/null || true
                pids+=("$pid")
                count=$((count + 1))
            fi
        done < "$PID_FILE"

        if [ ${#pids[@]} -gt 0 ]; then
            for _ in $(seq 1 30); do
                local alive=0
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        alive=1
                        break
                    fi
                done
                if [ "$alive" -eq 0 ]; then
                    break
                fi
                sleep 1
            done
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo "  PID $pid 未退出，强制结束"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        fi
        rm -f "$PID_FILE"
    fi

    # 停止 Nginx
    if [ -f "$NGINX_PID" ]; then
        local npid
        npid=$(cat "$NGINX_PID" 2>/dev/null || true)
        if [ -n "$npid" ] && kill -0 "$npid" 2>/dev/null; then
            echo "  正在停止 Nginx PID $npid ..."
            kill "$npid" 2>/dev/null || true
            for _ in $(seq 1 10); do
                if ! kill -0 "$npid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$npid" 2>/dev/null; then
                echo "  Nginx PID $npid 未退出，强制结束"
                kill -9 "$npid" 2>/dev/null || true
            fi
            count=$((count + 1))
        fi
        rm -f "$NGINX_PID"
    fi

    log_ok "共停止 $count 个进程"
}

# ========================== 主入口 ==========================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        start)  shift; do_start "$@" ;;
        nginx)  shift; do_nginx "$@" ;;
        status) shift; do_status "$@" ;;
        stop)   shift; do_stop "$@" ;;
        *)
            echo "用法: bash $0 {start|nginx|status|stop}"
            echo ""
            echo "  start  <model_path> <model_name> <tp_size> [date_str] [max_model_len] [benchmark]"
            echo "  nginx  [listen_port] [model_name] [date_str] [benchmark]"
            echo "  status [model_name] [date_str] [benchmark]   # 缺省显示所有任务"
            echo "  stop   <model_name> [date_str] [benchmark]"
            echo ""
            echo "  date_str 默认 ${DEFAULT_DATE_STR}"
            echo "  benchmark 可选；指定后任务目录为 logs/<date_str>/<model_name>/<benchmark>/"
            echo "  TASK_ID 可选；指定后追加 __<TASK_ID>，用于同 benchmark 多任务并发（挂载同一目录时消歧）"
            exit 1
            ;;
    esac
fi
