#!/bin/bash
# start_server.sh - vLLM 多实例启动 + Nginx 反向代理管理（Ascend NPU / NVIDIA GPU 通用）
# 用法:
#   bash start_server.sh start <model_path> <model_name> <tp_size> [max_model_len]
#   bash start_server.sh nginx <listen_port>
#   bash start_server.sh status
#   bash start_server.sh stop
#
# 可选环境变量:
#   DEVICE_BACKEND=auto|npu|gpu   # 默认 auto，按 npu-smi / nvidia-smi 自动识别
#   DEVICE_COUNT=N                # 覆盖自动检测到的设备数量
#   VLLM_GPU_MEM_UTIL=0.90        # GPU 显存利用率（NPU 使用 TP 档位内置值）
#   VLLM_DTYPE=auto|bfloat16      # 覆盖 dtype（GPU 默认 auto，NPU 默认 bfloat16）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.vllm_pids"
TASK_LOG_DIR="${SCRIPT_DIR}/logs"
NGINX_PID="/tmp/nginx_vllm.pid"
NGINX_CONF="/tmp/nginx_vllm.conf"
NGINX_ERROR_LOG="/tmp/nginx_vllm_error.log"
NGINX_ACCESS_LOG="/tmp/nginx_vllm_access.log"

# 默认后端：优先读取 DEVICE_BACKEND，未设置则自动检测，默认值为 auto，可选项npu|gpu
DEFAULT_BACKEND="${DEVICE_BACKEND:-auto}"

# ========================== 工具函数 ==========================

log_info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
log_ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
log_err()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

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
    local ports=""
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            [ -z "$pid" ] && continue
            if kill -0 "$pid" 2>/dev/null; then
                local p
                p=$(ps -p "$pid" -o args= 2>/dev/null | grep -oP '(?<=--port )\d+' || true)
                [ -n "$p" ] && ports="${ports}${p}"$'\n'
            fi
        done < "$PID_FILE"
    fi
    if [ -z "$ports" ]; then
        ports=$(ss -tlnp 2>/dev/null | grep '"vllm"' | grep -oP '(?<=0\.0\.0\.0:)\d+' || true)
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
            export TASK_QUEUE_ENABLE=1
            export COMBINED_ENABLE=1
            export ASCEND_SLOG_PRINT_TO_STDOUT=0
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

    case "$tp" in
        1) echo 0.92 ;;
        2) echo 0.95 ;;
        4) echo 0.93 ;;
        8) echo 0.90 ;;
        *) echo 0.95 ;;
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

# --- 基线参数查表（按 TP 档位） ---
tp_lookup() {
    local tp=$1
    case $tp in
        1) MAX_NUM_SEQS=128; MAX_BATCHED_TOKENS=32768; EXTRA_CHUNKED_ARGS="" ;;
        2) MAX_NUM_SEQS=128; MAX_BATCHED_TOKENS=16384; EXTRA_CHUNKED_ARGS="--max-num-partial-prefills 1" ;;
        4) MAX_NUM_SEQS=96;  MAX_BATCHED_TOKENS=32768; EXTRA_CHUNKED_ARGS="--max-num-partial-prefills 1" ;;
        8) MAX_NUM_SEQS=64;  MAX_BATCHED_TOKENS=65536; EXTRA_CHUNKED_ARGS="--max-num-partial-prefills 1" ;;
        *) MAX_NUM_SEQS=128; MAX_BATCHED_TOKENS=16384; EXTRA_CHUNKED_ARGS="" ;;
    esac
}

# ========================== start 命令 ==========================

do_start() {
    local MODEL_PATH="${1:-}"
    local MODEL_NAME="${2:-}"
    local TP_SIZE="${3:-2}"
    local MAX_MODEL_LEN="${4:-131072}"
    local BASE_PORT="${VLLM_BASE_PORT:-8000}"

    [ -z "$MODEL_PATH" ] && { log_err "缺少 model_path"; exit 1; }
    [ -z "$MODEL_NAME" ] && { log_err "缺少 model_name"; exit 1; }
    if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        log_err "tp_size 必须是正整数: $TP_SIZE"
        exit 1
    fi

    # 1) 后端识别
    local BACKEND
    BACKEND=$(resolve_backend)

    # 2) 设备列表解析
    local DEVICE_TOTAL
    local -a DEVICE_IDS=()
    device_ids_for_backend "$BACKEND"

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
            log_err "已有 vLLM 实例在运行 (PID: ${alive_pids[*]})，请先执行 stop"
            exit 1
        fi
    fi
    > "$PID_FILE"

    # 3) 基线参数
    local GPU_MEM_UTIL MAX_NUM_SEQS MAX_BATCHED_TOKENS EXTRA_CHUNKED_ARGS
    tp_lookup "$TP_SIZE"
    GPU_MEM_UTIL=$(backend_memory_util "$BACKEND" "$TP_SIZE")

    local DTYPE
    DTYPE=$(backend_dtype "$BACKEND")

    # 4) 模型特殊参数
    local EXTRA_VLLM_ARGS=""
    local TOOL_CALL_PARSER="hermes"

    if echo "$MODEL_NAME" | grep -iqE "Qwen3[._][5-9]"; then
        log_info "检测到 Qwen3.5+ 模型，启用 language_model_only + qwen3_coder parser"
        TOOL_CALL_PARSER="qwen3_coder"
        EXTRA_VLLM_ARGS="--language-model-only"
    elif echo "$MODEL_NAME" | grep -iqE "MiniCPM5|minicpm5"; then
        log_info "检测到 MiniCPM5 模型，使用 minicpm5 tool parser"
        TOOL_CALL_PARSER="minicpm5"
        EXTRA_VLLM_ARGS="--tool-parser-plugin ${MINICPM5_PLUGIN:-/path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py}"
    fi

    local TIMESTAMP
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local started_ports=()

    echo "========================================="
    echo "  模型路径:       $MODEL_PATH"
    echo "  模型名称:       $MODEL_NAME"
    echo "  设备后端:       $BACKEND"
    echo "  设备总数:       $DEVICE_TOTAL"
    echo "  设备列表:       $(IFS=,; echo "${DEVICE_IDS[*]:0:$DEVICE_TOTAL}")"
    echo "  每模型卡数:     $TP_SIZE"
    echo "  实例数量:       $LOOP_COUNT"
    echo "  端口范围:       $BASE_PORT ~ $((BASE_PORT + LOOP_COUNT - 1))"
    echo "  显存利用率:     $GPU_MEM_UTIL"
    echo "  dtype:          $DTYPE"
    echo "  max-num-seqs:   $MAX_NUM_SEQS"
    echo "  batched-tokens: $MAX_BATCHED_TOKENS"
    echo "  extra-chunked:  ${EXTRA_CHUNKED_ARGS:-(none)}"
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

        # 在子 shell 中设置设备相关环境变量，exec 后 PID 即为 vllm 进程
        (
            export_backend_env "$BACKEND" "$CURRENT_DEVICES"

            exec vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port "${CURRENT_PORT}" \
                --served-model-name "${MODEL_NAME}" \
                --tensor-parallel-size "${TP_SIZE}" \
                --dtype "${DTYPE}" \
                --gpu-memory-utilization "${GPU_MEM_UTIL}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --max-num-seqs "${MAX_NUM_SEQS}" \
                --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
                ${EXTRA_CHUNKED_ARGS} \
                --trust-remote-code \
                --enable-prefix-caching \
                --enable-chunked-prefill \
                --enable-auto-tool-choice \
                --tool-call-parser "${TOOL_CALL_PARSER}" \
                ${EXTRA_VLLM_ARGS} \
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
        if wait_for_port "$port" 300; then
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
        echo "  查看日志: tail -f ${TASK_LOG_DIR}/vllm_*.log"
        echo "  查看状态: bash $0 status"
        echo "  停止服务: bash $0 stop"
        echo "  健康检查: curl --noproxy '*' http://localhost:${started_ports[0]}/health"
        echo "========================================="
    else
        log_warn "部分实例未就绪: ${ready_count}/${#started_ports[@]}"
    fi

    # 输出端口列表供外部脚本使用
    echo "检测到 ${#started_ports[@]} 个 vLLM 实例端口: ${started_ports[*]}"
}

# ========================== NGINX 配置生成 ==========================
do_nginx() {
    local listen_port="${1:-1358}"
    local template_file="$(dirname "$0")/nginx_vllm.conf.template"

    if ! command -v nginx >/dev/null 2>&1; then
        log_err "未找到 nginx 命令，请先安装 Nginx 或将其加入 PATH"
        log_err "  Debian/Ubuntu: sudo apt-get install -y nginx"
        log_err "  RHEL/CentOS:   sudo yum install -y nginx"
        exit 1
    fi

    [ -f "$template_file" ] || { log_err "模板文件不存在: ${template_file}"; exit 1; }

    # 动态检测 vLLM 端口（优先从 PID 文件解析，ss 作为兜底）
    local ports
    ports=$(detect_vllm_ports)
    [ -n "$ports" ] || { log_err "未检测到任何 vllm 监听端口，请先执行 start"; exit 1; }

    local port_count=$(echo "$ports" | wc -l)
    echo "检测到 ${port_count} 个 vLLM 实例端口: $(echo $ports | tr '\n' ' ')"

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

    local tmp_conf=$(mktemp)
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

do_status() {
    echo "=== vLLM 进程 ==="
    if [ -f "$PID_FILE" ]; then
        local alive=0 dead=0
        while read -r pid; do
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
    echo "=== Nginx ==="
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

# ========================== stop 命令 ==========================

do_stop() {
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
        status) do_status ;;
        stop)   do_stop ;;
        *)
            echo "用法: bash $0 {start|nginx|status|stop}"
            echo ""
            echo "  start  <model_path> <model_name> <tp_size> [max_model_len]"
            echo "  nginx  [listen_port]          # 默认 1358"
            echo "  status                        # 查看运行状态"
            echo "  stop                          # 停止所有服务"
            exit 1
            ;;
    esac
fi
