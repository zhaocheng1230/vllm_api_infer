#!/bin/bash
# start_work.sh - vLLM 多实例启动 + Nginx 反向代理管理
# 用法:
#   bash start_work.sh start <model_path> <model_name> <tp_size> [max_model_len]
#   bash start_work.sh nginx <listen_port>
#   bash start_work.sh status
#   bash start_work.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.vllm_pids"
TASK_LOG_DIR="${SCRIPT_DIR}/start_log"

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

# ========================== start 命令 ==========================

cmd_start() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local NPU_PER_MODEL="${3:-2}"
    local MAX_MODEL_LEN="${4:-131072}"
    local BASE_PORT="${VLLM_BASE_PORT:-8000}"

    [ -z "$MODEL_PATH" ] && { log_err "缺少 model_path"; exit 1; }
    [ -z "$MODEL_NAME" ] && { log_err "缺少 model_name"; exit 1; }

    # 检测 NPU 总数
    local NPU_TOTAL
    NPU_TOTAL=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo 8)
    local LOOP_COUNT=$((NPU_TOTAL / NPU_PER_MODEL))

    [ "$LOOP_COUNT" -le 0 ] && { log_err "NPU 不足: total=$NPU_TOTAL, per_model=$NPU_PER_MODEL"; exit 1; }

    mkdir -p "$TASK_LOG_DIR"

    # --- 根据模型名称动态添加启动参数 ---
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
    echo "  模型路径:   $MODEL_PATH"
    echo "  模型名称:   $MODEL_NAME"
    echo "  NPU总数:    $NPU_TOTAL"
    echo "  每模型卡数: $NPU_PER_MODEL"
    echo "  实例数量:   $LOOP_COUNT"
    echo "  端口范围:   $BASE_PORT ~ $((BASE_PORT + LOOP_COUNT - 1))"
    echo "========================================="

    for i in $(seq 0 $((LOOP_COUNT - 1))); do
        local CURRENT_PORT=$((BASE_PORT + i))
        local NPU_START=$((i * NPU_PER_MODEL))
        local CURRENT_NPUS
        CURRENT_NPUS=$(seq -s, "$NPU_START" $((NPU_START + NPU_PER_MODEL - 1)))

        if ! check_port_available "$CURRENT_PORT"; then
            log_warn "端口 $CURRENT_PORT 已被占用，跳过该实例"
            continue
        fi

        local LOG_FILE="${TASK_LOG_DIR}/vllm_${TIMESTAMP}_${CURRENT_PORT}.log"

        echo "正在启动任务 $i: NPU [${CURRENT_NPUS}], 端口 [${CURRENT_PORT}], 日志 [${LOG_FILE}]..."

        VLLM_NO_USAGE_STATS=1 \
        OTEL_SDK_DISABLED=true \
        ASCEND_RT_VISIBLE_DEVICES="${CURRENT_NPUS}" \
        vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${CURRENT_PORT}" \
            --served-model-name "${MODEL_NAME}" \
            --tensor-parallel-size "${NPU_PER_MODEL}" \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --max-model-len "${MAX_MODEL_LEN}" \
            --trust-remote-code \
            --enable-prefix-caching \
            --enable-auto-tool-choice \
            --tool-call-parser "${TOOL_CALL_PARSER}" \
            ${EXTRA_VLLM_ARGS} \
            > "${LOG_FILE}" 2>&1 &

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

# ========================== nginx 命令 ==========================

cmd_nginx() {
    local listen_port="${1:-1358}"
    local nginx_conf="/etc/nginx/nginx.conf"
    local nginx_error_log="${SCRIPT_DIR}/start_log/nginx/error.log"
    local nginx_access_log="${SCRIPT_DIR}/start_log/nginx/access.log"
    local nginx_pid="${SCRIPT_DIR}/start_log/nginx/nginx.pid"

    mkdir -p "$(dirname "$nginx_error_log")"

    # 从 PID 文件或运行中的进程获取 vLLM 端口
    local ports=()
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            local p
            p=$(ps -p "$pid" -o args= 2>/dev/null | grep -oP '(?<=--port )\d+' || true)
            [ -n "$p" ] && ports+=("$p")
        done < "$PID_FILE"
    fi

    # 去重并排序
    IFS=$'\n' ports=($(printf '%s\n' "${ports[@]}" | sort -un)); unset IFS

    [ ${#ports[@]} -eq 0 ] && { log_err "未找到运行中的 vLLM 实例"; exit 1; }

    local port_count=${#ports[@]}
    local upstream_servers=""
    for p in "${ports[@]}"; do
        upstream_servers+="        server 127.0.0.1:${p};\n"
    done

    cat > "$nginx_conf" << NGINX_EOF
worker_processes  auto;
error_log  ${nginx_error_log} warn;
pid        ${nginx_pid};

events {
    worker_connections  4096;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    log_format vllm_format '\$remote_addr - [\$time_local] "\$request" '
                           'status:\$status upstream:\$upstream_addr '
                           'rt:\$request_time urt:\$upstream_response_time';

    client_max_body_size 100m;

    upstream vllm_backend {
        least_conn;

$(echo -e "$upstream_servers")
        keepalive 32;
    }

    server {
        listen ${listen_port};
        server_name localhost;
        access_log ${nginx_access_log} vllm_format;

        location / {
            proxy_pass http://vllm_backend;
            proxy_next_upstream error timeout invalid_header http_504 http_429;
            proxy_next_upstream_tries ${port_count};
            proxy_next_upstream_timeout 120s;

            # 流式输出支持
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_request_buffering off;

            # 超时控制
            proxy_read_timeout 1800s;
            proxy_send_timeout 1800s;
            proxy_connect_timeout 300s;

            # SSE/流式响应必需
            proxy_set_header Cache-Control "no-cache";
            chunked_transfer_encoding on;

            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
    }
}
NGINX_EOF

    nginx -t -c "$nginx_conf" 2>&1 || { log_err "Nginx 配置校验失败"; exit 1; }

    # 停止旧 nginx（如果有）
    if [ -f "$nginx_pid" ]; then
        local old_pid
        old_pid=$(cat "$nginx_pid" 2>/dev/null || true)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            nginx -s reload -c "$nginx_conf" 2>/dev/null && { log_ok "Nginx 已重载"; return 0; }
            kill "$old_pid" 2>/dev/null || true
            sleep 1
        fi
    fi

    nginx -c "$nginx_conf"
    log_ok "Nginx 已启动"
    echo "统一访问入口: http://localhost:${listen_port}"
}

# ========================== status 命令 ==========================

cmd_status() {
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
    local nginx_pid_file="${SCRIPT_DIR}/start_log/nginx/nginx.pid"
    if [ -f "$nginx_pid_file" ]; then
        local npid
        npid=$(cat "$nginx_pid_file" 2>/dev/null || true)
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

cmd_stop() {
    local count=0

    # 停止 vLLM
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  正在停止 PID $pid ..."
                kill "$pid" 2>/dev/null || true
                count=$((count + 1))
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    # 停止 Nginx
    local nginx_pid_file="${SCRIPT_DIR}/start_log/nginx/nginx.pid"
    if [ -f "$nginx_pid_file" ]; then
        local npid
        npid=$(cat "$nginx_pid_file" 2>/dev/null || true)
        if [ -n "$npid" ] && kill -0 "$npid" 2>/dev/null; then
            echo "  正在停止 Nginx PID $npid ..."
            kill "$npid" 2>/dev/null || true
            count=$((count + 1))
        fi
        rm -f "$nginx_pid_file"
    fi

    log_ok "共停止 $count 个进程"
}

# ========================== 主入口 ==========================

case "${1:-}" in
    start)  shift; cmd_start "$@" ;;
    nginx)  shift; cmd_nginx "$@" ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
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
