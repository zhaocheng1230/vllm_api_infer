# vLLM 多实例启动 + Nginx 负载均衡管理脚本

本项目提供一套在 **昇腾 NPU**（Ascend）环境下批量启动多个 vLLM 推理实例，并通过 Nginx 反向代理统一对外提供服务的管理脚本。

## 项目结构

```
.
├── start_work.sh                # 主管理脚本（start / nginx / status / stop）
├── "start_work copy.sh"         # 旧版脚本备份（基于模板文件生成 Nginx 配置）
└── nginx_vllm.conf.template     # Nginx 配置模板（一致性哈希负载均衡版）
```

## 功能特性

- **自动检测 NPU 数量**：通过 `npu-smi info -l` 获取 NPU 总数，按每实例卡数自动计算并启动相应数量的 vLLM 实例
- **NPU 分组隔离**：每个实例通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定独立的 NPU 卡组，端口从 `VLLM_BASE_PORT`（默认 8000）起依次分配
- **模型自适应参数**：根据模型名称动态选择 tool-call parser 与额外参数
  - `Qwen3.5+`：启用 `--language-model-only` + `qwen3_coder` parser
  - `MiniCPM5`：使用 `minicpm5` parser + tool-parser-plugin
  - 其他模型：默认 `hermes` parser
- **健康检查**：启动后轮询各实例 `/health` 接口（超时 300s），汇报就绪情况
- **Nginx 负载均衡**：自动生成 Nginx 配置，`least_conn` 策略 + keepalive 长连接，支持 SSE 流式输出、故障转移（`proxy_next_upstream`）、1800s 长超时
- **统一管理**：一键查看状态、停止全部 vLLM 实例与 Nginx

## 环境要求

- Linux + Bash（脚本使用 `set -euo pipefail`）
- [vLLM](https://docs.vllm.ai/)（昇腾适配版，含 `vllm` CLI）
- Ascend NPU 及 `npu-smi` 工具链
- Nginx ≥ 1.19.8（如需使用模板中的一致性哈希）
- 常用工具：`ss` / `netstat`、`curl`、`ps`

## 快速开始

```bash
# 1. 启动多个 vLLM 实例（自动铺满所有 NPU）
bash start_work.sh start /path/to/model MyModel 2 131072

# 2. 生成并启动 Nginx 反向代理（默认监听 1358 端口）
bash start_work.sh nginx

# 3. 查看运行状态
bash start_work.sh status

# 4. 停止所有服务（vLLM + Nginx）
bash start_work.sh stop
```

## 命令说明

| 命令 | 语法 | 说明 |
| --- | --- | --- |
| `start` | `bash start_work.sh start <model_path> <model_name> <tp_size> [max_model_len]` | 按 NPU 总数 / 每模型卡数计算实例数量，逐个后台启动 vLLM，并进行健康检查。`max_model_len` 默认 `131072` |
| `nginx` | `bash start_work.sh nginx [listen_port]` | 从 PID 文件收集运行中的 vLLM 端口，生成 `/etc/nginx/nginx.conf`，校验后启动或重载 Nginx。`listen_port` 默认 `1358` |
| `status` | `bash start_work.sh status` | 显示各 vLLM 进程（PID / 端口）存活情况及 Nginx 运行状态 |
| `stop` | `bash start_work.sh stop` | 停止所有 vLLM 实例与 Nginx，并清理 PID 文件 |

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLLM_BASE_PORT` | `8000` | vLLM 实例起始端口，后续实例依次 +1 |
| `MINICPM5_PLUGIN` | `/path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py` | MiniCPM5 模型的 tool-parser-plugin 路径 |

## vLLM 关键启动参数

脚本为每个实例使用以下核心参数（可按需修改 `cmd_start`）：

```bash
vllm serve <model_path> \
    --host 0.0.0.0 --port <port> \
    --served-model-name <model_name> \
    --tensor-parallel-size <tp_size> \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --max-model-len <max_model_len> \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser <parser>
```

## Nginx 配置说明

### 运行时生成版（`start_work.sh nginx`）

- 负载均衡：`least_conn`，upstream 自动填入检测到的所有 vLLM 端口
- 长连接：`keepalive 32`
- 流式支持：`proxy_buffering off`、`proxy_http_version 1.1`、清空 `Connection` 头、`Cache-Control: no-cache`
- 容错：`proxy_next_upstream error timeout invalid_header http_504 http_429`，重试次数等于实例数
- 超时：读/写 1800s，连接 300s；`client_max_body_size 100m`
- 日志：`start_log/nginx/access.log`、`error.log`（自定义 `vllm_format` 记录 upstream 与耗时）

### 模板版（`nginx_vllm.conf.template`）

占位符：`{{NGINX_ERROR_LOG}}`、`{{NGINX_PID}}`、`{{UPSTREAM_SERVERS}}`、`{{LISTEN_PORT}}`、`{{NGINX_ACCESS_LOG}}`。

特点：采用 `hash $request_id consistent` **一致性哈希**解决 keepalive 场景下的流量倾斜问题（需 Nginx ≥ 1.19.8），`keepalive 128` / `keepalive_requests 10000`，并开启 `reuseport`。可通过 `envsubst` 等工具渲染后使用，例如：

```bash
sed -e "s|{{LISTEN_PORT}}|1358|g" \
    -e "s|{{NGINX_ACCESS_LOG}}|/var/log/nginx/vllm_access.log|g" \
    -e "s|{{NGINX_ERROR_LOG}}|/var/log/nginx/vllm_error.log|g" \
    -e "s|{{NGINX_PID}}|/tmp/nginx_vllm.pid|g" \
    -e "s|{{UPSTREAM_SERVERS}}|server 127.0.0.1:8000;\n server 127.0.0.1:8001;|g" \
    nginx_vllm.conf.template > /tmp/nginx_vllm.conf
nginx -c /tmp/nginx_vllm.conf
```

## 产物与日志

| 路径 | 说明 |
| --- | --- |
| `.vllm_pids` | 所有 vLLM 实例的 PID 列表（与脚本同目录） |
| `start_log/vllm_<时间戳>_<端口>.log` | 各 vLLM 实例的 stdout/stderr 日志 |
| `start_log/nginx/access.log` / `error.log` | Nginx 访问与错误日志 |
| `start_log/nginx/nginx.pid` | Nginx 主进程 PID |
| `/etc/nginx/nginx.conf` | `nginx` 命令生成的配置文件 |

## 验证服务

```bash
# 单实例健康检查
curl --noproxy '*' http://localhost:8000/health

# 通过 Nginx 统一入口调用 OpenAI 兼容接口
curl http://localhost:1358/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MyModel", "messages": [{"role": "user", "content": "你好"}]}'
```

## 注意事项

- 端口被占用时该实例会被跳过并输出 WARN，不会中断整体启动流程
- `nginx` 命令会直接覆盖 `/etc/nginx/nginx.conf`，如机器上已有其他 Nginx 业务请谨慎使用
- 部分实例健康检查超时只会给出 WARN 提示，请查看对应日志定位原因（如显存不足、模型加载失败）
- `start_work copy.sh` 为旧版本备份：其 `do_nginx` 基于 `nginx_vllm.conf.template` 渲染配置到 `/tmp`，并用 `ss` 实时探测 vLLM 端口，可作为参考
