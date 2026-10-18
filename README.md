# vLLM 多实例推理框架（Ascend NPU / NVIDIA GPU + Nginx）

本项目面向 **昇腾 NPU 或 NVIDIA GPU** 环境，启动时会自动识别设备后端，并提供一套完整的批量推理流水线：

- 自动启动多个 vLLM 实例，并通过 Nginx 统一对外提供 OpenAI 兼容接口
- 异步流式调用接口，高并发处理 `datasets/` 目录下的所有 `.jsonl` 文件
- 为每条数据的 `prompt` 生成 `output`，结果写入 `output/` 目录

默认输入目录为 `datasets/`，默认输出目录为 `output/`。

## 项目结构

```
.
├── start_server.sh              # 服务端管理：启动 vLLM 多实例 + Nginx
├── vllm_infer.py                # 客户端：异步流式批量推理
├── run_pipeline.sh              # 任务入口：启动服务 + 执行推理
├── nginx_vllm.conf.template     # Nginx 配置模板
├── .gitignore                   # 忽略运行时产物
├── datasets/                    # 输入目录（*.jsonl）
├── output/                      # 输出目录
└── logs/                        # vLLM 实例日志（运行后生成）
```

## 环境要求

- Linux + Bash（脚本使用 `set -euo pipefail`）
- vLLM（NPU 使用昇腾适配版；GPU 使用官方 GPU 版）
- NPU：Ascend NPU + `npu-smi`
- GPU：NVIDIA GPU + `nvidia-smi`
- Nginx ≥ 1.19.8（一致哈希需要）
- Python 依赖：`openai`、`httpx`、`tqdm`（可选）
- 常用工具：`ss` / `netstat`、`curl`、`ps`

## 快速开始

### 1. 准备输入数据

把需要推理的 `.jsonl` 文件放入 `datasets/`。每行一个 JSON，至少包含 `prompt` 字段：

```json
{"prompt": "请解释什么是张量并行"}
{"prompt": "写一个快速排序的 Python 实现"}
```

也支持直接传入 messages 列表：

```json
{"prompt": [{"role": "user", "content": "你好"}]}
```

客户端依次识别以下字段作为 prompt：`prompt`、`input`、`question`、`query`、`content`、`messages`。

### 2. 配置模型并启动任务

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=MyModel
export TP_SIZE=8

bash run_pipeline.sh
```

`run_pipeline.sh` 会自动完成：

1. 检查 Nginx 端口上的服务是否已运行
2. 若未运行，调用 `start_server.sh start` 启动 vLLM 多实例
3. 调用 `start_server.sh nginx` 启动 Nginx 反向代理
4. 等待 `/v1/models` 就绪
5. 调用 `vllm_infer.py` 处理 `datasets/` 下所有 jsonl
6. 结果写入 `output/`

### 3. 指定目录和端口

```bash
bash run_pipeline.sh ./datasets ./output 1358
```

## 设备后端自动识别

`start_server.sh` 同时支持 Ascend NPU 与 NVIDIA GPU：

| 后端 | 检测方式 | 可见设备变量 |
| --- | --- | --- |
| NPU | `npu-smi info -l` 中出现 `NPU ID` | `ASCEND_RT_VISIBLE_DEVICES` |
| GPU | `nvidia-smi -L` 中出现 `GPU N` | `CUDA_VISIBLE_DEVICES` |

默认 `DEVICE_BACKEND=auto`。如果机器上同时存在 NPU 和 GPU，或自动识别失败，可手动指定：

```bash
export DEVICE_BACKEND=npu   # 或 gpu
```

脚本会优先复用已有的 `CUDA_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES`。也可以进一步限制设备数量（例如设备工具不可用，或只想使用部分卡）：

```bash
export DEVICE_COUNT=8
```

GPU 默认参数：

- 显存利用率：`0.90`，可通过 `VLLM_GPU_MEM_UTIL` 覆盖
- dtype：`auto`，可通过 `VLLM_DTYPE` 覆盖
- 其余 TP 档位参数与 NPU 相同

NPU 默认参数：

- 显存利用率按 TP 档位取值（见下表）
- dtype：`bfloat16`

## 常用命令

### 服务端管理

```bash
# 启动 vLLM 实例（按设备总数 / tp_size 自动计算实例数）
bash start_server.sh start <model_path> <model_name> <tp_size> [max_model_len]

# 启动/重载 Nginx（默认监听 1358）
bash start_server.sh nginx [listen_port]

# 查看 vLLM 与 Nginx 状态
bash start_server.sh status

# 停止所有 vLLM 实例与 Nginx
bash start_server.sh stop
```

### 客户端推理

```bash
# 使用默认 datasets/ 和 output/
python vllm_infer.py --model MyModel

# 完整参数示例
python vllm_infer.py \
    --input_dir ./datasets \
    --output_dir ./output \
    --host http://localhost:1358/v1 \
    --model MyModel \
    --concurrency 128 \
    --max_tokens 4096
```

## `run_pipeline.sh` 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | 必填 | 模型路径 |
| `MODEL_NAME` | 必填 | 模型名，需与 `--served-model-name` 一致 |
| `TP_SIZE` | `2` | 每个实例使用的 NPU 卡数 |
| `MAX_MODEL_LEN` | `131072` | 最大模型长度 |
| `VLLM_BASE_PORT` | `8000` | vLLM 实例起始端口 |
| `NGINX_PORT` | `1358` | Nginx 监听端口 |
| `CONCURRENCY` | `64` | 推理并发数 |
| `STOP_AFTER` | `0` | 推理结束后是否停止服务（`1` 为停止） |
| `PYTHON` | `python` | Python 解释器命令 |
| `DEVICE_BACKEND` | `auto` | 设备后端：`auto` / `npu` / `gpu` |
| `DEVICE_COUNT` | 自动检测 | 覆盖设备数量 |
| `VLLM_GPU_MEM_UTIL` | `0.90` | GPU 显存利用率 |
| `VLLM_DTYPE` | GPU 为 `auto`，NPU 为 `bfloat16` | 覆盖 dtype |

示例：

```bash
export MODEL_PATH=/data/models/Qwen3-235B-A22B
export MODEL_NAME=Qwen3-235B-A22B
export TP_SIZE=8
export CONCURRENCY=128
export STOP_AFTER=1

bash run_pipeline.sh
```

## `vllm_infer.py` 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input_dir` | `<项目>/datasets` | 输入 jsonl 目录 |
| `--output_dir` | `<项目>/output` | 输出目录 |
| `--host` | `http://localhost:1358/v1` | OpenAI 兼容接口 base_url |
| `--model` | 必填 | 模型名 |
| `--key` | `EMPTY` | API Key |
| `--concurrency` | `64` | 并发请求数 |
| `--max_tokens` | `4096` | 最大生成 token 数 |
| `--temperature` | `0.0` | 采样温度 |
| `--top_p` | `1.0` | top-p |
| `--top_k` | `-1` | top-k |
| `--presence_penalty` | `0.0` | 重复惩罚 |
| `--system_prompt` | 无 | 统一系统提示词 |
| `--recursive` | 关闭 | 递归扫描子目录 |
| `--resume` | 关闭 | 已完整处理的文件跳过 |
| `--save_jsonl` / `--no-save_jsonl` | 开启 | 是否保存完整字段的 jsonl 结果 |

## 输入与输出

### 输入

```text
datasets/
├── group_a.jsonl
└── subdir/
    └── group_b.jsonl
```

### 输出

```text
output/
├── group_a_output.txt
├── group_a_output.jsonl
└── subdir/
    └── group_b_output.txt
    └── group_b_output.jsonl
```

- `*_output.txt`：每行对应输入中的一条数据，内容为生成的 `output`。输出内部的换行会被替换为空格，保证行对齐。
- `*_output.jsonl`：保留原始字段，并追加：
  - `output`：生成内容
  - `prompt_token_len` / `input_token_len` / `output_token_len`：token 数
  - `output_time`：总耗时
  - `ttft`：首 token 时间
  - `tps`：输出 token/秒
  - `success`：是否成功
  - `error`：失败原因（成功时为空）
- 使用 `--no-save_jsonl` 时只生成 `.txt`。
- 使用 `--recursive` 时会保留输入的子目录结构。

## 服务端参数说明

`start_server.sh` 会根据设备总数和 `tp_size` 自动计算实例数量，并按 TP 档位选择性能参数。下表为 NPU 档位；GPU 的显存利用率统一默认为 `0.90`（可通过 `VLLM_GPU_MEM_UTIL` 覆盖），其余参数相同：

| TP | 显存利用率（NPU） | max-num-seqs | max-num-batched-tokens | partial-prefills |
| --- | --- | --- | --- | --- |
| 1 | 0.92 | 128 | 32768 | 不启用 |
| 2 | 0.95 | 128 | 16384 | 1 |
| 4 | 0.93 | 96 | 32768 | 1 |
| 8 | 0.90 | 64 | 65536 | 1 |
| 其他 | 0.95 | 128 | 16384 | 不启用 |

模型自适应参数：

- **Qwen3.5+**：启用 `--language-model-only` + `qwen3_coder` tool parser
- **MiniCPM5**：启用 `minicpm5` tool parser，可通过 `MINICPM5_PLUGIN` 指定插件路径
- **其他模型**：默认 `hermes` tool parser

## Nginx 说明

`start_server.sh nginx` 会读取 `nginx_vllm.conf.template`，动态填入当前运行的 vLLM 端口并生成：

| 路径 | 说明 |
| --- | --- |
| `/tmp/nginx_vllm.conf` | 生成的 Nginx 配置 |
| `/tmp/nginx_vllm.pid` | Nginx 主进程 PID |
| `/tmp/nginx_vllm_access.log` | Nginx 访问日志 |
| `/tmp/nginx_vllm_error.log` | Nginx 错误日志 |

配置特点：

- `location /` 统一反向代理到所有 vLLM 实例
- `least_conn` 负载均衡：把请求发到当前活跃连接最少的实例，避免单机批量推理时流量集中到某一个端口
- `proxy_buffering off`，支持 SSE 流式输出
- 读写超时 1800s，连接超时 30s
- 后端故障自动切换

## 运行产物

| 路径 | 说明 |
| --- | --- |
| `.vllm_pids` | 所有 vLLM 实例 PID 列表 |
| `logs/vllm_<时间戳>_<端口>.log` | 各 vLLM 实例运行日志 |
| `/tmp/nginx_vllm_*.log` | Nginx 访问/错误日志 |
| `output/*_output.txt` | 每条输入对应的生成结果 |
| `output/*_output.jsonl` | 完整结果与性能指标 |

## 验证服务

```bash
# 单实例健康检查
curl --noproxy '*' http://localhost:8000/health

# 通过 Nginx 查看模型列表
curl --noproxy '*' http://localhost:1358/v1/models

# 调用 OpenAI 兼容接口
curl --noproxy '*' http://localhost:1358/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MyModel","messages":[{"role":"user","content":"你好"}]}'
```

## 注意事项

- 运行前安装客户端依赖：`pip install openai httpx tqdm`
- NPU 环境需要 `npu-smi`，GPU 环境需要 `nvidia-smi`；自动识别失败时请设置 `DEVICE_BACKEND=npu|gpu`。
- GPU 默认 `--dtype auto`，NPU 默认 `--dtype bfloat16`；如需强制指定，设置 `VLLM_DTYPE`。
- 如果 `run_pipeline.sh` 检测到服务已运行，会跳过启动；请确保 `MODEL_NAME` 与 `/v1/models` 中的模型名一致，否则会给出 WARN。
- `start_server.sh start` 检测到已存在存活的 vLLM 实例时会拒绝启动，需先执行 `stop`，避免旧进程失去 PID 跟踪。
- `start_server.sh stop` 会先发送 SIGTERM，等待最多 30s；仍未退出的 vLLM 进程会被强制结束，Nginx 最多等待 10s。
- 端口被占用时，对应 vLLM 实例会被跳过并输出 WARN，不影响其他实例。
- Nginx 使用 `least_conn` 负载均衡，适合单机异步批量推理；如果仍出现某个实例明显空闲，可检查客户端是否只建立了单一长连接，或尝试降低 `keepalive` 数量。
- 部分实例健康检查超时只会输出 WARN，请查看 `logs/` 下的日志定位原因。
- `STOP_AFTER=1` 时，无论推理成功还是失败，脚本退出前都会停止 vLLM 与 Nginx。
- `--resume` 依据 `.txt` 行数与输入行数是否一致判断，若中途失败较多，建议先删除对应输出后重跑。
- 默认 `--concurrency 64`；如果 Nginx 或 vLLM 出现超时/拒绝连接，可适当降低并发，或根据 NPU 数量提高。
- 输入字段不限于 `prompt`：例如 `datasets/gpqa.jsonl` 使用 `question` 字段，客户端会自动识别。
