#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/conf2sh.py - 读取 benchmark 配置文档，输出 shell 变量 / 调度计划

用法:
  conf2sh.py <conf.json> <bench>        输出 shell 变量（可直接 eval）
  conf2sh.py <conf.json> --list          列出所有 benchmark 名
  conf2sh.py <conf.json> --plan a,b,c    输出调度计划，每行 "<bench> <需重启服务:0|1>"

配置文档结构（configs/benchmarks.json）:
{
  "<benchmark名>": {
    "dataset": { "input_file": "./datasets/xxx.jsonl", "output_dir": "./output/xxx" },
    "server":  { "max_model_len": 8192, "max_num_seqs": 64,
                 "max_num_batched_tokens": 32768 },
    "client":  { "max_tokens": 4096, "temperature": 0.0, "top_p": 1.0,
                 "system_prompt_file": null }
  }
}

- dataset.input_file : 该 benchmark 唯一的数据集文件路径（直接读，不扫描目录）
- server 段的键 -> start_server.sh 的启动参数
- client 段的键 -> vllm_infer.py 的命令行参数名（去掉 --），直接透传
- client 段的 "system_prompt_file" 特殊：读出文件内容，调用方转成 --system_prompt
"""
from __future__ import annotations

import json
import shlex
import sys
from typing import Any, Dict, List

# client 段里需要特殊处理、不直接作为命令行参数透传的键
SPECIAL_KEYS = {"system_prompt_file"}


def load_conf(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def server_fp(cfg: Dict[str, Any]) -> str:
    """服务端参数指纹：指纹相同 = 可以复用同一个服务实例。"""
    server = cfg.get("server", {}) or {}
    return json.dumps(server, sort_keys=True, ensure_ascii=False)


def shell_out(name: str, bench: str, cfg: Dict[str, Any]) -> str:
    dataset = cfg.get("dataset", {}) or {}
    server = cfg.get("server", {}) or {}
    client = cfg.get("client", {}) or {}

    lines: List[str] = [f"BENCH_NAME={shlex.quote(bench)}"]
    lines.append(f"INPUT_FILE={shlex.quote(str(dataset.get('input_file', '')))}")
    lines.append(f"OUTPUT_DIR={shlex.quote(str(dataset.get('output_dir', '')))}")

    # 服务端参数
    lines.append(f"MAX_MODEL_LEN={shlex.quote(str(server.get('max_model_len', '')))}")
    lines.append(f"MAX_NUM_SEQS={shlex.quote(str(server.get('max_num_seqs', '')))}")
    lines.append(f"MAX_NUM_BATCHED_TOKENS={shlex.quote(str(server.get('max_num_batched_tokens', '')))}")
    lines.append(f"SERVER_FP={shlex.quote(server_fp(cfg))}")

    # 客户端参数 -> 数组
    args: List[str] = []
    for k, v in client.items():
        if k in SPECIAL_KEYS:
            continue
        if v is None:
            continue
        args.append(f"--{k}")
        args.append(shlex.quote(str(v)))
    lines.append("CLIENT_ARGS=(" + " ".join(args) + ")")

    # 特殊键
    sp = client.get("system_prompt_file")
    lines.append(f"SYSTEM_PROMPT_FILE={shlex.quote(str(sp) if sp else '')}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    conf_path = sys.argv[1]
    arg = sys.argv[2]
    try:
        conf = load_conf(conf_path)
    except FileNotFoundError:
        sys.exit(f"找不到配置文件: {conf_path}")
    except json.JSONDecodeError as e:
        sys.exit(f"配置文件格式错误 ({conf_path}): {e}")

    if arg == "--list":
        for name in conf:
            if not name.startswith("_"):
                print(name)
        return

    if arg == "--plan":
        names = [x.strip() for x in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if x.strip()]
        prev = None
        for n in names:
            if n not in conf:
                sys.exit(f"配置里没有 benchmark: {n}")
            fp = server_fp(conf[n])
            print(f"{n} {1 if fp != prev else 0}")
            prev = fp
        return

    if arg not in conf:
        available = ", ".join(k for k in conf if not k.startswith("_"))
        sys.exit(f"配置里没有 benchmark: {arg}\n可用: {available}")

    print(shell_out(arg, arg, conf[arg]))


if __name__ == "__main__":
    main()
