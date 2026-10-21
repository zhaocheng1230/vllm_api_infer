#!/usr/bin/env python3
# vllm_infer.py - 异步批量调用本地 vLLM/Nginx 服务，为 jsonl 文件生成 output
# 用法:
#   python vllm_infer.py \
#       --input_dir ./data \
#       --output_dir ./output \
#       --host http://localhost:1358/v1 \
#       --model MyModel \
#       --concurrency 64

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import openai
from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def build_chat_messages(args, prompt):
    """将 prompt 字段转换为 OpenAI messages 格式"""
    if isinstance(prompt, list):
        # 已经是 messages 列表
        if all(isinstance(m, dict) and "role" in m for m in prompt):
            messages = list(prompt)
            if args.system_prompt and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": args.system_prompt})
            return messages
        # 字符串列表，拼成一段文本
        prompt = "\n".join(str(x) for x in prompt)

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="vLLM 异步批量推理客户端")
    parser.add_argument(
        "--input_dir", default=str(project_dir / "datasets"),
        help="输入 jsonl 文件所在目录（默认 <项目>/datasets）",
    )
    parser.add_argument(
        "--output_dir", default=str(project_dir / "output"),
        help="输出目录（默认 <项目>/output）",
    )
    parser.add_argument(
        "--host", default="http://localhost:1358/v1",
        help="vLLM / OpenAI 兼容接口 base_url",
    )
    parser.add_argument("--model", required=True, help="模型名（需与 served-model-name 一致）")
    parser.add_argument("--key", default="EMPTY", help="API Key")
    parser.add_argument("--concurrency", type=int, default=64, help="并发请求数")
    parser.add_argument("--max_tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--system_prompt", default=None, help="统一系统提示词")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--resume", action="store_true", help="已完整处理过的文件直接跳过")
    parser.add_argument(
        "--save_jsonl", action=argparse.BooleanOptionalAction, default=True,
        help="是否同时保存包含完整字段的 .jsonl 结果（默认开启）",
    )
    return parser.parse_args()


class VLLMInfer:
    def __init__(self, args):
        self.args = args
        self.client_async = None

    def init_client_async(self):
        """初始化异步OpenAI客户端，匹配Nginx/vLLM服务端配置"""
        if self.client_async is None:
            import httpx
            # 连接池上限必须 >= 并发数。否则超出部分的请求会在 httpx 内部排队，
            # 表现就是"把 --concurrency 调大却没反应"。
            # 写死数值（150 / 512）只是把天花板挪了个位置，这里跟着并发走。
            pool_size = max(512, self.args.concurrency + 32)
            self.client_async = AsyncOpenAI(
                api_key=self.args.key,
                base_url=self.args.host,
                max_retries=3,  # ← 从2提升到3，增加网络抖动容错
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=1800.0,
                    write=60.0,
                    pool=120.0  # ← 从60s提升到120s，给连接池更多等待时间
                ),
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=pool_size,
                        max_keepalive_connections=pool_size
                    )
                )
            )

    @staticmethod
    def _get_prompt(item: Any) -> Any:
        """从单条数据中提取 prompt：支持字符串或 messages 列表"""
        if not isinstance(item, dict):
            return str(item)
        for key in ("prompt", "input", "question", "query", "content", "messages"):
            if key in item:
                val = item[key]
                if isinstance(val, (str, list)):
                    return val
                return str(val)
        # 兜底：跳过 id / answer 这类元数据字段，别把它们当 prompt 发出去
        meta_keys = {"id", "_id", "idx", "index", "uid", "answer", "answers",
                     "label", "labels", "target", "gt", "solution", "reason"}
        for k, v in item.items():
            if k in meta_keys:
                continue
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(item, ensure_ascii=False)

    async def chat_completions_async(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """异步流式调用，返回生成内容、首 token 时间、总耗时与 token 数"""
        extra_body = {"top_k": self.args.top_k}
        if "qwen3" in self.args.model.lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        stream = await self.client_async.chat.completions.create(
            model=self.args.model,
            messages=messages,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            presence_penalty=self.args.presence_penalty,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )

        collected_content = []
        first_token_time = None
        output_tokens = 0
        prompt_tokens = 0
        finish_reason = None

        start_time = time.perf_counter()
        async for chunk in stream:
            if chunk.usage:
                output_tokens = chunk.usage.completion_tokens or output_tokens
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens

            if chunk.choices:
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    collected_content.append(chunk.choices[0].delta.content)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if output_tokens == 0 and collected_content:
            output_tokens = len(collected_content)

        # 正常结束的流一定带 finish_reason（stop / length）。
        # 缺它说明连接被中途掐断 —— nginx 后端 failover 时会产生截断流，
        # 这时必须当失败重试，否则半截答案会被当成正确答案写进结果。
        if finish_reason is None:
            raise RuntimeError(
                f"stream ended without finish_reason (truncated), "
                f"got {output_tokens} tokens"
            )

        return {
            "content": "".join(collected_content),
            "ttft": first_token_time - start_time if first_token_time else None,
            "total_time": total_time,
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "tps": output_tokens / total_time if total_time > 0 else 0,
            "finish_reason": finish_reason,
        }

    async def _process_item_async(
        self, item: Dict[str, Any], i: int, semaphore: asyncio.Semaphore
    ) -> Tuple[int, Dict[str, Any]]:
        """处理单条数据：失败自动重试，返回 (原始索引, 更新后的 item)"""
        prompt = self._get_prompt(item)
        messages = build_chat_messages(self.args, prompt)

        response = ""
        prompt_token_len = 0
        input_token_len = 0
        output_token_len = 0
        output_time = 0.0
        ttft = 0.0
        tps = 0.0
        success = False
        last_error = ""
        _res = None

        repeat = 0
        attempts = 0
        while repeat < 3:
            attempts += 1
            async with semaphore:
                try:
                    self.init_client_async()
                    _res = await self.chat_completions_async(messages)
                    response = _res["content"]
                    output_token_len = _res.get("output_tokens", 0)
                    prompt_token_len = _res.get("prompt_tokens", 0)
                    input_token_len = prompt_token_len
                    output_time = _res.get("total_time", 0.0)
                    ttft = _res.get("ttft", 0.0)
                    tps = _res.get("tps", 0.0)

                    success = True
                    break

                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    # 网络抖动值得重试：走和业务错误一样的退避路径
                    repeat += 1
                    last_error = str(e)
                    print(f"[Retry {repeat}/3] Network error for item {i}: {e}")

                except Exception as e:
                    repeat += 1
                    last_error = str(e)
                    print(f"[Retry {repeat}/3] Business error for item {i}: {e}")

            if not success and repeat < 3:
                await asyncio.sleep(2 ** repeat)

        # 每条都 print 会在高并发下把终端 I/O 变成瓶颈，采样打印
        if (i + 1) % 100 == 0:
            print(
                f">>> [{i + 1}] time:{(output_time or 0):.2f}s | "
                f"ttft:{(ttft or 0):.3f}s | tps:{(tps or 0):.1f}"
            )

        item.update({
            "output": response,
            "prompt_token_len": prompt_token_len,
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
            "output_time": output_time,
            "ttft": ttft if ttft is not None else 0.0,
            "tps": tps,
            "success": success,
            "error": last_error if not success else "",
            "attempts": attempts,
        })
        return i, item

    async def _run_api_call_async(
        self, data: List[Tuple[int, Dict[str, Any]]], concurrency: int
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """为给定的 (原始索引, 数据) 列表发起全部异步请求，按完成顺序收集结果"""
        self.init_client_async()
        semaphore = asyncio.Semaphore(concurrency)

        tasks = [
            self._process_item_async(item, i, semaphore)
            for i, item in data
        ]

        results = []
        if tqdm is not None:
            iterator = tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Progress",
                ncols=80,
            )
        else:
            iterator = asyncio.as_completed(tasks)

        for coro in iterator:
            results.append(await coro)

        if self.client_async:
            await self.client_async.close()
            self.client_async = None

        return results

    def _process_file(self, input_path: Path, output_dir: Path):
        """处理单个 jsonl 文件：读入 -> 推理 -> 按原始顺序写出 .txt（必须）与 .jsonl（可选）"""
        print(f"\n[File] {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        data = []
        for line in lines:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warn] 跳过非法 JSON 行: {e}")
                parsed = {"_raw": line}
            if not isinstance(parsed, dict):
                parsed = {"prompt": parsed}
            data.append(parsed)

        if not data:
            print("[Warn] 空文件，跳过")
            return

        rel = input_path.relative_to(Path(self.args.input_dir))
        out_base = output_dir / rel.with_suffix("")
        out_base.parent.mkdir(parents=True, exist_ok=True)

        txt_path = out_base.parent / (out_base.name + "_output.txt")
        jsonl_path = out_base.parent / (out_base.name + "_output.jsonl")

        # 断点续跑：读回上次结果，只重跑「没结果」和「上次失败」的条目。
        # （原逻辑是「输出行数 == 输入行数就跳过」，但失败条目也会写一行空串，
        #   行数永远相等 → 失败条目一次都不会重跑。）
        prev_results: Dict[int, Dict[str, Any]] = {}
        if self.args.resume and jsonl_path.exists():
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for n, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("success"):
                            prev_results[n] = rec
                if prev_results:
                    print(f"[Resume] 复用上次成功结果 {len(prev_results)} 条")
            except Exception:
                prev_results = {}

        todo = [(i, it) for i, it in enumerate(data) if i not in prev_results]
        results = asyncio.run(
            self._run_api_call_async(todo, self.args.concurrency)
        )
        # 合并上次成功的结果
        merged = dict(prev_results)
        for idx, item in results:
            merged[idx] = item
        results = [(i, merged[i]) for i in sorted(merged)]

        with open(txt_path, "w", encoding="utf-8") as ftxt:
            if self.args.save_jsonl:
                with open(jsonl_path, "w", encoding="utf-8") as fjsonl:
                    for _, item in results:
                        output = item.get("output", "")
                        ftxt.write(output.replace("\r", "").replace("\n", " ") + "\n")
                        fjsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                for _, item in results:
                    output = item.get("output", "")
                    ftxt.write(output.replace("\r", "").replace("\n", " ") + "\n")

        success_count = sum(1 for _, item in results if item.get("success"))
        failed_count = len(results) - success_count
        tps_values = [
            item.get("tps", 0) for _, item in results
            if item.get("success") and item.get("tps", 0) > 0
        ]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0

        print(f"[Done] 结果已保存: {txt_path}", end="")
        if self.args.save_jsonl:
            print(f", {jsonl_path}")
        else:
            print()
        print(
            f"[Summary] total={len(results)}, success={success_count}, "
            f"failed={failed_count}, avg_tps={avg_tps:.1f}"
        )

    def run(self):
        input_dir = Path(self.args.input_dir)
        output_dir = Path(self.args.output_dir)

        if not input_dir.exists():
            print(f"[Error] 输入目录不存在: {input_dir}")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        pattern = "**/*.jsonl" if self.args.recursive else "*.jsonl"
        output_resolved = output_dir.resolve()
        files = [
            fp for fp in sorted(input_dir.glob(pattern))
            if output_resolved not in fp.resolve().parents
        ]
        if not files:
            print(f"[Warn] 未找到 .jsonl 文件: {input_dir}")
            return

        print(f"找到 {len(files)} 个输入文件，输出目录: {output_dir}")
        for fp in files:
            self._process_file(fp, output_dir)

        print("\n[All Done]")


if __name__ == "__main__":
    args = parse_args()
    infer = VLLMInfer(args)
    infer.run()
