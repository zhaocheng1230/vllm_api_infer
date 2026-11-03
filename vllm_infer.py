#!/usr/bin/env python3
# vllm_infer.py - 异步批量调用本地 vLLM/Nginx 服务，为 jsonl 文件生成 output
#
# 用法:
#   # benchmark 模式：输入文件和客户端参数都从 configs/benchmarks.json 读
#   python vllm_infer.py --benchmarks gsm8k_zh --model MyModel
#
#   # 单文件模式
#   python vllm_infer.py --input ./datasets/gsm8k_zh.jsonl --model MyModel

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai
from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ========================== benchmark 配置 ==========================
# 客户端参数：命令行显式传的 > 配置里的 client 段 > 这里的内置默认值。
# argparse 里这批参数的 default 统一设成 None，就是为了区分
# 「用户没传」和「用户传了 0.0 / -1 这种本身是合法值的参数」。
CLIENT_PARAM_DEFAULTS: Dict[str, Any] = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "presence_penalty": 0.0,
    "system_prompt": None,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "benchmarks.json"


def load_benchmarks(conf_path: Path) -> Dict[str, Any]:
    """读 benchmark 配置文档。"""
    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] 配置文件不存在: {conf_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[Error] 配置文件格式错误 ({conf_path}): {e}")
        sys.exit(1)


def pick(cli_value: Any, cfg_value: Any, default: Any) -> Any:
    """取值优先级：命令行显式值 > 配置值 > 内置默认值。"""
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


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
        "--input", default=None,
        help="输入 jsonl 文件路径（单文件模式）。benchmark 模式由配置文档给出",
    )
    parser.add_argument(
        "--benchmarks", default=None,
        help="要跑的 benchmark 名，逗号分隔。输入文件和客户端参数都从配置文档读",
    )
    parser.add_argument(
        "--benchmark_conf", default=str(DEFAULT_CONFIG_PATH),
        help="benchmark 配置文件路径",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="输出目录（默认 <项目>/output/<日期>/<模型名>）",
    )
    parser.add_argument(
        "--host", default="http://localhost:1358/v1",
        help="vLLM / OpenAI 兼容接口 base_url",
    )
    parser.add_argument("--model", required=True, help="模型名（需与 served-model-name 一致）")
    parser.add_argument("--key", default="EMPTY", help="API Key")
    parser.add_argument("--concurrency", type=int, default=64, help="并发请求数")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument(
        "--read_timeout",
        type=int,
        default=0,
        help="单请求读超时（秒）。0=自动按 max_tokens 推算，见 auto_read_timeout()",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--system_prompt", default=None, help="统一系统提示词")
    parser.add_argument(
        "--max_model_len", type=int, default=0,
        help="服务端的 max_model_len。只用于校验 client.max_tokens 是否超限（超了服务端直接返 400）",
    )
    parser.add_argument("--resume", action="store_true", help="已完整处理过的文件直接跳过")
    parser.add_argument(
        "--save_jsonl", action=argparse.BooleanOptionalAction, default=True,
        help="是否同时保存包含完整字段的 .jsonl 结果（默认开启）",
    )
    args = parser.parse_args()

    # 快照命令行的客户端参数：跑每个 benchmark 前都要从这个快照重新算，
    # 否则上一个 benchmark 的配置会残留到下一个。
    args.cli_client = {k: getattr(args, k) for k in CLIENT_PARAM_DEFAULTS}

    # 输出目录默认值：<项目>/output/<日期>/<模型名>
    # 日期优先用 run_pipeline.sh 传下来的 DATE_STR，保证两边落在同一个目录。
    if args.output_dir is None:
        date_str = os.environ.get("DATE_STR") or time.strftime("%Y%m%d")
        args.output_dir = str(project_dir / "output" / date_str / args.model)
    args.project_dir = str(project_dir)
    return args


def _pct(sorted_values: List[float], p: float) -> float:
    """分位数。传入的必须已排序。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


# ========================== 超时推算 ==========================
# 长输出场景（max_tokens 很大）下单请求耗时 = 排队 + 生成：
#   生成时间上界 ≈ max_tokens ÷ 单请求生成速率
#   单请求速率   ≈ 服务端总吞吐 ÷ 并发数
# 910B4 + Qwen3.5-4B 实测（2卡满负载）：2061 tok/s ÷ 2 实例 ÷ 128 并发 ≈ 8 tok/s/req
# 8 卡乐观值约 31 tok/s/req。这里取保守下界，并留 4 个 slot 的排队余量。
CONSERVATIVE_TPS_PER_REQ = 8
QUEUE_SLOT_MARGIN = 4
BASE_READ_TIMEOUT = 1800


def auto_read_timeout(max_tokens: int) -> int:
    """按 max_tokens 推算读超时（秒）。

    固定 1800s 会在「max_tokens 很大 + 高并发」时必然超时：
    单请求生成时间本身就能到 2000s，再叠加服务端排队，轻松超过阈值。
    """
    if max_tokens <= 0:
        return BASE_READ_TIMEOUT
    est = int(max_tokens / CONSERVATIVE_TPS_PER_REQ * (1 + QUEUE_SLOT_MARGIN))
    return max(BASE_READ_TIMEOUT, est)


class VLLMInfer:
    def __init__(self, args):
        self.args = args
        self.client_async = None
        self.input_base = Path(".")  # 输入文件相对输出目录的基准路径，每个文件处理前重设
        self.bench_configs: Dict[str, Any] = {}  # benchmark 名 -> 配置节
        self.total_count = 0
        self.success_count = 0
        # 系统级指标（跨文件累计）
        self.t_wall_start: Optional[float] = None
        self.wall_s: float = 0.0
        self.total_output_tokens: int = 0
        self.all_ttft: List[float] = []
        self.all_e2e: List[float] = []
        # 重试原因聚合计数：长输出高并发下失败条目可能上千，
        # 逐条 print 会把终端 I/O 变成瓶颈（同步阻塞 event loop）
        self._retry_stats: Dict[str, int] = {}

    def init_client_async(self):
        """初始化异步OpenAI客户端，匹配Nginx/vLLM服务端配置"""
        if self.client_async is None:
            import httpx
            # 读超时必须覆盖「排队 + 生成」的全过程。
            # 固定 1800s 在 max_tokens 很大时会必然超时（单请求生成本身就能到 2000s），
            # 超时后重试又会占着服务端的 slot，形成雪崩。
            if getattr(self.args, "read_timeout", 0) and self.args.read_timeout > 0:
                read_timeout = float(self.args.read_timeout)
            else:
                read_timeout = float(auto_read_timeout(self.args.max_tokens))

            # 连接池上限必须 >= 并发数。否则超出部分的请求会在 httpx 内部排队，
            # 表现就是"把 --concurrency 调大却没反应"。
            #
            # 但只留「并发数 + 一点点余量」是不够的：读超时重试时，旧连接还占着
            # 池位（nginx 的 proxy_read_timeout 是 86400s，不会掐断，服务端那边
            # 的 slot 也还在跑），客户端要新建连接发起重试 -> 需要额外余量。
            #
            # 实测（910B4 + 4 实例）：
            #   并发 256 -> pool_size=512，余量 256 -> 重试 5 次
            #   并发 512 -> pool_size=544，余量  32 -> 重试 200+ 次（余量被挤爆，
            #              新请求在池里等到 pool timeout(120s) -> PoolTimeout，
            #              openai 把它映射成 APITimeoutError -> 雪崩）
            # 所以余量按并发的倍数给。
            pool_size = max(512, self.args.concurrency * 2 + 64)

            # 把超时阈值打出来：超时类问题排查时，第一个要看的就是这个数
            print(f"[Config] read_timeout={read_timeout:.0f}s "
                  f"(max_tokens={self.args.max_tokens}) | "
                  f"连接池上限={pool_size} | 并发={self.args.concurrency}")

            self.client_async = AsyncOpenAI(
                api_key=self.args.key,
                base_url=self.args.host,
                max_retries=3,  # ← 从2提升到3，增加网络抖动容错
                timeout=httpx.Timeout(
                    # 连接超时不能太小：并发高 + 大量重试时瞬时要建上千条连接，
                    # nginx 的 SYN 队列一满内核就丢 SYN，TCP 握手要重传
                    # （1s/2s/4s…），30s 建不上就 ConnectTimeout。
                    # 实测 910B4 + 4 实例 + 并发 512：200+ 次 ConnectTimeout。
                    connect=120.0,
                    read=read_timeout,
                    write=max(60.0, read_timeout),
                    # 等池位的时间不能比等响应的时间还短：池位紧张时（重试挤占）
                    # 120s 就会先超时，把「排队慢」误报成「请求失败」。
                    pool=max(120.0, read_timeout)
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
        }

    def _bump_retry(self, kind: str, exc: Exception) -> None:
        """重试原因聚合计数。

        逐条 print 在「长输出 + 高并发 + 大量失败」时会把终端 I/O 变成瓶颈：
        print 是同步阻塞调用，卡住的就是整个 event loop。
        这里只在前 5 条和每 100 条时打一次汇总。

        openai SDK 把 httpx 的各种超时全都包装成 APITimeoutError，光看类型分不出
        到底是哪一种（ReadTimeout=服务端生成慢 / ConnectTimeout=连不上 /
        PoolTimeout=连接池挤爆 / RemoteProtocolError=服务端断开）。
        原始异常在 __cause__ 里，必须打出来才能判断。
        """
        cause = exc.__cause__
        raw = type(cause).__name__ if cause is not None else "-"
        key = f"{kind}/{type(exc).__name__}/{raw}"
        self._retry_stats[key] = self._retry_stats.get(key, 0) + 1
        total = sum(self._retry_stats.values())
        if total <= 5 or total % 100 == 0:
            detail = " ".join(f"{k}×{v}" for k, v in sorted(self._retry_stats.items()))
            print(f"[Retry] 累计 {total} 次 | {detail}")

    async def _process_item_async(
        self,
        item: Dict[str, Any],
        i: int,
        semaphore: asyncio.Semaphore,
        output_queue: Optional["asyncio.Queue"] = None,
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
                    self._bump_retry("net", e)

                except Exception as e:
                    repeat += 1
                    last_error = str(e)
                    self._bump_retry("biz", e)

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
        # 流式落盘：完成的条目立刻进队列，任务中断也不丢已完成的结果
        if output_queue is not None:
            await output_queue.put({"_idx": i, **item})
        return i, item

    async def _writer_worker(self, output_queue: "asyncio.Queue", output_path: Path):
        """后台写盘协程：把同步文件 IO 挪进线程，不阻塞 event loop。

        为什么必须挪走：长输出场景下一条 output 可能几十 KB，
        而 NAS/NFS 上一次 write+flush 的延迟足以卡住整个 event loop，
        连带把在飞请求的读超时拖爆。
        """
        def _write(line: str) -> None:
            f.write(line)
            f.flush()

        f = open(output_path, "a", encoding="utf-8")
        try:
            while True:
                item = await output_queue.get()
                try:
                    await asyncio.to_thread(
                        _write, json.dumps(item, ensure_ascii=False) + "\n"
                    )
                except Exception as e:
                    print(f"[Writer Error] 写盘失败: {e}")
                finally:
                    output_queue.task_done()
        finally:
            f.close()

    async def _run_api_call_async(
        self,
        data: List[Tuple[int, Dict[str, Any]]],
        concurrency: int,
        stream_path: Optional[Path] = None,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """为给定的 (原始索引, 数据) 列表发起全部异步请求，按完成顺序收集结果。

        stream_path 不为空时，每条完成的条目会立刻追加写入该文件（流式落盘）。
        """
        self.init_client_async()
        semaphore = asyncio.Semaphore(concurrency)

        output_queue: "asyncio.Queue" = asyncio.Queue()
        writer_task = None
        if stream_path is not None:
            writer_task = asyncio.create_task(
                self._writer_worker(output_queue, stream_path)
            )

        tasks = [
            self._process_item_async(item, i, semaphore, output_queue)
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

        if writer_task is not None:
            await output_queue.join()
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass

        if self.client_async:
            await self.client_async.close()
            self.client_async = None

        return results

    def _process_file(self, input_path: Path, output_dir: Path, name: str = ""):
        """处理单个 jsonl 文件：读入 -> 推理 -> 按原始顺序写出 .txt（必须）与 .jsonl（可选）

        name 不为空时用它作为输出文件名（benchmark 模式）。
        直接用输入文件名有个坑：不同 benchmark 的输入文件可能同名
        （比如都叫 data.jsonl），写到一个输出目录里会互相覆盖。
        """
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

        # 输入路径唯一，直接读这个文件（不再支持目录扫描）
        self.input_base = input_path.parent
        out_base = output_dir / (name or input_path.stem)
        out_base.parent.mkdir(parents=True, exist_ok=True)

        txt_path = out_base.parent / (out_base.name + "_output.txt")
        jsonl_path = out_base.parent / (out_base.name + "_output.jsonl")
        # 流式落盘的临时文件：每条完成立刻追加写在这里，任务中断也不丢已完成的结果。
        # 正常跑完后再合并成正式的 jsonl（按原始顺序）并删除它。
        stream_path = out_base.parent / (out_base.name + "_output.jsonl.tmp")

        # 断点续跑：读回上次结果，只重跑「没结果」和「上次失败」的条目。
        # （原逻辑是「输出行数 == 输入行数就跳过」，但失败条目也会写一行空串，
        #   行数永远相等 → 失败条目一次都不会重跑。）
        prev_results: Dict[int, Dict[str, Any]] = {}
        if self.args.resume:
            # 优先读流式临时文件 —— 那是「上次跑崩了」的现场，内容最全；
            # 没有才退回读正式 jsonl。
            for src in (stream_path, jsonl_path):
                if not src.exists():
                    continue
                try:
                    with open(src, "r", encoding="utf-8") as f:
                        for n, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            idx = rec.pop("_idx", n)
                            if rec.get("success"):
                                prev_results[idx] = rec
                    if prev_results:
                        print(f"[Resume] 复用上次成功结果 {len(prev_results)} 条（来源 {src.name}）")
                        break
                except Exception:
                    prev_results = {}

        todo = [(i, it) for i, it in enumerate(data) if i not in prev_results]

        # 新一轮的流式落盘从头写，避免和上一轮的残留混在一起
        if stream_path.exists():
            stream_path.unlink()

        results = asyncio.run(
            self._run_api_call_async(todo, self.args.concurrency, stream_path)
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

        # 正式文件已落盘，临时文件可以删了
        if stream_path.exists():
            stream_path.unlink()

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

        # 累加给 run() 做全局判断
        self.total_count += len(results)
        self.success_count += success_count
        for _, item in results:
            if not item.get("success"):
                continue
            self.total_output_tokens += int(item.get("output_token_len") or 0)
            if item.get("ttft"):
                self.all_ttft.append(float(item["ttft"]))
            if item.get("output_time"):
                self.all_e2e.append(float(item["output_time"]))

    def _build_jobs(self) -> List[Tuple[str, Path]]:
        """决定要跑哪些 (benchmark 名, 输入文件)。

        - --benchmarks a,b,c : 输入文件和客户端参数都来自配置文件
        - --input file        : 单文件模式，benchmark 名为空
        """
        if self.args.benchmarks:
            conf = load_benchmarks(Path(self.args.benchmark_conf))
            project_dir = Path(self.args.project_dir)
            jobs: List[Tuple[str, Path]] = []
            names = [x.strip() for x in self.args.benchmarks.split(",") if x.strip()]
            if not names:
                print("[Error] --benchmarks 后面是空的")
                sys.exit(1)
            for name in names:
                if name not in conf:
                    avail = ", ".join(k for k in conf if not k.startswith("_"))
                    print(f"[Error] 配置里没有 benchmark: {name}\n可用: {avail}")
                    sys.exit(1)
                cfg = conf[name]
                rel = (cfg.get("dataset", {}) or {}).get("input_file", "")
                if not rel:
                    print(f"[Error] benchmark '{name}' 没配 dataset.input_file")
                    sys.exit(1)
                path = Path(rel)
                if not path.is_absolute():
                    path = project_dir / path
                jobs.append((name, path))
            self.bench_configs = conf
            return jobs

        if not self.args.input:
            print("[Error] 缺少输入：--benchmarks 和 --input 二选一")
            sys.exit(1)
        return [("", Path(self.args.input))]

    def _apply_client_config(self, bench_name: str, cfg: Dict[str, Any]) -> None:
        """把配置里的 client 段套到 args 上（命令行显式传的优先）。

        每个 benchmark 都要重来一遍，否则上一个 benchmark 的参数会残留到下一个。
        """
        client = (cfg.get("client", {}) or {})
        for key, default in CLIENT_PARAM_DEFAULTS.items():
            setattr(self.args, key, pick(
                self.args.cli_client.get(key), client.get(key), default
            ))

        # max_tokens 不能超过服务端的 max_model_len，否则服务端对每个请求
        # 直接返回 400，整批全失败 —— 这种错留到运行时才发现代价太大。
        mml = self.args.max_model_len
        if mml and self.args.max_tokens and self.args.max_tokens > mml:
            print(
                f"[Error] benchmark '{bench_name}': "
                f"client.max_tokens({self.args.max_tokens}) > 服务端 max_model_len({mml})\n"
                f"       服务端会对每个请求直接返回 400，整批全失败。\n"
                f"       请调小 max_tokens，或调大 run_pipeline.sh 的 max_model_len。"
            )
            sys.exit(1)

    def run(self):
        output_dir = Path(self.args.output_dir)
        jobs = self._build_jobs()

        for name, path in jobs:
            if not path.is_file():
                print(f"[Error] 输入文件不存在: {path}")
                print("       配置里的 dataset.input_file 必须指向具体的 .jsonl 文件")
                sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_dir}")

        self.t_wall_start = time.perf_counter()
        for name, path in jobs:
            # 每个 benchmark 都重新解析一遍客户端参数：
            # 单文件模式没有配置节，就用内置默认值。
            self._apply_client_config(name, self.bench_configs.get(name, {}))
            if name:
                print(
                    f"\n[BENCH] {name} | max_tokens={self.args.max_tokens} "
                    f"temperature={self.args.temperature}"
                )
            self._process_file(path, output_dir, name)
        self.wall_s = time.perf_counter() - self.t_wall_start

        print("\n[All Done]")

        # ---- 系统级指标（这才是能横向对比的数）----
        # 注意：上面每条请求自己的 tps 是「单请求速率」，并发越高它必然越低
        # （请求互相排队），不能用它做 A/B。这里给的是整批的真实吞吐。
        if self.wall_s > 0:
            sys_tps = self.total_output_tokens / self.wall_s
            req_ps = self.total_count / self.wall_s
            print(
                f"[System] 墙钟 {self.wall_s:.1f}s | "
                f"输出 {self.total_output_tokens} tok | "
                f"系统吞吐 {sys_tps:.1f} tok/s | "
                f"请求吞吐 {req_ps:.2f} req/s"
            )
            if self.all_ttft:
                tt = sorted(self.all_ttft)
                print(
                    f"[System] TTFT p50 {_pct(tt, 50):.3f}s | "
                    f"p95 {_pct(tt, 95):.3f}s | "
                    f"端到端 p50 {_pct(sorted(self.all_e2e), 50):.2f}s | "
                    f"p95 {_pct(sorted(self.all_e2e), 95):.2f}s"
                )

        # 全军覆没 = 服务/网络/参数出了系统性问题，不能当成跑成功。
        # 部分失败是正常的（个别条目出错），不因此中断。
        if self.total_count > 0 and self.success_count == 0:
            print(
                f"[Error] {self.total_count} 条全部失败，成功 0 条 —— "
                f"这不是正常结果，退出码置 1 以免上层误判成「跑成功」"
            )
            sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    infer = VLLMInfer(args)
    infer.run()
