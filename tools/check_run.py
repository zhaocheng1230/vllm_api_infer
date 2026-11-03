#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_run.py - 任务健康检查

看一件事：这一轮跑得健不健康。
  1) 实例负载均衡（有没有哪个实例被灌爆）
  2) 请求耗时分布（nginx 视角）
  3) 超时统计（客户端 + 结果文件）
  4) 吞吐与时延（客户端 [System] 行）
  5) nginx 上游重试（有没有重复投递）

用法:
    python3 tools/check_run.py                       # 自动找最新任务目录
    python3 tools/check_run.py <任务目录>             # 指定任务目录
    python3 tools/check_run.py <任务目录> <客户端日志>  # 额外指定客户端 stdout 日志

任务目录长这样:
    logs/<date>/<model>/<benchmark>/
        vllm_<时间戳>_<端口>.log     每个 vLLM 实例一份
        nginx_access.log             nginx 访问日志
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"

# 判定阈值
IMBALANCE_WARN = 0.20   # 实例间峰值差异超过 20% 视为不均衡
IMBALANCE_BAD = 0.40    # 超过 40% 视为严重（一个实例灌满、其余不到一半）
TIMEOUT_WARN = 1        # 超时条数


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def latest_task_dir() -> Optional[Path]:
    """最新任务目录（优先带 benchmark 层的，深度 3）。"""
    if not LOGS.is_dir():
        return None
    for depth in (3, 2):
        cands = [
            p for p in LOGS.glob("*/" * (depth - 1) + "*")
            if p.is_dir() and any(p.glob("vllm_*.log"))
        ]
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def parse_vllm_logs(task_dir: Path, only_file: Optional[Path] = None,
                    extra_files: Optional[List[Path]] = None) -> Dict[str, Dict]:
    """每个实例的 Running / Waiting 统计。

    only_file 不为空时只解析这一个日志（单实例场景）。
    """
    stats: Dict[str, Dict] = {}
    if only_file is not None:
        files = [only_file] + list(extra_files or [])
    elif extra_files:
        files = list(extra_files)
    else:
        files = sorted(task_dir.glob("vllm_*.log"))
    for f in files:
        m = re.search(r"_(\d+)\.log$", f.name)
        port = m.group(1) if m else f.stem
        running, waiting, gen_tps = [], [], []
        # 三个字段在日志里的先后顺序会随 vLLM 版本变，分别独立匹配更稳
        pat_run = re.compile(r"Running:\s*(\d+)\s*reqs")
        pat_wait = re.compile(r"Waiting:\s*(\d+)\s*reqs")
        pat_gen = re.compile(r"Avg generation throughput:\s*([\d.]+)")
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m_run = pat_run.search(line)
                    m_wait = pat_wait.search(line)
                    if not (m_run and m_wait):
                        continue
                    running.append(int(m_run.group(1)))
                    waiting.append(int(m_wait.group(1)))
                    m_gen = pat_gen.search(line)
                    if m_gen:
                        gen_tps.append(float(m_gen.group(1)))
        except OSError:
            continue
        if running:
            stats[port] = {
                "running": running,
                "waiting": waiting,
                "gen_tps": gen_tps,
                "peak": max(running),
                "wait_peak": max(waiting),
                "avg_running": sum(running) / len(running),
                "avg_gen_tps": sum(gen_tps) / len(gen_tps) if gen_tps else 0.0,
            }
    return stats


def parse_nginx(task_dir: Path) -> Dict:
    """nginx 访问日志：$status $upstream_addr $request_time $upstream_response_time"""
    out = {
        "total": 0, "per_upstream": defaultdict(int),
        "rt": [], "retries": 0, "status": defaultdict(int),
    }
    f = task_dir / "nginx_access.log"
    if not f.exists():
        return out
    # 日志尾部: $status $upstream_addr $request_time $upstream_response_time
    # $upstream_addr 里可能带逗号+空格（nginx 换过上游），按空白切字段会错位，
    # 所以用正则从行尾锚定。
    tail = re.compile(
        r"\s(\d{3})\s+((?:\S+:\d+)(?:,\s*\S+:\d+)*)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = tail.search(line)
                if not m:
                    continue
                out["total"] += 1
                out["status"][m.group(1)] += 1
                up = m.group(2)
                # 重试时 $upstream_addr 是 "a, b" 形式，b 才是真正服务的那个
                chain = [x.strip() for x in up.split(",")]
                out["per_upstream"][chain[-1]] += 1
                if len(chain) > 1:
                    out["retries"] += 1
                try:
                    out["rt"].append(float(m.group(3)))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def parse_client_log(path: Optional[Path]) -> Dict:
    """客户端 stdout 日志：超时次数 + [System] 汇总行。"""
    out = {"timeouts": 0, "retry_lines": 0, "system": None, "found": False}
    if path is None or not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                out["found"] = True
                if "Request timed out" in line or "timed out" in line:
                    out["timeouts"] += 1
                if "[Retry]" in line:
                    out["retry_lines"] += 1
                if line.startswith("[System]"):
                    out["system"] = line.strip()
    except OSError:
        pass
    return out


def find_output_files(task_dir: Path) -> List[Path]:
    """找结果文件。

    任务跑的过程中，正式 jsonl 还没生成，只有流式落盘的 *.jsonl.tmp；
    跑完后正式 jsonl 才有，tmp 被删掉。两个都要找。
    结果文件也可能落在仓库的 output/<benchmark>/ 下（不一定是 output/，以配置为准）。
    """
    tried: List[Path] = []
    candidates = [
        task_dir,
        REPO / "output" / task_dir.name,
        REPO / "output",
    ]
    files: List[Path] = []
    for d in candidates:
        tried.append(d)
        if not d.is_dir():
            continue
        # 递归找：输出目录名不一定等于 benchmark 名（以配置为准）
        found = sorted(d.rglob("*_output.jsonl")) + sorted(d.rglob("*_output.jsonl.tmp"))
        if found:
            files = found
            break
    find_output_files.tried = tried  # type: ignore[attr-defined]
    return files


def parse_output_jsonl(task_dir: Path) -> Dict:
    """结果文件：成功/失败/超时统计。"""
    out = {"total": 0, "success": 0, "timeout": 0, "errors": defaultdict(int), "found": False}
    for f in find_output_files(task_dir):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out["found"] = True
                    out["total"] += 1
                    if rec.get("success"):
                        out["success"] += 1
                    else:
                        err = (rec.get("error") or "")[:50] or "unknown"
                        out["errors"][err] += 1
                        if "timed out" in err or "timeout" in err.lower():
                            out["timeout"] += 1
        except OSError:
            continue
    return out


def kv_trend(log_file: Path) -> List[float]:
    """从 vLLM 日志里取 GPU KV cache 占用率序列。"""
    pat = re.compile(r"GPU KV cache usage:\s*([\d.]+)%")
    vals: List[float] = []
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    vals.append(float(m.group(1)))
    except OSError:
        pass
    return vals


def count_in(log_file: Path, keyword: str) -> int:
    """日志里关键字出现次数（忽略大小写）。"""
    n = 0
    k = keyword.lower()
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if k in line.lower():
                    n += 1
    except OSError:
        pass
    return n


def check_compile_cache(task_dir: Path) -> int:
    """编译缓存崩溃（Compiling model again）次数 —— 应该恒为 0。"""
    n = 0
    for f in task_dir.glob("vllm_*.log"):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                n += sum(1 for line in fh if "Compiling model again" in line)
        except OSError:
            continue
    return n


def main() -> None:
    single_log: Optional[Path] = None
    extra_logs: List[Path] = []
    if len(sys.argv) >= 2:
        arg = Path(sys.argv[1]).resolve()
        if arg.is_file():
            # 直接给了一个 vllm 日志：单实例场景
            single_log = arg
            task_dir = arg.parent
            # 额外再传的日志文件也一并分析（多实例散落在同一目录的场景）
            extra_logs = [Path(a).resolve() for a in sys.argv[2:]
                          if Path(a).is_file() and a.endswith(".log")]
        else:
            task_dir = arg
    else:
        task_dir = latest_task_dir()
        if task_dir is None:
            print("[Error] 找不到任务目录（logs/<date>/<model>/<bench>/），先跑一轮任务")
            sys.exit(1)

    if not task_dir.is_dir():
        print(f"[Error] 目录不存在: {task_dir}")
        sys.exit(1)

    client_log = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    print("=" * 72)
    print(f" 任务健康检查: {task_dir}")
    print("=" * 72)

    problems: List[str] = []
    skipped: List[str] = []

    # ---------- 1) 负载均衡 ----------
    print("\n【1】实例负载均衡")
    stats = parse_vllm_logs(task_dir, single_log, extra_logs)
    if not stats:
        if not list(task_dir.glob("vllm_*.log")):
            print("  ⚠️  没找到 vllm_*.log，这项无法检查")
        else:
            print("  ⚠️  找到 vllm 日志，但没有 Running:/Waiting: 调度指标行，这项无法检查")
        skipped.append("实例负载均衡（无 vLLM 调度指标）")
    else:
        peaks = {p: s["peak"] for p, s in stats.items()}
        max_peak = max(peaks.values()) or 1
        min_peak = min(peaks.values())
        print(f"  {'实例':>8} {'Running峰值':>11} {'Running均值':>11} {'Waiting峰值':>11} {'生成tok/s':>10}")
        for port, s in sorted(stats.items(), key=lambda kv: -kv[1]["peak"]):
            print(f"  {port:>8} {s['peak']:>11} {s['avg_running']:>11.1f} "
                  f"{s['wait_peak']:>11} {s['avg_gen_tps']:>10.1f}")
        ratio = (max_peak - min_peak) / max_peak if max_peak else 0
        if ratio > IMBALANCE_BAD:
            print(f"  → ❌ 严重不均衡（峰值差异 {ratio*100:.0f}%）")
            problems.append(f"实例负载严重不均衡（差异 {ratio*100:.0f}%）")
        elif ratio > IMBALANCE_WARN:
            print(f"  → ⚠️  轻度不均衡（峰值差异 {ratio*100:.0f}%）")
        else:
            print(f"  → ✅ 均衡（峰值差异 {ratio*100:.0f}%）")

    # ---------- 2) 请求耗时 ----------
    print("\n【2】请求耗时（nginx 视角）")
    ng = parse_nginx(task_dir)
    if not ng["total"]:
        if not (task_dir / "nginx_access.log").exists():
            print("  ⚠️  没找到 nginx_access.log，这项无法检查")
        else:
            print("  ⚠️  nginx_access.log 是空的，这项无法检查")
        skipped.append("请求耗时 / 上游重试（无 nginx 访问日志）")
    else:
        rt = ng["rt"]
        print(f"  样本 {ng['total']} 条 | p50={_pct(rt,.5):.1f}s  p95={_pct(rt,.95):.1f}s  "
              f"max={max(rt) if rt else 0:.1f}s")
        print("  各上游请求数:")
        for up, c in sorted(ng["per_upstream"].items(), key=lambda kv: -kv[1]):
            print(f"    {up:<28} {c}")
        print(f"  上游重试（重复投递）: {ng['retries']} 次"
              + ("  → ✅ 无重复投递" if ng["retries"] == 0 else "  → ⚠️  有重复投递"))
        if ng["retries"]:
            problems.append(f"nginx 上游重试 {ng['retries']} 次")
        codes = {k: v for k, v in ng["status"].items() if k != "200"}
        if codes:
            print(f"  非 200 状态: {dict(codes)}")

    # ---------- 3) 超时 ----------
    print("\n【3】超时统计")
    cl = parse_client_log(client_log)
    res = parse_output_jsonl(task_dir)
    if cl["found"]:
        print(f"  客户端日志 {client_log}")
        print(f"    超时行数: {cl['timeouts']}")
    else:
        print("  （未指定客户端 stdout 日志，跳过该项）")
    if res["total"]:
        print(f"  结果文件: 共 {res['total']} 条，成功 {res['success']}，失败 {res['total']-res['success']}")
        print(f"    其中超时: {res['timeout']}")
        if res["errors"]:
            print("    失败原因 Top5:")
            for err, c in sorted(res["errors"].items(), key=lambda kv: -kv[1])[:5]:
                print(f"      {c:>6}  {err}")
    elif not res["found"]:
        tried = getattr(find_output_files, "tried", [])
        print("  ⚠️  没找到结果文件，这项无法检查")
        for d in tried:
            print(f"        找过: {d}")
        skipped.append("超时统计（无结果文件）")
    timeout_total = cl["timeouts"] + res["timeout"]
    if timeout_total:
        print(f"  → ⚠️  共 {timeout_total} 条超时")
        problems.append(f"{timeout_total} 条超时")
    else:
        print("  → ✅ 无超时")

    # ---------- 4) 吞吐与时延 ----------
    print("\n【4】吞吐与时延")
    if cl["system"]:
        print(f"  {cl['system']}")
    else:
        print("  ⚠️  没拿到客户端 [System] 行（未指定客户端日志，或任务还在跑）")
        skipped.append("吞吐与时延（无客户端 [System] 行）")

    # ---------- 5) 编译缓存 + KV/抢占（单实例时尤其要看）----------
    print("\n【5】编译缓存 / KV / 抢占")
    kv_files = [single_log] if single_log else extra_logs
    if not kv_files:
        kv_files = sorted(task_dir.glob("vllm_*.log"))
    if kv_files:
        all_kv: List[float] = []
        pre_total = 0
        for kf in kv_files:
            kv = kv_trend(kf)
            if kv:
                all_kv += kv
            pre_total += count_in(kf, "preempt")
        if all_kv:
            print(f"  KV cache 占用: 最早 {all_kv[0]:.1f}%  ->  最新 {all_kv[-1]:.1f}%"
                  f"   峰值 {max(all_kv):.1f}%")
            if max(all_kv) > 85:
                print("  → ⚠️  KV cache 逼近上限，很可能已经在抢占（preemption）")
                problems.append(f"KV cache 峰值 {max(all_kv):.1f}%（可能触发抢占）")
            elif max(all_kv) > 60:
                print("  → ⚠️  KV cache 偏高，注意观察")
        print(f"  抢占(preempt)记录: {pre_total} 次"
              + ("  → ✅ 没有抢占" if pre_total == 0 else "  → ⚠️ 有抢占"))
        if pre_total:
            problems.append(f"发生 {pre_total} 次抢占")
    n = check_compile_cache(task_dir)
    if n:
        print(f"  → ⚠️  'Compiling model again' 出现 {n} 次（编译缓存又崩了）")
        problems.append(f"编译缓存崩溃 {n} 次")
    else:
        print("  → ✅ 无编译缓存崩溃")

    # ---------- 总结 ----------
    print("\n" + "=" * 72)
    if problems:
        print(" 结论: 存在问题")
        for p in problems:
            print(f"   - {p}")
    elif skipped:
        print(" 结论: ⚠️  已检查的项正常，但有项无法检查")
        for p in skipped:
            print(f"   - {p}")
    else:
        print(" 结论: ✅ 全部正常")
    print("=" * 72)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
