#!/usr/bin/env python3
"""Concurrent non-stream benchmark (server-true caliber).

Extends benchx to the concurrency dimension for the ablation matrix
(K x concurrency, doc 20): N threads each fire sequential NON-STREAM
completions (no streaming-chunk artifact), report wall time / total tokens /
aggregate tok/s / per-request elapsed percentiles, and diff the server-side
SpecDecoding counters from the server log (windowed counters -> diff).

Usage:
  python3 benchxc.py --url http://127.0.0.1:8000 --model /data/models/Qwen3.5-9B \
      --concurrency 4 --rounds 3 [--max-tokens 128] [--server-log /data/vllm_server.log]
"""
import argparse
import json
import re
import statistics
import threading
import time
import urllib.request

PROMPT = "Write a detailed essay about the history of computing."
METRIC_RE = re.compile(
    r"Mean acceptance length: ([0-9.]+).*?Accepted: (\d+) tokens, Drafted: (\d+) tokens"
)


def read_metrics(server_log: str):
    """Return (acceptance_len, accepted, drafted) from the LAST metrics line."""
    try:
        with open(server_log, encoding="utf-8", errors="replace") as f:
            last = None
            for line in f:
                if "SpecDecoding metrics" in line:
                    last = line
        if last is None:
            return None
        m = METRIC_RE.search(last)
        return (float(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    except OSError:
        return None


def one_request(url: str, model: str, max_tokens: int, timeout: int):
    t0 = time.time()
    body = json.dumps(
        {"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return time.time() - t0, out["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--server-log", default="/data/vllm_server.log")
    args = ap.parse_args()

    # Warmup: fire the same number of requests as we will measure, but
    # concurrently and with a short per-request cap. A serial warmup (the
    # original behaviour) made a single hung request block for the full
    # --timeout, and N of them blocked for N x timeout -- which is what a
    # 600s default turned into here. Warmup failures are reported, not fatal.
    w_timeout = min(args.timeout, 60)
    warm_err: list[str] = []

    def warm():
        try:
            one_request(args.url, args.model, min(16, args.max_tokens), w_timeout)
        except Exception as exc:  # noqa: BLE001
            warm_err.append(str(exc))

    wt = [threading.Thread(target=warm) for _ in range(args.concurrency)]
    for t in wt:
        t.start()
    for t in wt:
        t.join()
    if warm_err:
        print(f"WARN warmup: {len(warm_err)}/{args.concurrency} failed: {warm_err[0][:120]}")
        if len(warm_err) == args.concurrency:
            print("!! all warmup requests failed -- server not serving; abort")
            raise SystemExit(2)
    if args.concurrency > 1:
        # Make sure a metrics window has rolled over since warmup, so the
        # first diff is not polluted by warmup traffic.
        time.sleep(12)

    m_before = read_metrics(args.server_log)

    barrier = threading.Barrier(args.concurrency)
    elapsed: list[float] = []
    tokens: list[int] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()  # start all streams together
        for _ in range(args.rounds):
            dt, tok = one_request(args.url, args.model, args.max_tokens, args.timeout)
            with lock:
                elapsed.append(dt)
                tokens.append(tok)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    time.sleep(12)  # let the 10s metrics window roll over
    m_after = read_metrics(args.server_log)

    total_tok = sum(tokens)
    per_tok = wall / max(total_tok, 1) * 1000
    line = (
        f"[conc={args.concurrency} rounds={args.rounds}] wall={wall:.2f}s "
        f"reqs={len(elapsed)} tokens={total_tok} agg={total_tok/wall:.1f}tok/s "
        f"per_tok={per_tok:.1f}ms "
        f"req_p50={statistics.median(elapsed):.2f}s "
        f"req_p99={sorted(elapsed)[max(0, int(len(elapsed)*0.99)-1)]:.2f}s "
        f"tok_med={statistics.median(tokens):.0f}"
    )
    if m_before and m_after and m_after[2] > m_before[2]:
        iters = m_after[2] - m_before[2]
        acc_len = m_after[0]
        line += (
            f" | iters={iters} acc_len={acc_len:.2f} "
            f"per_iter={wall/iters*1000:.1f}ms"
        )
    elif m_after is None and m_before is None:
        line += " | (k0: no spec metrics)"
    else:
        line += " | (metrics window miss; rerun with --rounds >= 2)"
    print(line)


if __name__ == "__main__":
    main()
