#!/usr/bin/env python3
"""Benchmark: send N requests to /verify with given concurrency."""

import time
import argparse
import concurrent.futures
from pathlib import Path

import requests


def send_one(url: str, image_path: str) -> float:
    """Send one request, return latency in ms."""
    with open(image_path, "rb") as f:
        t0 = time.perf_counter()
        resp = requests.post(url, files={"image": ("test.jpg", f, "image/jpeg")}, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return elapsed


def bench_verify(url: str, image_path: str, n: int, concurrency: int) -> None:
    """Run n requests with given concurrency against /verify."""
    print(f"\nBenchmarking POST /verify  x{n}  concurrency={concurrency}")
    print(f"Target: {url}")
    print(f"Image:  {image_path}\n")

    latencies = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_one, url, image_path) for _ in range(n)]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())

    total = time.perf_counter() - t_start
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"Results:")
    print(f"  Total time:   {total:.2f}s")
    print(f"  Throughput:   {n / total:.1f} req/s")
    print(f"  Avg latency:  {avg:.1f} ms")
    print(f"  P50:          {p50:.1f} ms")
    print(f"  P95:          {p95:.1f} ms")
    print(f"  P99:          {p99:.1f} ms")


def bench_batch(url: str, image_path: str, n: int, batch_size: int) -> None:
    """Send batched requests to /verify_batch."""
    batch_url = url.replace("/verify", "/verify_batch")
    batches = (n + batch_size - 1) // batch_size
    print(f"\nBenchmarking POST /verify_batch  x{n}  batch_size={batch_size}  batches={batches}")

    t_start = time.perf_counter()
    for _ in range(batches):
        files = [("images", ("test.jpg", open(image_path, "rb"), "image/jpeg")) for _ in range(batch_size)]
        t0 = time.perf_counter()
        resp = requests.post(batch_url, files=files, timeout=60)
        elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        print(f"  batch -> {elapsed:.1f} ms ({resp.json().get('count', '?')} images)")

    total = time.perf_counter() - t_start
    print(f"  Total: {total:.2f}s for {n} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark liveness service")
    parser.add_argument("image", help="Test image path")
    parser.add_argument("--url", default="http://localhost:8090/verify")
    parser.add_argument("-n", type=int, default=200, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--batch", action="store_true", help="Also benchmark /verify_batch")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return

    bench_verify(args.url, args.image, args.n, args.concurrency)
    if args.batch:
        bench_batch(args.url, args.image, args.n, args.batch_size)


if __name__ == "__main__":
    main()
