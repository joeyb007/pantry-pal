"""
Latency benchmark for the PantryPal /generate endpoint.

Usage (run after `uvicorn main:app`):
    python scripts/benchmark_latency.py --n 20 --host http://localhost:8000
"""
import argparse
import statistics
import time

import requests

SAMPLE_REQUESTS = [
    {"restrictions": ["gluten-free", "dairy-free"], "pantry_items": ["chicken", "tomatoes", "olive oil"]},
    {"restrictions": ["nut-free", "egg-free"], "pantry_items": ["salmon", "lemon", "garlic", "rice"]},
    {"restrictions": ["vegan", "soy-free"], "pantry_items": ["lentils", "spinach", "coconut milk"]},
    {"restrictions": ["pork-free", "shellfish-free"], "pantry_items": ["beef", "potatoes", "onions"]},
]


def run(n: int, host: str) -> None:
    url = f"{host}/generate"
    latencies: list[float] = []

    print(f"Firing {n} requests at {url} …\n")
    for i in range(n):
        payload = SAMPLE_REQUESTS[i % len(SAMPLE_REQUESTS)]
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=120)
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies.append(elapsed_ms)
        server_ms = resp.headers.get("X-Response-Time-Ms", "—")
        print(f"  [{i+1:>3}/{n}] {elapsed_ms:7.0f} ms  (server: {server_ms} ms)")

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]

    print(f"\n── Results ({n} requests) ──────────────────")
    print(f"  p50  : {p50:,.0f} ms")
    print(f"  p95  : {p95:,.0f} ms")
    print(f"  p99  : {p99:,.0f} ms")
    print(f"  min  : {min(latencies):,.0f} ms")
    print(f"  max  : {max(latencies):,.0f} ms")
    print(f"  mean : {statistics.mean(latencies):,.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of requests")
    parser.add_argument("--host", default="http://localhost:8000", help="Base URL")
    args = parser.parse_args()
    run(args.n, args.host)
