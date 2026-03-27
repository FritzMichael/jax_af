#!/usr/bin/env bash
set -euo pipefail

NSYS="$(command -v nsys)"
PYTHON3="$(command -v python3)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

"$NSYS" profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --output benchmark_artifacts/warp_bench --force-overwrite true "$PYTHON3" benchmarks/warp_bench.py
"$NSYS" profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --output benchmark_artifacts/warp_bench_ad --force-overwrite true "$PYTHON3" benchmarks/warp_bench_ad.py
"$NSYS" profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --output benchmark_artifacts/jax_vjp_bench --force-overwrite true "$PYTHON3" benchmarks/jax_vjp_bench.py
"$NSYS" profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --output benchmark_artifacts/jax_grad_bench --force-overwrite true "$PYTHON3" benchmarks/jax_grad_bench.py