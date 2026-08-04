import argparse
import json
import pickle
from pathlib import Path

import torch


def add_benchmark_arguments(parser: argparse.ArgumentParser, default_trace_dir: str) -> None:
    parser.add_argument("--mode", choices=("fwd", "fwd-bwd"), default="fwd-bwd")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-iterations", type=int, default=1)
    parser.add_argument("--trace-dir", type=Path, default=Path(default_trace_dir))


def validate_benchmark_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in (
        "tokens",
        "hidden_size",
        "intermediate_size",
        "experts",
        "top_k",
        "iterations",
        "trace_iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.top_k > args.experts:
        parser.error("--top-k must not exceed --experts")


def check_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise RuntimeError(f"{name} shape {tuple(tensor.shape)} != {shape}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")


def median_cuda_ms(fn, iterations: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(torch.tensor(samples).median()), samples


def peak_cuda_memory_bytes(fn) -> int:
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def profile_cuda(
    fn,
    trace_path: Path,
    iterations: int,
    *,
    row_limit: int = 30,
    cold_l2: bool = True,
    l2_flush_multiple: int = 3,
) -> tuple[torch.profiler.profile, int, Path]:
    """Profile a callable and capture its peak incremental CUDA memory.

    When ``cold_l2`` is enabled, a same-stream write through a buffer larger
    than L2 is issued immediately before every invocation. CUDA profiler
    collection is disabled around that write so its kernel does not appear in
    GPU totals/traces.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    memory_snapshot = trace_path.with_name(
        f"{trace_path.stem.removesuffix('_trace')}_memory.pickle"
    )
    flush = None
    if cold_l2:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2_bytes = properties.L2_cache_size
        flush = torch.empty(
            l2_flush_multiple * l2_bytes + 1,
            device="cuda",
            dtype=torch.uint8,
        )
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        for _ in range(iterations):
            if flush is not None:
                activities = [torch.profiler.ProfilerActivity.CUDA]
                profile.toggle_collection_dynamic(False, activities)
                flush.zero_()
                profile.toggle_collection_dynamic(True, activities)
            fn()
        torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated() - baseline
    profile.export_chrome_trace(str(trace_path))
    with memory_snapshot.open("wb") as f:
        pickle.dump(torch.cuda.memory._snapshot(), f)
    return profile, peak_memory_bytes, memory_snapshot


def print_profile_report(
    title: str,
    profile: torch.profiler.profile,
    trace_path: Path,
    memory_snapshot: Path,
    peak_memory_bytes: int,
    *,
    row_limit: int = 30,
) -> None:
    """Print the common human-readable trace report used by both backends."""
    print(f"\n{title}")
    print(profile.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
    print(f"chrome trace: {trace_path}")
    print(f"memory snapshot: {memory_snapshot}")
    print(
        f"peak incremental CUDA memory: {peak_memory_bytes / 2**30:.3f} GiB "
        f"({peak_memory_bytes} bytes)"
    )


def export_profile(fn, trace_dir: Path, mode: str, iterations: int) -> tuple[Path, Path]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    chrome_trace = trace_dir / f"{mode}-trace.json"
    memory_snapshot = trace_dir / f"{mode}-memory.pickle"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        for _ in range(iterations):
            fn()
            profile.step()
    torch.cuda.synchronize()
    profile.export_chrome_trace(str(chrome_trace))
    with memory_snapshot.open("wb") as f:
        pickle.dump(torch.cuda.memory._snapshot(), f)
    return chrome_trace, memory_snapshot


def write_metadata(
    path: Path,
    implementation: str,
    args: argparse.Namespace,
    score_ms: float,
    samples_ms: list[float],
    peak_memory_bytes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "implementation": implementation,
        "mode": args.mode,
        "shape": {
            "tokens": args.tokens,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "experts": args.experts,
            "top_k": args.top_k,
        },
        "score_ms": score_ms,
        "samples_ms": samples_ms,
        "peak_memory_bytes": peak_memory_bytes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
