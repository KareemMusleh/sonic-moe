"""Nsight Compute driver for SonicMoE's down-projection MXFP8 dGated backward.

The tensors reproduce fp8_trace.py's T=8192, H=4096, I=1024, E=64, K=10
workload with an even 1280 routed rows per expert.  Allocation, quantization,
and compilation happen before the measured target launches.
"""

import argparse

import torch
from quack.gemm_blockscaled_sm90 import (
    mxfp8_gemm_dgated_tuned_sm90,
    quantize_act,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig

T, H, I, E, K = 8192, 4096, 1024, 64, 10
TK = T * K

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("full", "no-reduce", "no-scale"),
        default="full",
    )
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--epi-tile-n",
        type=int,
        choices=(0, 32, 64, 128),
        default=32,
        help="0 selects QuACK's default epilogue tile",
    )
    parser.add_argument("--tile-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--tile-n", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "--cluster", choices=("1x1", "2x1", "1x2"), default="2x1"
    )
    parser.add_argument("--materialize-a", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.target <= 0:
        raise ValueError("--warmup must be non-negative and --target must be positive")

    torch.manual_seed(0)
    device = torch.device("cuda")
    cluster_m, cluster_n = map(int, args.cluster.split("x"))
    config = GemmConfig(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        epi_tile_n=args.epi_tile_n or None,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        pingpong=False,
        is_dynamic_persistent=False,
    )

    # The real call gathers T source rows into TK expert-grouped rows.
    dout = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    dout_q, dout_sc = quantize_act(dout)
    rows_per_expert = TK // E
    cu_seqlens_m = torch.arange(
        0, TK + 1, rows_per_expert, dtype=torch.int32, device=device
    )
    a_idx = torch.arange(TK, dtype=torch.int32, device=device).remainder_(T)
    a_scale = torch.empty(
        H // 128, TK, dtype=torch.float32, device=device
    ).mT
    a_scale.copy_(dout_sc[a_idx.long()])
    gemm_a = dout_q[a_idx.long()].contiguous() if args.materialize_a else dout_q
    gemm_a_idx = None if args.materialize_a else a_idx

    # Match backward_fp8.py: quantize the (E,H,I) down weight transposed, then
    # expose the backward GEMM's logical (E,H,I) B view.
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device)
    w2_bwd_q, w2_bwd_sc = quantize_weight_sm90(w2, transpose=True)
    b = w2_bwd_q.mT
    b_scale = w2_bwd_sc.mT

    preact = torch.randn(TK, 2 * I, dtype=torch.bfloat16, device=device)
    colvec_scale = torch.rand(TK, dtype=torch.float32, device=device)
    dx_out = torch.empty_like(preact)
    postact_out = torch.empty(TK, I, dtype=torch.bfloat16, device=device)

    def target():
        return mxfp8_gemm_dgated_tuned_sm90(
            gemm_a,
            b,
            a_scale,
            b_scale,
            preact,
            dx_out,
            postact_out,
            colvec_scale=None if args.variant == "no-scale" else colvec_scale,
            activation="swiglu",
            colvec_reduce=args.variant == "full",
            cu_seqlens_m=cu_seqlens_m,
            A_idx=gemm_a_idx,
            config=config,
        )

    # Compile before NCU's requested target launch. benchmark.sh filters on the
    # CUTLASS kernel name, so these are skipped with its --launch-skip setting.
    target()
    torch.cuda.synchronize()
    if args.check:
        candidate_dx = dx_out.clone()
        candidate_postact = postact_out.clone()
        candidate_reduce = target()
        candidate_reduce = candidate_reduce.clone()
        reference_config = GemmConfig(
            tile_m=128,
            tile_n=256,
            epi_tile_n=128,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
        )
        reference_dx = torch.empty_like(dx_out)
        reference_postact = torch.empty_like(postact_out)
        reference_reduce = mxfp8_gemm_dgated_tuned_sm90(
            gemm_a,
            b,
            a_scale,
            b_scale,
            preact,
            reference_dx,
            reference_postact,
            colvec_scale=None if args.variant == "no-scale" else colvec_scale,
            activation="swiglu",
            colvec_reduce=args.variant == "full",
            cu_seqlens_m=cu_seqlens_m,
            A_idx=gemm_a_idx,
            config=reference_config,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(candidate_dx, reference_dx, rtol=0, atol=0)
        torch.testing.assert_close(
            candidate_postact, reference_postact, rtol=0, atol=0
        )
        torch.testing.assert_close(
            candidate_reduce, reference_reduce, rtol=1e-2, atol=2e-3
        )
    for _ in range(args.warmup):
        target()
    torch.cuda.synchronize()
    if args.measure:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.target):
            target()
        end.record()
        end.synchronize()
        print(f"{start.elapsed_time(end) / args.target:.6f}")
    else:
        for _ in range(args.target):
            target()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
