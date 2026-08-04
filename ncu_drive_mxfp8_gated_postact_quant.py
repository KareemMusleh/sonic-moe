"""NCU driver for SonicMoE's fused MXFP8 gated-GG + postact quantization."""

import argparse
import math

import torch
from quack.gemm_blockscaled_sm90 import (
    mxfp8_gemm_gated_postact_quant_sm90,
    quantize_act,
    quantize_weight_sm90,
)
from quack.gemm_config import GemmConfig


T, H, I, E, TOP_K = 8192, 4096, 1024, 64, 10
TK = T * TOP_K


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--cluster-m", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    rows_per_expert = TK // E
    cu = torch.arange(0, TK + 1, rows_per_expert, dtype=torch.int32, device=device)
    a_idx = torch.arange(TK, dtype=torch.int32, device=device).remainder_(T)

    source = torch.randn(T, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    weight = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    qa, dense_sfa = quantize_act(source)
    qb, sfb = quantize_weight_sm90(weight)
    sfa = torch.empty(H // 128, TK, dtype=torch.float32, device=device).mT
    sfa.copy_(dense_sfa[a_idx.long()])

    preact = torch.empty(TK, 2 * I, dtype=torch.bfloat16, device=device)
    postact = torch.empty(TK, I, dtype=torch.float8_e4m3fn, device=device)
    postact_scale = torch.empty(I // 128, TK, dtype=torch.float32, device=device).mT
    config = GemmConfig(
        tile_m=128,
        tile_n=256,
        epi_tile_n=256,
        cluster_m=args.cluster_m,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    )

    props = torch.cuda.get_device_properties(device)
    flush = torch.empty(3 * props.L2_cache_size + 1, dtype=torch.uint8, device=device)

    def target() -> None:
        mxfp8_gemm_gated_postact_quant_sm90(
            qa,
            qb.mT,
            sfa,
            sfb.mT,
            preact,
            postact,
            postact_scale,
            cu,
            A_idx=a_idx,
            config=config,
        )

    for _ in range(args.warmup):
        target()
    torch.cuda.synchronize()
    for _ in range(args.target):
        flush.zero_()
        target()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
