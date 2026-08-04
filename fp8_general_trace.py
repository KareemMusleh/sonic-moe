"""Trace SonicMoE's SM90 MXFP8 general-routing path."""

import argparse
import math
from pathlib import Path

import torch

from sonicmoe import MoE, moe_general_routing_inputs_fp8
from sonicmoe.enums import ActivationType
from trace_benchmark_utils import check_tensor, print_profile_report, profile_cuda


def capacity_filtered_topk(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    capacity_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a token-sorted routing graph, keeping at most ``capacity`` entries per expert."""
    with torch.no_grad():
        logits = torch.nn.functional.linear(x, router_weight)
        scores, experts = logits.float().softmax(dim=-1).topk(top_k, dim=-1)

    tokens_cpu = (
        torch.arange(x.shape[0], dtype=torch.int32)[:, None]
        .expand(x.shape[0], top_k)
        .reshape(-1)
    )
    scores_cpu = scores.detach().cpu().reshape(-1)
    experts_cpu = experts.detach().cpu().to(torch.int32).reshape(-1)
    capacity = math.ceil(capacity_factor * x.shape[0] * top_k / router_weight.shape[0])
    used = torch.zeros(router_weight.shape[0], dtype=torch.int32)
    keep = torch.zeros(experts_cpu.numel(), dtype=torch.bool)
    for assignment, expert in enumerate(experts_cpu.tolist()):
        if used[expert] < capacity:
            keep[assignment] = True
            used[expert] += 1

    return (
        scores_cpu[keep].cuda().requires_grad_(),
        tokens_cpu[keep].cuda(),
        experts_cpu[keep].cuda(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.capacity_factor <= 0:
        parser.error("--capacity-factor must be positive")

    torch.manual_seed(0)
    moe = (
        MoE(
            num_experts=args.experts,
            num_experts_per_tok=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        )
        .to(torch.bfloat16)
        .cuda()
    )
    x = (
        0.2
        * torch.randn(
            args.tokens,
            args.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    dout = torch.randn_like(x)
    scores, tokens, experts = capacity_filtered_topk(
        x, moe.router.weight, args.top_k, args.capacity_factor
    )
    assignments = scores.numel()
    dropped = args.tokens * args.top_k - assignments
    zero_route_tokens = args.tokens - torch.unique(tokens).numel()
    print(
        f"assignments={assignments}, dropped={dropped}, "
        f"zero_route_tokens={zero_route_tokens}, capacity_factor={args.capacity_factor}"
    )

    def fwd():
        with torch.no_grad():
            return moe_general_routing_inputs_fp8(
                x,
                scores,
                tokens,
                experts,
                moe.c_fc.weight,
                moe.c_proj.weight,
                args.experts,
                ActivationType.SWIGLU,
                is_inference_mode=True,
            )[0]

    def fwd_bwd():
        output = moe_general_routing_inputs_fp8(
            x,
            scores,
            tokens,
            experts,
            moe.c_fc.weight,
            moe.c_proj.weight,
            args.experts,
            ActivationType.SWIGLU,
        )[0]
        output.backward(dout, retain_graph=True)
        x.grad = scores.grad = moe.c_fc.weight.grad = moe.c_proj.weight.grad = None
        return output

    for _ in range(args.warmup):
        fwd()
        fwd_bwd()
    torch.cuda.synchronize()
    check_tensor("output", fwd(), (args.tokens, args.hidden_size))

    tag = f"cf{args.capacity_factor:g}".replace(".", "p")
    for title, target, mode in (
        ("General routing forward", fwd, "fwd"),
        ("General routing forward + backward", fwd_bwd, "fwd_bwd"),
    ):
        trace = Path(f"scratchpad/fp8_general_{tag}_{mode}_trace.json")
        profile, peak_memory, memory_snapshot = profile_cuda(
            target,
            trace,
            iterations=args.iterations,
            cold_l2=True,
        )
        print_profile_report(title, profile, trace, memory_snapshot, peak_memory)


if __name__ == "__main__":
    main()
