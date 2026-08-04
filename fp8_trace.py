"""Profile SonicMoE's FP8 forward and forward/backward paths."""

import argparse
import torch
from pathlib import Path

from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType
from trace_benchmark_utils import (
    check_tensor,
    print_profile_report,
    profile_cuda,
)

T, H, I, E, K = 8192, 4096, 1024, 64, 10
WARMUP = 20
PROFILE_ITERATIONS = 10

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--up-proj-order",
        choices=("sfa-first", "weight-first", "both"),
        default="sfa-first",
        help="Select the launch order of up-projection SFA gather and weight quantization",
    )
    parser.add_argument(
        "--trace-repeats",
        type=int,
        default=1,
        help="Number of independent profiler captures per selected launch order",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("fwd", "both"),
        default="both",
        help="Capture only forward or both forward and forward/backward",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Number of warmup iterations before profiling",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        help="Optional process-run index included in trace filenames",
    )
    args = parser.parse_args()
    if args.trace_repeats < 1:
        parser.error("--trace-repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import sonicmoe.functional.backward_fp8 as backward_fp8

    backward_fp8._UP_PROJ_SFA_FIRST = args.up_proj_order != "weight-first"
    print("up-projection postact quant: separate GG + quant")
    print(f"up-projection launch order: {args.up_proj_order}")

    torch.manual_seed(0)
    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
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
            T,
            H,
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    dout = torch.randn_like(x)

    def fwd():
        with torch.no_grad():
            return moe(
                x,
                kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8,
                is_inference_mode=True,
            )[0]

    def fwd_bwd(clear_grads: bool = True):
        output = moe(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8,
            is_inference_mode=False,
        )[0]
        output.backward(dout, retain_graph=True)
        if clear_grads:
            x.grad = w1.grad = w2.grad = router_w.grad = None
        return output

    for _ in range(args.warmup):
        fwd()
        if args.profile_mode == "both":
            fwd_bwd()
    torch.cuda.synchronize()

    output = fwd_bwd(clear_grads=False) if args.profile_mode == "both" else fwd()
    check_tensor("output", output, (T, H))
    if args.profile_mode == "both":
        check_tensor("input gradient", x.grad, (T, H))
    x.grad = w1.grad = w2.grad = router_w.grad = None

    quant_variant = "separate_quant"
    orders = ("sfa-first", "weight-first") if args.up_proj_order == "both" else (args.up_proj_order,)
    targets = [("Forward", "fwd", fwd)]
    if args.profile_mode == "both":
        targets.append(("Forward + backward", "fwd_bwd", fwd_bwd))
    for repeat in range(args.trace_repeats):
        for order in orders:
            backward_fp8._UP_PROJ_SFA_FIRST = order == "sfa-first"
            for _ in range(5):
                fwd()
            torch.cuda.synchronize()
            variant = f"{quant_variant}_{order.replace('-', '_')}"
            run_number = args.run_index if args.run_index is not None else repeat + 1
            suffix = (
                f"_run{run_number}"
                if args.trace_repeats > 1 or args.run_index is not None
                else ""
            )
            for title, target_name, target in targets:
                trace = Path(f"scratchpad/fp8_{variant}_{target_name}{suffix}_trace.json")
                prof, peak_memory, memory_snapshot = profile_cuda(
                    target,
                    trace,
                    iterations=PROFILE_ITERATIONS,
                    cold_l2=True,
                )
                print_profile_report(
                    f"{title} ({order}, run {repeat + 1})",
                    prof,
                    trace,
                    memory_snapshot,
                    peak_memory,
                )


if __name__ == "__main__":
    main()
