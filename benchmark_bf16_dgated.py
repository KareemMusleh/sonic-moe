"""Benchmark BF16 dGated with arguments captured from a real SonicMoE backward."""

import torch

import quack.gemm_blockscaled_sm90 as blockscaled_sm90


def _unused_fp8_postact_quant(*args, **kwargs):
    raise RuntimeError("FP8 postact quantization is unavailable in the isolated QuACK checkout")


if not hasattr(blockscaled_sm90, "mxfp8_gemm_gated_postact_quant_sm90"):
    blockscaled_sm90.mxfp8_gemm_gated_postact_quant_sm90 = _unused_fp8_postact_quant

import sonicmoe.functional.backward as backward
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType

T, H, I, E, K = 8192, 4096, 1024, 64, 10


def main() -> None:
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
    x = (0.2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    dout = torch.randn_like(x)

    original = backward.gemm_dgated
    captured = {}

    def capture(*args, **kwargs):
        if not captured:
            captured["args"] = args
            captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    backward.gemm_dgated = capture
    try:
        output = moe(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe,
            is_inference_mode=False,
        )[0]
        output.backward(dout)
        torch.cuda.synchronize()
    finally:
        backward.gemm_dgated = original

    args = captured["args"]
    kwargs = captured["kwargs"]
    for _ in range(10):
        original(*args, **kwargs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        original(*args, **kwargs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    samples_tensor = torch.tensor(samples)
    print(f"median_ms={samples_tensor.median().item():.6f}")
    print(f"min_ms={samples_tensor.min().item():.6f}")
    print(f"max_ms={samples_tensor.max().item():.6f}")


if __name__ == "__main__":
    main()
