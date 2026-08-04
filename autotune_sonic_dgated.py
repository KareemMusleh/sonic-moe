"""Autotune MXFP8 dGated using arguments captured from a real SonicMoE backward."""

import torch

import sonicmoe.functional.backward_fp8 as backward_fp8
from quack.gemm_config import GemmConfig
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

    original = backward_fp8.mxfp8_gemm_dgated_sm90
    captured = {}

    def capture(*args, **kwargs):
        if not captured:
            captured["args"] = args
            captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "config"}
        return original(*args, **kwargs)

    backward_fp8.mxfp8_gemm_dgated_sm90 = capture
    try:
        output = moe(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe_fp8,
            is_inference_mode=False,
        )[0]
        output.backward(dout)
        torch.cuda.synchronize()
    finally:
        backward_fp8.mxfp8_gemm_dgated_sm90 = original

    args = captured["args"]
    kwargs = captured["kwargs"]
    candidates = [
        GemmConfig(
            tile_m=128,
            tile_n=tile_n,
            epi_tile_n=epi_tile_n,
            cluster_m=cluster_m,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
        )
        for cluster_m in (1, 2)
        for tile_n in (128, 256)
        for epi_tile_n in (None, 32, 64, 128)
    ]

    results = []
    for config in candidates:
        for _ in range(5):
            original(*args, **kwargs, config=config)
        torch.cuda.synchronize()
        samples = []
        for _ in range(25):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            original(*args, **kwargs, config=config)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        median = float(torch.tensor(samples).median())
        results.append((median, config))
        print(
            f"{median:.6f} ms  tile_m={config.tile_m} tile_n={config.tile_n} "
            f"epi_tile_n={config.epi_tile_n} cluster={config.cluster_m}x{config.cluster_n}"
        )

    median, config = min(results, key=lambda item: item[0])
    print(
        f"BEST {median:.6f} ms  tile_m={config.tile_m} tile_n={config.tile_n} "
        f"epi_tile_n={config.epi_tile_n} cluster={config.cluster_m}x{config.cluster_n}"
    )


if __name__ == "__main__":
    main()
