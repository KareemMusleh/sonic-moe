# YDT Core: Complete SonicMoE SM90 MXFP8 Backend

## Goal

Implement a complete, selectable SonicMoE backend in:

```text
/home/kareemm/arcadia/ml/torch/ydt/core
```

The work is specifically for the Hopper/SM90 MXFP8 training path. It is not a
request for a generic BF16 SonicMoE backend.

Selecting:

```python
Float8Config(moe_impl="sonic-moe-sm90")
```

should make SonicMoE own both routing and expert execution:

```text
router linear logits
    -> SonicMoE/QuACK top-k + softmax over the selected experts
    -> SonicMoE routing metadata and statistics
    -> SM90 MXFP8 grouped up projection + fused SwiGLU
    -> SM90 MXFP8 grouped down projection
    -> SonicMoE weighted aggregation
```

The user should not also have to set `router_impl="sonic-moe"`.

Keep the existing generic BF16 and DeepGEMM FP8 paths unchanged.

## Workspaces and Source Locations

### YDT Core

The target package is:

```text
/home/kareemm/arcadia/ml/torch/ydt/core
```

Important YDT files:

```text
ydt_core/fp8/__init__.py
ydt_core/fp8/tensor.py
ydt_core/fp8/fsdp_utils.py
ydt_core/optimizations.py

ydt_core/nn/modules/moe/__init__.py
ydt_core/nn/modules/moe/router.py
ydt_core/nn/modules/moe/experts.py
ydt_core/nn/modules/moe/permute.py

ydt_core/experimental/nn/modules/moe/__init__.py
ydt_core/experimental/nn/modules/moe/router.py
ydt_core/experimental/nn/modules/moe/sonic_moe.py

tests/nn/modules/moe/test_mlp.py
tests/nn/modules/moe/test_fp8_all_gather.py
tests/nn/modules/moe/test_fp8_checkpoint.py
tests/test_optimizations.py
```

### SonicMoE Repository

The reference implementation is:

```text
/home/kareemm/src/sonic-moe
```

The main SonicMoE sources are under:

```text
sonicmoe_bench/sonicmoe/moe.py
sonicmoe_bench/sonicmoe/functional/forward_fp8.py
sonicmoe_bench/sonicmoe/functional/backward_fp8.py
sonicmoe_bench/sonicmoe/functional/fp8_tensor.py
sonicmoe_bench/sonicmoe/functional/topk.py
sonicmoe_bench/sonicmoe/functional/triton_kernels/
sonicmoe_bench/tests/moe_fp8_test.py
```

### QuACK Location

In this SonicMoE checkout, QuACK is located at:

```text
/home/kareemm/src/sonic-moe/third_party/quack
```

Do not use `quack_bench` as the authoritative dependency for this work. The
relevant bundled QuACK implementation is:

```text
third_party/quack/quack/gemm_blockscaled_sm90.py
third_party/quack/quack/gemm_sm90.py
third_party/quack/quack/gemm_dact.py
third_party/quack/quack/gemm_config.py
third_party/quack/quack/quant.py
third_party/quack/quack/topk.py

third_party/quack/tests/test_gemm_sm90_mxfp8.py
third_party/quack/tests/test_topk.py
```

YDT imports QuACK as the Python package `quack`, for example:

```python
from quack.gemm_blockscaled_sm90 import mxfp8_gemm_act_sm90
from quack.topk import topk
```

## How the SonicMoE SM90 MXFP8 Path Works

SonicMoE owns MoE routing metadata and grouped expert computation. QuACK owns
the Hopper block-scaled grouped GEMMs, scale layouts, scheduling, and fused
epilogues.

### Quantization

The format uses `torch.float8_e4m3fn` payloads and FP32 dequantization scales.

- Activations use `1 x 128` blocks: one scale per row and per 128 values on K.
- Weights use `128 x 128` blocks: one scale per 128 rows and 128 K values.
- Hidden and intermediate dimensions must therefore be compatible with the
  128-element block geometry.

On SM90 this is software-applied block scaling. Hopper does not use Blackwell's
native block-scaled MMA. QuACK:

1. Executes FP8 WGMMA for one K block into a temporary FP32 accumulator.
2. Loads the activation and weight scales for that block.
3. Multiplies the scales together.
4. Promotes the partial accumulator into the full accumulator:

```text
acc += fp8_wgmma_partial * activation_scale * weight_scale
```

The relevant implementation is `GemmSm90.mma_blockscaled()` in:

```text
third_party/quack/quack/gemm_sm90.py
```

The PyTorch-facing SM90 API and weight quantizer are in:

```text
third_party/quack/quack/gemm_blockscaled_sm90.py
```

### Forward

The reference SonicMoE FP8 forward is:

```text
x (BF16)
  -> router logits
  -> top-k + selected-expert softmax
  -> expert-major routing metadata
  -> quantize x to 1x128 FP8
  -> grouped up GEMM against 128x128 FP8 weights
  -> fused SwiGLU/GeGLU epilogue
  -> BF16 postactivation
  -> quantize postactivation to 1x128 FP8
  -> grouped down GEMM
  -> routing-score-weighted gather and sum
  -> output
```

The up projection may store the BF16 preactivation for backward. In inference
mode it can skip that store.

The reference forward orchestration is in:

```text
sonicmoe_bench/sonicmoe/functional/forward_fp8.py
sonicmoe_bench/sonicmoe/functional/backward_fp8.py
```

### Backward

The backward path intentionally uses split precision:

- Activation gradients use SM90 MXFP8 grouped GEMMs.
- Weight gradients use BF16 grouped GEMMs.

For the down projection, QuACK fuses:

```text
raw_da = dout @ w2
da = routing_score * raw_da
dh = swiglu_backward(preact, da)
a_prime = routing_score * swiglu(preact)
ds = dot(raw_da, swiglu(preact))
```

`a_prime` is reused by the BF16 weight-gradient GEMM. `ds` supplies the gradient
for the selected routing scores.

Forward saves quantized weights. Backward transposes the saved FP8 payload and
scale matrix rather than requantizing the BF16 weight. This is valid because
weight blocks are square `128 x 128`, so their scale is invariant under
transpose.

### Routing

The desired routing convention is softmax over the selected top-k logits:

```python
selected_logits, selected_experts = topk(logits, k)
selected_scores = softmax(selected_logits, dim=-1)
```

QuACK supplies a fused differentiable implementation:

```python
from quack.topk import topk

selected_scores, selected_experts = topk(logits.float(), k, softmax=True)
```

Its forward and backward are implemented in:

```text
third_party/quack/quack/topk.py
```

This differs from computing a full softmax over all experts and then selecting
top-k. Preserve the selected-expert-softmax semantics used by the existing
experimental Sonic router.

Routing metadata must provide:

- Expert-major row ordering.
- A source-token gather index for every retained route.
- An inverse mapping from `(token, top-k slot)` to the expert-major result.
- Per-expert counts and cumulative sequence offsets.
- Correct handling of dropped routes represented by the sentinel expert ID
  `num_experts`.

The existing `_routing_metadata()` in YDT's experimental
`sonic_moe.py` already constructs these objects using a stable sort.

## Current YDT Implementation

YDT already has a partial experimental integration.

### Configuration

`ydt_core/fp8/__init__.py` currently defines:

```python
@attrs.define(slots=False)
class Float8Config:
    quant_all_gather: bool = True
    moe_impl: Literal["deep-gemm", "sonic-moe-sm90"] = "deep-gemm"
```

### Expert Backend

`MoEMLP.apply_float8()` in `ydt_core/nn/modules/moe/experts.py` dispatches:

```python
if config.moe_impl == "sonic-moe-sm90":
    apply_sonic_moe_float8(self, config)
```

`apply_sonic_moe_float8()` currently:

- Requires SiLU/SwiGLU.
- Replaces separate `up_proj` and `gate_proj` parameters with an interleaved
  `gate_up_proj`.
- Transposes `down_proj` into the layout consumed by QuACK.
- Optionally wraps parameters in
  `WeightWithDynamicFloat8CastTensor` for quantized YaFSDP all-gather.
- Installs `SonicMoEFP8ForwardImpl`.

The experimental forward implementation already has:

- SM90 MXFP8 grouped up projection.
- Fused SwiGLU.
- SM90 MXFP8 grouped down projection.
- Fused dGated backward.
- BF16 weight gradients.
- Selected-score gradients.
- Dropped-route handling through an inverse-offset sentinel.
- Canonical/HuggingFace checkpoint layout conversion.
- FP8 all-gather support.

### Current Routing Problem

The parent `MoE.forward()` still calls the generic `TopKRouter` first:

```text
generic router
  -> expert_scores
  -> top_k_expert_scores
  -> top_k_expert_indices
  -> num_tokens_per_expert
  -> token dispatcher
  -> Sonic expert implementation
```

There is also an independent optimization option:

```python
OptimizationsConfig(router_impl="sonic-moe")
```

This only swaps `TopKRouter.token_choice_impl` with
`SonicMoETopKChoiceImpl`. Consequently:

- Users must select two options to get the intended complete backend.
- Sonic does not own the routing boundary.
- Routing statistics live under the generic router.
- Generic routing still materializes and processes data that the Sonic backend
  should control.
- The routing and expert choices can become inconsistent if only one option is
  selected.

## Desired Architecture

Introduce a MoE-level implementation boundary, not just an expert-level
implementation boundary.

A reasonable shape is:

```python
class MoEForwardImplBase(Protocol):
    def __call__(
        self,
        module: MoE,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
```

The generic implementation should contain the current `MoE.forward()` logic.
The Sonic implementation should live under:

```text
ydt_core/experimental/nn/modules/moe/
```

and own:

1. Router linear projection.
2. QuACK fused top-k plus selected-expert softmax.
3. Padding/token-drop masking.
4. Per-expert counts.
5. Sonic routing metadata.
6. Sonic statistics.
7. MXFP8 expert execution.
8. Weighted aggregation.

`MoE.apply_float8(Float8Config(moe_impl="sonic-moe-sm90"))` should install both:

- The Sonic MoE-level forward implementation.
- The Sonic expert parameter layout and MXFP8 kernels.

If introducing a full `MoEForwardImplBase` is too invasive, another acceptable
design is a Sonic router module/strategy installed automatically by
`MoE.apply_float8`. The important invariants are:

- `moe_impl="sonic-moe-sm90"` alone selects Sonic top-k and selected softmax.
- The Sonic experimental code owns its statistics.
- The generic and DeepGEMM paths do not change behavior.
- There is one coherent backend selection, not two independent toggles.

Avoid a circular import between `ydt_core.nn.modules.moe` and
`ydt_core.experimental.nn.modules.moe`. Use a local import inside
`apply_float8()` if necessary, matching the current expert integration.

## Statistics

The Sonic backend should save its own activation/routing statistics through
YDT's `ActivationStatsMixin.save_activation_stat()`.

At minimum record:

```text
router/input                 ABSMAX
router/logits                ABSMAX
router/topk_scores           ABSMAX
router/tokens_per_expert     appropriate MIN/MAX/AVG summaries
router/load_imbalance        MAX or AVG scalar
router/dropped_routes        SUM or AVG scalar
experts/down                 ABSMAX
experts/unpermute_out        ABSMAX
output                       ABSMAX
```

Because `save_activation_stat()` reduces its input immediately according to
`ReduceOp`, use distinct names for minimum, maximum, and mean counts if all
three are required. For example:

```python
save_activation_stat("tokens_per_expert_min", counts, ReduceOp.MIN)
save_activation_stat("tokens_per_expert_max", counts, ReduceOp.MAX)
save_activation_stat("tokens_per_expert_mean", counts, ReduceOp.AVG)
```

Define load imbalance robustly for an empty batch:

```text
max(tokens_per_expert) / max(mean(tokens_per_expert), 1)
```

Do not force synchronization through `.item()` in the forward path.

The existing generic router records only input and logits absmax. Sonic's
statistics should not be duplicated under both the generic router and Sonic
implementation.

## Compatibility Requirements

### Load-Balancing Loss

`MoE.forward()` returns:

```text
output, expert_scores, num_tokens_per_expert_masked
```

`expert_scores` is consumed by `LoadBalancingLoss`, which currently expects
full per-expert float32 probabilities.

The QuACK selected-expert-softmax kernel only produces selected scores. Decide
explicitly how to preserve the load-balancing-loss contract:

1. If the configured loss requires full expert probabilities, compute the
   full float32 softmax from router logits solely for that loss while using
   QuACK top-k/selected softmax for dispatch; or
2. Extend the loss contract to consume logits or sparse selected scores without
   changing its mathematical meaning.

Prefer the first option initially because it is localized and preserves
existing training behavior. Do not accidentally use full-softmax-selected
scores for expert weighting; dispatch must continue to use softmax over the
selected top-k logits.

### Bias Correction and Alternative Gating

The current generic router supports:

- `gating="softmax"` or `"sigmoid"`.
- DeepSeek-style selection bias correction.
- Optimizer-step token-count updates.

The existing QuACK Sonic top-k path supports only:

- Softmax gating.
- No expert selection bias.

For the first implementation, explicitly reject incompatible configurations
when `moe_impl="sonic-moe-sm90"`:

```text
router.gating != "softmax"
router.bias_correction is True
expert router bias if unsupported by the selected kernel path
```

Do not silently fall back to generic routing under a configuration explicitly
requesting the Sonic backend.

### Padding and Token Dropping

Preserve these conventions:

- Padding routes must not contribute to expert counts or aggregation.
- Dropped routes use expert ID `num_experts` as a sentinel.
- Dropped selected scores are zero.
- Gradients for dropped scores are zero.
- Per-expert counts exclude the sentinel.
- The inverse mapping may point dropped routes to a single appended zero row.

### Expert Parallelism

The current `MoE.forward()` owns `token_dispatcher.begin_dispatch()` and
`finish_dispatch()`. The experimental Sonic expert implementation currently
assumes local expert-major routing.

Do not claim general EP support without testing it. Either:

- Integrate Sonic routing with the existing dispatcher contract; or
- Validate and reject unsupported EP modes for the initial SM90 backend.

Preserve single-rank and non-EP behavior first. Existing all-gather/FSDP weight
support is separate from token expert parallelism.

### Shared Experts

Preserve the existing shared-expert branch:

```text
output = routed_expert_output + shared_expert(x)
```

It can run alongside the Sonic routed-expert computation.

### Architecture Validation

Fail early with useful messages when:

- CUDA capability is not SM90/Hopper.
- Activation is not SiLU/SwiGLU.
- Required hidden/intermediate dimensions are not divisible by 128.
- Bias or another unsupported feature is enabled.

## Tests

Add focused tests rather than relying only on low-level QuACK coverage.

### Configuration

- `Float8Config(moe_impl="sonic-moe-sm90")` automatically selects Sonic
  routing and Sonic experts.
- No separate `router_impl` is required.
- Generic and DeepGEMM defaults remain unchanged.
- Unsupported sigmoid/bias-correction configurations raise clear errors.
- Non-SM90 devices are rejected or tests are skipped appropriately.

### Routing Equivalence

Compare QuACK routing against:

```python
selected_logits, selected_indices = torch.topk(logits.float(), top_k, dim=-1)
selected_scores = torch.softmax(selected_logits, dim=-1)
```

Check:

- Selected indices.
- Selected scores.
- Backward gradient with respect to logits.
- Per-expert counts.
- Padding and dropped-route behavior.

Avoid exact index assertions for deliberately tied logits unless a stable
tie-breaking rule is specified.

### End-to-End Forward

Compare the complete Sonic backend with a dequantized reference using the same:

- `1 x 128` activation quantization.
- `128 x 128` weight quantization.
- Intermediate activation requantization.
- Selected-expert-softmax routing.

Comparing only against an unquantized BF16 reference conflates kernel/layout
correctness with expected FP8 rounding error.

### Backward

Validate:

- Input gradient.
- Interleaved gate/up weight gradient.
- Down weight gradient.
- Router-logit/selected-score gradient.
- Dropped-route gradients.

### Statistics

Enable activation statistics and assert:

- Sonic-specific routing tags exist.
- Counts and load summaries match a PyTorch reference.
- Generic router statistics are not duplicated for the Sonic path.
- No forward-path host synchronization is introduced.

### Checkpointing and FSDP

Retain the existing coverage for:

- Canonical and HuggingFace state-dict conversion.
- Interleaved `gate_up_proj`.
- Quantized and non-quantized all-gather.
- Saved/restored FP8 parameter layouts.

## Likely Files to Modify

The exact design may vary, but expect changes in:

```text
ydt_core/fp8/__init__.py
ydt_core/optimizations.py
ydt_core/nn/modules/moe/__init__.py
ydt_core/nn/modules/moe/experts.py
ydt_core/experimental/nn/modules/moe/__init__.py
ydt_core/experimental/nn/modules/moe/router.py
ydt_core/experimental/nn/modules/moe/sonic_moe.py
tests/nn/modules/moe/test_mlp.py
tests/test_optimizations.py
```

Do not modify pipeline-parallel code for this task unless a directly affected
test exposes a genuine compatibility issue. Pipeline parallelism is not the
focus.

## Acceptance Criteria

The work is complete when:

1. `Float8Config(moe_impl="sonic-moe-sm90")` selects the full Sonic routing and
   expert path.
2. QuACK performs differentiable top-k plus selected-expert softmax.
3. Sonic builds and owns its routing metadata.
4. Sonic records its own routing and MXFP8 execution statistics.
5. Load-balancing-loss behavior is preserved deliberately.
6. Unsupported router and architecture configurations fail explicitly.
7. Forward, backward, token dropping, statistics, checkpoint conversion, and
   relevant FP8 all-gather tests pass.
8. The generic BF16 and DeepGEMM FP8 paths remain behaviorally unchanged.

