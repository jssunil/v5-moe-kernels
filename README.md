# v5-moe-kernels

From-scratch GPU kernels for the hot paths of the ERA V5 model, written in Triton (and one in CUDA C++), each proven correct against a PyTorch reference before any speed is claimed.

> **Status: plan.** This README lists the work, the pass/fail gate for each piece, and the references. Code lands milestone by milestone; the table below is updated as each gate is met.

## Target shapes (ERA V5 reference model)

| Part | Value |
|---|---|
| Hidden size `d` | 2,560 |
| Routed experts | 384, FFN width 512, top-6 routing |
| Shared expert | 1 |
| Attention | GQA, 48 query heads, 4 KV heads, head dim 128, QK-norm; 4 sliding-window layers to 1 full layer |
| Vocabulary | 128,000 (input embedding and LM head not tied) |
| Training run | B200 GPUs, sequence 16,384, BF16 |

## Rules every kernel follows

1. **Correct before fast.** Output and gradients match the PyTorch reference within BF16 tolerance, at V5 shapes, including edge cases (skewed expert load, empty experts, ragged last tile).
2. **Deterministic.** Two runs with the same inputs give bit-identical outputs and gradients (stable sort, fixed reduction order, no atomics in reductions).
3. **No host syncs.** No `.item()`, no Python loop over experts, no data-dependent shapes read back to the CPU.
4. **Measured against real baselines.** Eager PyTorch and `torch.compile`, reported as time saved per step, not only speedup.
5. **Profile first.** Each kernel is chosen by its share of step time (Amdahl): speeding up a part that takes 10% of the step gives at most 1.11×.

## Milestones

| ID | Work | Gate (done means all of this) | Status |
|---|---|---|---|
| M01 | Repo scaffold, PyTorch reference MoE layer at V5 shapes, test and benchmark harness | Reference layer + `pytest` parity harness + benchmark script committed | Planned |
| M07 | Profile the reference MoE block (forward + backward) | `torch.profiler` and Nsight Systems traces; time share per op; Amdahl ceiling per planned kernel; kernel order chosen from it | Planned |
| M02 | Fused RMSNorm + residual add, forward and backward (Triton) | Output and gradient parity; bit-identical reruns; benchmark vs eager and `torch.compile` | Planned |
| M03 | Grouped MoE GEMM forward: one launch for all experts | Parity at V5 shapes with skewed load and empty experts; uses `tl.dot` (tensor cores); no host sync | Planned |
| M08 | Token permute / unpermute (MoE dispatch and combine) | On-device count per expert, aligned prefix sum, permute to expert order, gate-weighted scatter back; parity; deterministic | Planned |
| M04 | Grouped MoE GEMM backward: dX and dW | Gradient parity for inputs and weights; bit-identical reruns | Planned |
| M05 | Fused linear + cross-entropy for a 128K vocabulary (stretch) | Chunked, FP32 log-sum-exp; loss and gradient parity; memory saved vs eager measured | Planned |
| M09 | Fused RMSNorm in CUDA C++ as a PyTorch extension (stretch) | Passes the same M02 tests; Triton vs CUDA vs `torch.compile` table | Planned |
| M06 | Results write-up | Results table, M07 profile and Amdahl table, limits section, lessons | Planned |

Order of work: M01 → M07 → M02 → M03 → M08, then M04, M05 and M09 as time allows.

## Planned layout

```
reference/     PyTorch reference layers (RMSNorm, MoE layer, router, LM head + CE)
kernels/       triton/ and cuda/ implementations
tests/         parity and determinism tests (pytest), one file per kernel
bench/         benchmark scripts; results/ holds JSON + CSV outputs
profile/       profiler traces and the Amdahl table
```

## How each kernel is built

**M02 fused RMSNorm + residual.** One pass reads `x` and the residual, writes the sum and the normalised output; the backward reduces the weight gradient across rows. Start from Triton tutorial 5 (layer norm), which shows the two-stage weight-gradient reduction; Liger Kernel's RMSNorm is a production reference.

**M03 grouped GEMM forward.** Triton has no native grouped GEMM, so build a tile schedule on the GPU: for each expert compute how many M-tiles its tokens need, prefix-sum them, and map each program to an (expert, tile) pair. Each program loads its expert's weights and a tile of that expert's tokens and uses `tl.dot`. ScatterMoE avoids padding and copies this way; MegaBlocks pads to blocks and uses block-sparse matmul. Compare against `torch._grouped_mm` if your PyTorch has it.

**M08 permute / unpermute.** Count tokens per expert, take an aligned prefix sum for expert offsets, write tokens in expert order, then scatter expert outputs back to token order weighted by the gate. A stable order is what makes it deterministic. Megatron-Core documents the same steps as separate Triton kernels.

**M04 grouped GEMM backward.** dX is a grouped GEMM with transposed weights; dW is a grouped GEMM over each expert's tokens and needs a fixed reduction order to stay deterministic. SonicMoE reports that this backward pass is where simpler designs lose the most time.

**M05 fused linear + cross-entropy.** Never materialise the full `tokens × 128,000` logits: process the vocabulary in chunks with an online, FP32 log-sum-exp, and compute the gradient in the same pass. Cut Cross-Entropy and Liger's fused linear cross-entropy are the references.

**M09 CUDA RMSNorm.** One warp or block per row, vectorised 128-bit loads, warp-shuffle reduction, FP32 accumulation, exposed through a PyTorch C++ extension.

## References

### Papers
- Zhang and Sennrich, *Root Mean Square Layer Normalization* — [arXiv 1910.07467](https://arxiv.org/abs/1910.07467)
- Tan et al., *Scattered Mixture-of-Experts Implementation* (ScatterMoE) — [arXiv 2403.08245](https://arxiv.org/abs/2403.08245)
- Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* — [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)
- *SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations* — [arXiv 2512.14080](https://arxiv.org/abs/2512.14080)
- *Cross-Platform Fused MoE Dispatch in Triton* — [arXiv 2605.23911](https://arxiv.org/abs/2605.23911)
- DeepSeek-AI, *DeepSeekMoE* — [arXiv 2401.06066](https://arxiv.org/abs/2401.06066); *DeepSeek-V3 Technical Report* — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training* — [arXiv 2410.10989](https://arxiv.org/abs/2410.10989)
- Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (Cut Cross-Entropy) — [arXiv 2411.09009](https://arxiv.org/abs/2411.09009)

### Code
- [Triton](https://github.com/triton-lang/triton) and its [tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) (05: layer norm forward and backward; 03: matmul)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): RMSNorm, fused linear cross-entropy
- [scattermoe](https://github.com/shawntan/scattermoe): grouped expert matmul in Triton
- [megablocks](https://github.com/databricks/megablocks): block-sparse MoE
- [ml-cross-entropy](https://github.com/apple/ml-cross-entropy): Cut Cross-Entropy kernels
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and [DeepEP](https://github.com/deepseek-ai/DeepEP): DeepSeek's grouped GEMM and expert-parallel communication
- [torchtitan](https://github.com/pytorch/torchtitan) and [torchao](https://github.com/pytorch/ao): PyTorch's MoE training stack
- [Megatron-Core MoE permute kernels](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.inference.moe.permute.html)

### Tools
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html), [Nsight Systems](https://docs.nvidia.com/nsight-systems/), [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [PyTorch C++/CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) (for M09)
