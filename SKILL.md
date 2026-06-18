---
name: cuda2mps
description: Port CUDA-oriented PyTorch, Transformers, Diffusers, or custom torch model code to run on Apple Silicon Mac MPS. Use when code contains CUDA-specific device handling, torch.cuda APIs, flash-attn/Triton/bitsandbytes assumptions, CUDA AMP/autocast, multi-GPU device maps, dtype mismatches, unsupported MPS operators, or when debugging MacBook MPS runtime errors while preserving the upstream model's intended computation as much as possible.
---

# CUDA2MPS

Use this skill to adapt CUDA-first model code to Apple Silicon MPS without casually rewriting model math. Prefer making the environment and device plumbing fit the project before patching upstream model internals.

## Workflow

1. **Read the project requirements first.** Match package versions before changing source code. If a model repo pins `torch`, `transformers`, `accelerate`, or custom kernels, start there.
2. **Inventory CUDA assumptions.** Search for `cuda`, `torch.cuda`, `device_map`, `flash_attn`, `triton`, `bitsandbytes`, `xformers`, `autocast`, `.half()`, `.bfloat16()`, and `.to(`.
3. **Replace device selection conservatively.**
   ```python
   def choose_device(requested="auto"):
       if requested == "mps":
           if not torch.backends.mps.is_available():
               raise RuntimeError("MPS requested but unavailable")
           return torch.device("mps")
       if requested == "cuda" and torch.cuda.is_available():
           return torch.device("cuda")
       if requested == "auto":
           if torch.cuda.is_available():
               return torch.device("cuda")
           if torch.backends.mps.is_available():
               return torch.device("mps")
       return torch.device("cpu")
   ```
4. **Avoid CUDA API calls on MPS.** Replace `torch.cuda.device_count()`, `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, CUDA seed calls, and CUDA memory calls with device-gated helpers. Use `torch.mps.synchronize()` and `torch.mps.empty_cache()` only when the device is MPS.
5. **Treat dtype as a first-class porting decision.** Do not blindly convert CUDA bf16/fp16 code to MPS. Test `float32`, `float16`, and `bfloat16` on the real model path. Keep model checkpoints in their intended dtype when possible, but align temporary tensors and index assignment destinations with source dtypes.
6. **Handle autocast deliberately.** Use `torch.amp.autocast(device.type, ...)` only around regions that benefit and actually work on MPS. If a submodule fails under MPS autocast, isolate that submodule on MPS with autocast disabled before considering CPU fallback.
7. **Replace CUDA-only kernels with functionally equivalent PyTorch paths.** For inference, prefer `torch.nn.functional.scaled_dot_product_attention` over `flash_attn` on MPS. Keep the CUDA fast path intact when possible.
8. **Use CPU fallback as a last resort.** `PYTORCH_ENABLE_MPS_FALLBACK=1` can unblock unsupported ops but may hide performance problems and silently move work to CPU. If the user asks to keep work on MPS, do not enable it without saying so.
9. **Verify incrementally.** First import, then instantiate config/model skeleton, then load weights, then run the smallest understanding/generation path, then increase resolution/tokens/steps.
10. **Record device and dtype in demo output.** Print package versions, selected device, model dtype, autocast dtype, and output artifact paths.

## Common Fix Patterns

- **Hard-coded `.cuda()` methods:** replace with `.to(device)` and pass `device` through constructors and batch movement.
- **Tensor factories missing device:** replace `torch.zeros(...)` with `x.new_zeros(...)` or pass `device=x.device, dtype=x.dtype`.
- **Mixed dtype index assignment:** cast the source to destination dtype before assignment, or allocate the destination using the source-producing module's output dtype.
- **MPS matmul dtype assertion:** remove mixed fp16/bf16 regions; run a minimal repro with the same dtype, then choose one dtype for the whole block.
- **Conv input/bias dtype mismatch under autocast:** keep the module on MPS, but run that module in an autocast-disabled context or explicitly align input dtype to the module weights.
- **Multi-GPU device maps:** replace CUDA memory maps with a single-device map such as `{"": "mps"}` or use normal `.to("mps")` for smaller models.
- **bitsandbytes/NF4/INT8 CUDA quantization:** do not assume it works on MPS. Prefer unquantized, MLX, Core ML, or project-supported MPS alternatives.

## Reference

Read [references/pytorch-mps-notes.md](references/pytorch-mps-notes.md) when debugging a real port, especially for MPS environment variables, AMP/autocast behavior, unsupported ops, Transformers/Accelerate notes, and source links.
