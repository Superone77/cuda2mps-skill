# PyTorch CUDA-to-MPS Porting Notes

Use this reference when CUDA-oriented PyTorch or Transformers code must run on Apple Silicon MPS.

## Source Anchors

- Apple says PyTorch uses the MPS backend for GPU acceleration on Mac and verifies support with `torch.backends.mps.is_available()` and `torch.device("mps")`: https://developer.apple.com/metal/pytorch/
- PyTorch MPS backend docs show moving tensors and modules to `mps`: https://docs.pytorch.org/docs/2.12/notes/mps.html
- PyTorch MPS environment variables include allocator watermarks, profiling, preferred Metal matmul, fast math, and CPU fallback: https://docs.pytorch.org/docs/2.12/mps_environment_variables.html
- PyTorch AMP docs define `torch.autocast(device_type=...)`, autocast dtypes, thread-local state, and `custom_fwd`: https://docs.pytorch.org/docs/2.12/amp.html

## Environment First

Prefer matching the project requirements before patching source:

```bash
python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("mps_available", torch.backends.mps.is_available())
PY
```

For repos pinned around old `transformers`, do not add compatibility monkey patches until the pinned version has been tried. Version skew can cause config fields, RoPE initialization, cache classes, or generation APIs to break in misleading ways.

## Device Helpers

Use one set of device helpers and remove scattered CUDA calls:

```python
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

def seed_all(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
```

MPS normally has one logical GPU. CUDA patterns such as `torch.cuda.device_count()`, `max_memory={i: ...}`, and tensor-parallel sharding need redesign or removal.

## Dtype and Autocast

MPS dtype behavior is more fragile than CUDA. Use small tests before deciding:

```python
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    try:
        x = torch.randn(2, 2, device="mps", dtype=dtype)
        print(dtype, (x @ x).dtype)
    except Exception as e:
        print(dtype, type(e).__name__, e)
```

Rules of thumb:

- CUDA bf16 checkpoints may still contain hard-coded `torch.bfloat16` tensors. Running the whole model in fp16 can create `Index put requires the source and destination dtypes match`.
- MPS autocast can expose dtype mismatches, especially around `conv`, `matmul`, and custom assignment-heavy code. Test both `torch.autocast("mps", dtype=torch.float16)` and `torch.autocast("mps", dtype=torch.bfloat16)` if the PyTorch/macOS version supports them.
- If only one submodule fails under autocast, keep it on MPS and wrap only that submodule with `with torch.autocast("mps", enabled=False): ...` before considering CPU fallback.
- Prefer allocating intermediate tensors from existing tensors: `x.new_zeros(...)`, `torch.zeros(..., device=x.device, dtype=x.dtype)`.
- For assignment into preallocated tensors, cast source to destination dtype at the boundary, not deep inside model math.

## CUDA-only Libraries

These usually need replacement or bypass on MPS:

- `flash_attn`: replace inference paths with `torch.nn.functional.scaled_dot_product_attention`; keep the flash-attn path for CUDA.
- `triton`: most custom Triton CUDA kernels will not run on MPS. Look for a pure PyTorch equivalent.
- `bitsandbytes`: CUDA quantization paths such as NF4/INT8 generally do not map to MPS. Prefer unquantized MPS, MLX-specific ports, Core ML, or project-supported quantization.
- `xformers`: often CUDA-focused; use PyTorch SDPA when possible.

## Transformers and Accelerate

- Prefer `device_map={"": "mps"}` or direct `.to("mps")` over CUDA-oriented multi-GPU maps.
- Check the model repo's pinned `transformers` version before changing config handling.
- Avoid relying on CUDA memory planning arguments (`max_memory={0: ...}`) on MPS.
- Keep tokenization and preprocessing on CPU unless tensors are large and repeated transfer dominates.
- Print model/device/dtype after loading:

```python
param = next(model.parameters())
print(param.device, param.dtype)
```

## MPS Environment Variables

Set these before Python starts:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python script.py
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python script.py
PYTORCH_MPS_PREFER_METAL=1 python script.py
```

Use them carefully:

- `PYTORCH_ENABLE_MPS_FALLBACK=1` falls back to CPU for unsupported MPS ops. This is useful for diagnosis but violates an "all MPS" requirement and can be slow.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables the hard allocation limit and can risk system-wide OOM.
- `PYTORCH_MPS_PREFER_METAL=1` changes matmul implementation selection; test correctness and speed.
- `PYTORCH_MPS_FAST_MATH=1` can alter numerical behavior; do not enable when comparing correctness.

## Debugging Order

1. Reproduce on the smallest input, token count, image size, and diffusion step count.
2. Print package versions and selected device.
3. Print first parameter device/dtype for each major submodule.
4. Wrap only suspect blocks with autocast changes; avoid global dtype flips.
5. Replace CUDA-only kernels with PyTorch equivalents.
6. Enable CPU fallback only to identify unsupported ops, then remove it if native MPS is required.
7. Compare a small CPU run when correctness is suspect; compare shapes, dtypes, and a short output, not full performance.

## Common Error Hints

- `Torch not compiled with CUDA enabled`: remove CUDA-only imports or device calls.
- `NotImplementedError ... not currently implemented for MPS`: try native alternative; use `PYTORCH_ENABLE_MPS_FALLBACK=1` only as a diagnostic/last resort.
- `Input type (...) and bias type (...) should be the same`: an autocast or manual cast boundary is inconsistent; align module input to weight dtype or disable autocast for that module on MPS.
- `Destination NDArray and Accumulator NDArray cannot have different datatype`: MPS matmul received mixed accumulator/output dtypes; choose one dtype for the block or remove nested autocast conflict.
- `Index put requires source and destination dtypes match`: cast the source tensor to destination dtype at the assignment boundary.
