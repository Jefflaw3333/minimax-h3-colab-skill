---
name: minimax-h3-colab
description: Generate short MiniMax H3 videos through Google Colab CLI using Jeff's G4-first, CU-aware workflow. Use for one-image first-frame video, multi-image reference video, or sequential batches.
---

# MiniMax H3 on Colab — Jeff G4 Workflow

Use `scripts/jeff_runner.py` or `run_colab_inference.sh`.

## Defaults

- GPU: `G4`
- High RAM: off
- Duration: 5 seconds
- Production preset: `social` = 768×1376
- Fast test preset: `draft` = 544×960
- Landscape preset: `landscape` = 1376×768
- Mode: `auto`
  - exactly 1 image → `first-frame` / FL2VA
  - 2–9 images → `reference` / Ref2VA
- Low-CU guard: 3 CU
- Do not silently fall back from G4 to A100.

Jeff's verified Google AI Pro environment has exposed G4 as an RTX PRO 6000 Blackwell-class GPU with roughly 95 GB VRAM. A100 allocations observed on the same account were roughly 40 GB, so G4 is the default production target.

## Required flow

1. Inspect the requested local image(s) and prompt.
2. Check Colab CLI access and CU balance:
   ```bash
   python3 scripts/jeff_runner.py usage --json
   ```
3. If balance is below the configured minimum, stop before provisioning a GPU.
4. For setup validation, prefer the low-cost smoke test: `python3 scripts/jeff_runner.py smoke --session jeff-h3-smoke`. It creates G4, verifies hardware, then stops without downloading H3 models.
5. For real generation, let the runner create G4, then verify the actual remote GPU, VRAM, RAM, disk, and CUDA before any H3 model work.
6. If G4 allocation or the VRAM safety check fails, stop and report the failure. Do not quietly retry on A100.
7. Generate the requested jobs. Keep multiple jobs in one batch so one live session can be reused.
7. Download and validate each MP4.
8. Stop a session created by the runner after the requested work completes.
9. Report output path, elapsed time, actual GPU, and measured CU delta when available.

## One image

For a product image or other image that should anchor the beginning of the video, use auto or explicit first-frame mode.

Example:

```bash
./run_colab_inference.sh \
  --image /absolute/path/product.png \
  --prompt /absolute/path/prompt.txt \
  --preset social \
  --duration 5 \
  --output /absolute/path/product_h3.mp4
```

This resolves to FL2VA and sends the uploaded image through `H3_INPUT_IMAGE`.

## Multiple reference images

With 2–9 images, auto mode resolves to Ref2VA. Keep image order stable and use `<Picture 1>`, `<Picture 2>`, etc. only for reference mode.

Example:

```bash
./run_colab_inference.sh \
  --image /absolute/path/person.png \
  --image /absolute/path/product.png \
  --prompt /absolute/path/prompt.txt \
  --mode reference \
  --preset social \
  --duration 8 \
  --output /absolute/path/result.mp4
```

## Batch jobs

Use a UTF-8 JSON manifest and run one batch:

```json
{
  "jobs": [
    {
      "title": "product-01",
      "reference_images": ["/absolute/path/product.png"],
      "prompt_file": "/absolute/path/prompt.txt",
      "mode": "auto",
      "preset": "social",
      "duration_seconds": 5,
      "output_name": "product-01"
    }
  ]
}
```

```bash
python3 scripts/jeff_runner.py batch \
  --manifest /absolute/path/jobs.json \
  --gpu G4 \
  --min-cu 3 \
  --progress /absolute/path/progress.json \
  --output-dir /absolute/path/outputs
```

Supported per-job fields:
- `mode`: auto | first-frame | reference
- `preset`: draft | social | landscape
- `width`, `height`: optional advanced overrides, both must be multiples of 32
- `duration_seconds`: 4–15 seconds
- `seed`
- `output_name` or `output_path`

## Cost policy

Optimize for successful videos per CU, not idle GPU time.

- Single job: create → verify → execute → download → validate → stop.
- Batch: create once → verify once → run jobs sequentially → stop at end.
- Never leave G4 running by default after the requested work.
- Do not blindly retry a timed-out `colab exec`; the remote kernel may still be working.
- CU metadata is measured from `colab usage` when available. Do not present estimates as measured values.

## Safety

- OAuth2 is the default authentication path.
- Never place OAuth credentials, Google tokens, Hugging Face tokens, or browser credentials in source files, prompts, manifests, committed logs, or skill instructions.
- Do not start a paid/full H3 generation if the user explicitly requested only a smoke test or setup validation.
- The bundled notebook is patched into a temporary copy at runtime so width/height can be passed safely without overwriting the source notebook.
