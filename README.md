# MiniMax H3 Colab Skill — Jeff G4 Workflow

This fork adapts the original MiniMax H3 Colab Codex skill for Jeff's verified Google AI Pro / Colab environment.

## Jeff defaults

- **GPU:** G4
- **High RAM:** off unless explicitly requested
- **Default duration:** 5 seconds
- **Default production preset:** 768×1376 vertical social video
- **Draft preset:** 544×960
- **Landscape preset:** 1376×768
- **Auto mode:** 1 image → FL2VA first-frame, 2–9 images → Ref2VA
- **Low CU guard:** 3 CU
- **No silent fallback to A100**

The workflow verifies the actual remote GPU before downloading or loading H3 models. For G4, it requires a high-VRAM allocation and stops early if the assigned hardware is unexpectedly small.

## Architecture

```text
Codex
  ↓
minimax-h3-colab skill
  ↓
Google Colab CLI + OAuth2
  ↓
Google AI Pro compute units
  ↓
G4 runtime
  ↓
MiniMax H3 / ComfyUI
  ↓
validated MP4 + CU metadata
```

## Install

```bash
git clone https://github.com/Jefflaw3333/minimax-h3-colab-skill.git
cd minimax-h3-colab-skill
./install.sh --force

uv python install 3.12
uv tool install --python 3.12 google-colab-cli
colab --auth=oauth2 usage
```

The first OAuth2 authorization may require opening a browser and approving the Google account.

## Check CU balance

```bash
python3 scripts/jeff_runner.py usage --json
```

## Generate one vertical product video

```bash
./run_colab_inference.sh \
  --image /absolute/path/product.png \
  --prompt /absolute/path/prompt.txt \
  --preset social \
  --duration 5 \
  --output /absolute/path/result.mp4
```

With one image, auto mode uses first-frame / FL2VA.

## Multiple reference images

```bash
./run_colab_inference.sh \
  --image /absolute/path/person.png \
  --image /absolute/path/product.png \
  --prompt /absolute/path/prompt.txt \
  --mode reference \
  --duration 8 \
  --output /absolute/path/result.mp4
```

With 2–9 images, auto mode uses Ref2VA.

## Presets

| Preset | Size | Use |
|---|---:|---|
| `draft` | 544×960 | fast/low-CU prompt testing |
| `social` | 768×1376 | Shorts / Reels / TikTok production |
| `landscape` | 1376×768 | 16:9 video |

Advanced `--width` and `--height` overrides are supported; both must be multiples of 32.

## Cost behavior

The runner:
- checks CU before provisioning;
- records CU before/after when the CLI can return it;
- verifies the actual G4 GPU before expensive model work;
- reuses one session for sequential batch jobs;
- stops sessions it creates when work finishes;
- does not blindly retry a timed-out generation.

## Offline tests

```bash
python3 -m unittest discover -s tests -v
```

## Key files

- `SKILL.md` — instructions Codex should follow
- `scripts/jeff_runner.py` — Jeff's G4-first production runner
- `scripts/runner.py` — upstream-compatible base utilities
- `run_colab_inference.sh` — simple launcher
- `assets/MiniMax_H3_Turbo_Colab.ipynb` — bundled H3 notebook
- `tests/` — offline unit tests

The model itself runs on Google Colab GPU, not on the local computer.
