#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_colab_inference.sh --image IMAGE [--image IMAGE ...] --prompt PROMPT.txt [options]

Jeff defaults:
  GPU: G4
  High RAM: off
  Duration: 5 seconds
  Preset: social (768x1376)
  Mode: auto (1 image -> first-frame/FL2VA, 2-9 images -> reference/Ref2VA)

Options:
  -i, --image PATH       Local image (repeat 1-9 times)
  -p, --prompt PATH      Local UTF-8 prompt file
  -o, --output PATH      Output MP4 path
      --mode MODE        auto | first-frame | reference
      --preset PRESET    draft | social | landscape
      --duration SEC     4-15 seconds
      --width PX         Advanced override; multiple of 32
      --height PX        Advanced override; multiple of 32
      --gpu MODEL        Explicit GPU override (default: G4)
      --high-mem         Request high-RAM runtime explicitly
      --min-cu N         Refuse to provision when balance < N (default: 3)
      --timeout SEC      Notebook execution timeout
  -h, --help             Show this help

Environment overrides:
  COLAB_AUTH             CLI auth provider (default: oauth2)
  COLAB_GPU              GPU model (default: G4)
  COLAB_MIN_CU           Low-balance guard (default: 3)
  COLAB_EXEC_TIMEOUT     Execution timeout (default: 3600)
  COLAB_SESSION_NAME     Optional named session
  H3_MODE                Default mode (default: auto)
  H3_PRESET              Default preset (default: social)
  H3_DURATION_SECONDS    Default duration (default: 5)

The runner never silently falls back from G4 to A100.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="$SCRIPT_DIR/scripts/jeff_runner.py"
INPUT_IMAGES=()
PROMPT_FILE=""
OUTPUT_TARGET=""
MODE="${H3_MODE:-auto}"
PRESET="${H3_PRESET:-social}"
DURATION="${H3_DURATION_SECONDS:-5}"
GPU="${COLAB_GPU:-G4}"
MIN_CU="${COLAB_MIN_CU:-3}"
TIMEOUT="${COLAB_EXEC_TIMEOUT:-3600}"
HIGH_MEM=0
WIDTH=""
HEIGHT=""

while (($#)); do
  case "$1" in
    -i|--image) (($# >= 2)) || { echo "Missing path after $1" >&2; exit 2; }; INPUT_IMAGES+=("$2"); shift 2 ;;
    -p|--prompt) (($# >= 2)) || { echo "Missing path after $1" >&2; exit 2; }; PROMPT_FILE="$2"; shift 2 ;;
    -o|--output) (($# >= 2)) || { echo "Missing path after $1" >&2; exit 2; }; OUTPUT_TARGET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --high-mem) HIGH_MEM=1; shift ;;
    --min-cu) MIN_CU="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((${#INPUT_IMAGES[@]} < 1 || ${#INPUT_IMAGES[@]} > 9)); then
  echo "Provide between 1 and 9 images with --image." >&2
  exit 2
fi
[[ -n "$PROMPT_FILE" ]] || { echo "A prompt file is required." >&2; exit 2; }
for input_path in "${INPUT_IMAGES[@]}"; do
  [[ -f "$input_path" && -s "$input_path" ]] || { echo "Image is missing or empty: $input_path" >&2; exit 2; }
done
[[ -f "$PROMPT_FILE" && -s "$PROMPT_FILE" ]] || { echo "Prompt file is missing or empty: $PROMPT_FILE" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "Jeff runner not found: $RUNNER" >&2; exit 2; }
command -v colab >/dev/null 2>&1 || { echo "Colab CLI is missing. Install google-colab-cli first." >&2; exit 127; }

ARGS=(single)
for input_path in "${INPUT_IMAGES[@]}"; do ARGS+=(--image "$input_path"); done
ARGS+=(--prompt "$PROMPT_FILE" --gpu "$GPU" --mode "$MODE" --preset "$PRESET" --duration "$DURATION" --min-cu "$MIN_CU" --timeout "$TIMEOUT")
[[ -n "$OUTPUT_TARGET" ]] && ARGS+=(--output "$OUTPUT_TARGET")
[[ -n "$WIDTH" ]] && ARGS+=(--width "$WIDTH")
[[ -n "$HEIGHT" ]] && ARGS+=(--height "$HEIGHT")
(( HIGH_MEM )) && ARGS+=(--high-mem)

exec python3 "$RUNNER" "${ARGS[@]}"
