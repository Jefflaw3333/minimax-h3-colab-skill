#!/usr/bin/env python3
"""Jeff's MiniMax H3 Colab runner.

Production-oriented wrapper around the upstream runner:
- G4 by default, no implicit A100 fallback
- 5-second social/draft presets
- auto-select FL2VA for one image and Ref2VA for 2-9 images
- remote GPU preflight before expensive model work
- low-CU guard and measured CU metadata
- one Colab session reused for sequential batch jobs, then stopped
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import runner as base


PRESETS: dict[str, tuple[int, int]] = {
    "draft": (544, 960),
    "social": (768, 1376),
    "landscape": (1376, 768),
}
SUPPORTED_MODES = {"auto", "first_frame", "reference"}
SUPPORTED_GPUS = {"T4", "L4", "G4", "H100", "A100"}
DEFAULT_GPU = "G4"
DEFAULT_PRESET = "social"
DEFAULT_DURATION = 5.0
DEFAULT_MIN_CU = 3.0
G4_MIN_VRAM_GB = 80.0


def resolve_mode(requested: str, image_count: int) -> str:
    mode = str(requested or "auto").strip().lower().replace("-", "_")
    if mode not in SUPPORTED_MODES:
        raise ValueError("mode must be auto, first-frame, or reference")
    if mode == "auto":
        return "first_frame" if image_count == 1 else "reference"
    if mode == "first_frame" and image_count != 1:
        raise ValueError("first-frame mode requires exactly one image")
    return mode


def resolve_dimensions(
    preset: str | None,
    width: Any = None,
    height: Any = None,
) -> tuple[str, int, int]:
    preset_name = str(preset or DEFAULT_PRESET).strip().lower()
    if preset_name not in PRESETS:
        raise ValueError(f"preset must be one of: {', '.join(PRESETS)}")
    preset_width, preset_height = PRESETS[preset_name]
    resolved_width = preset_width if width in (None, "") else int(width)
    resolved_height = preset_height if height in (None, "") else int(height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("width and height must be positive")
    if resolved_width % 32 or resolved_height % 32:
        raise ValueError("width and height must both be multiples of 32")
    return preset_name, resolved_width, resolved_height


def safe_usage() -> dict[str, Any] | None:
    try:
        return base.get_usage()
    except Exception:
        return None


def cu_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> float | None:
    if not before or not after:
        return None
    try:
        return max(0.0, round(float(before["balance"]) - float(after["balance"]), 4))
    except (KeyError, TypeError, ValueError):
        return None


def require_minimum_cu(min_cu: float) -> dict[str, Any]:
    if not math.isfinite(min_cu) or min_cu < 0:
        raise ValueError("min_cu must be a non-negative finite number")
    usage = base.get_usage()
    if usage["balance"] < min_cu:
        raise RuntimeError(
            f"Colab balance is {usage['balance']:.2f} CU, below the configured "
            f"minimum of {min_cu:.2f} CU. No GPU session was created."
        )
    return usage


def patch_notebook(source: Path, target: Path) -> None:
    """Copy the bundled notebook and make width/height controllable by env vars."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    replaced = False
    for cell in notebook.get("cells", []):
        src = cell.get("source")
        if not isinstance(src, list):
            continue
        updated: list[str] = []
        for line in src:
            if line == "WIDTH, HEIGHT = 1376, 768\n":
                updated.extend([
                    'WIDTH = int(os.environ.get("H3_WIDTH", "1376"))\n',
                    'HEIGHT = int(os.environ.get("H3_HEIGHT", "768"))\n',
                ])
                replaced = True
            else:
                updated.append(line)
        cell["source"] = updated
    if not replaced:
        raise RuntimeError("Bundled notebook layout changed: WIDTH/HEIGHT line was not found.")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _preflight_script() -> str:
    return """import json, os, shutil, torch
ram_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
payload = {
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0,
    "ram_gb": round(ram_total / 2**30, 2),
    "disk_free_gb": round(shutil.disk_usage("/content").free / 2**30, 2),
    "cuda": torch.version.cuda,
}
print("H3_PREFLIGHT_JSON=" + json.dumps(payload, sort_keys=True))
"""


def remote_preflight(session: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h3-preflight-") as temp:
        script = Path(temp) / "preflight.py"
        script.write_text(_preflight_script(), encoding="utf-8")
        output = base.call_colab(
            ["exec", "--session", session, "--timeout", "120", "--file", str(script)],
            label="remote GPU preflight",
            timeout=180,
        )
    match = re.search(r"^H3_PREFLIGHT_JSON=(\{.*\})$", output, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not parse remote GPU preflight output.")
    return json.loads(match.group(1))


def validate_preflight(requested_gpu: str, info: dict[str, Any]) -> list[str]:
    gpu = requested_gpu.upper()
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(
            f"Unsupported GPU {requested_gpu!r}. Use one of: {', '.join(sorted(SUPPORTED_GPUS))}."
        )
    name = str(info.get("gpu_name") or "")
    vram = float(info.get("vram_gb") or 0)
    warnings: list[str] = []
    if not name:
        raise RuntimeError("Colab session was created but CUDA GPU was not detected.")
    if gpu == "G4":
        if vram < G4_MIN_VRAM_GB:
            raise RuntimeError(
                f"Requested G4 but remote preflight reported only {vram:.1f} GB VRAM "
                f"({name}). Stopping before H3 model work."
            )
        if "RTX PRO 6000" not in name.upper():
            warnings.append(
                f"G4 allocation reported {name!r} with {vram:.1f} GB VRAM; "
                "continuing because VRAM is above the safety threshold."
            )
    return warnings


def resolve_jobs(manifest: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must contain a non-empty jobs list.")
    if len(jobs) > 20:
        raise ValueError("A single batch may contain at most 20 videos.")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, raw in enumerate(jobs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Job {index} must be an object.")
        refs = raw.get("reference_images", raw.get("images"))
        if not isinstance(refs, list) or not 1 <= len(refs) <= 9:
            raise ValueError(f"Job {index} must have 1-9 images.")
        image_paths: list[Path] = []
        for image in refs:
            path = Path(str(image)).expanduser().resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Image is missing or empty: {path}")
            image_paths.append(path)

        prompt_file = raw.get("prompt_file")
        if prompt_file:
            prompt_path = Path(str(prompt_file)).expanduser().resolve()
            if not prompt_path.is_file() or prompt_path.stat().st_size == 0:
                raise ValueError(f"Prompt file is missing or empty: {prompt_path}")
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt = str(raw.get("prompt", ""))
        if not prompt.strip():
            raise ValueError(f"Job {index} needs a non-empty UTF-8 prompt.")

        image_count = len(image_paths)
        mode = resolve_mode(raw.get("mode", "auto"), image_count)
        if mode == "reference":
            invalid = sorted(
                {int(n) for n in base.PICTURE_RE.findall(prompt) if int(n) < 1 or int(n) > image_count}
            )
            if invalid:
                raise ValueError(
                    f"Job {index} prompt references Picture {invalid}; it has {image_count} images."
                )

        preset, width, height = resolve_dimensions(
            raw.get("preset", DEFAULT_PRESET),
            raw.get("width"),
            raw.get("height"),
        )
        duration = float(raw.get("duration_seconds", raw.get("duration", DEFAULT_DURATION)))
        if not math.isfinite(duration) or not 4 <= duration <= 15:
            raise ValueError(f"Job {index} duration must be between 4 and 15 seconds.")

        seed = raw.get("seed")
        seed = random.SystemRandom().randrange(0, 2**64) if seed in (None, "") else int(seed)
        if not 0 <= seed < 2**64:
            raise ValueError(f"Job {index} seed must be between 0 and 2^64-1.")

        job_id = base.safe_job_id(raw.get("id"))
        if job_id in seen_ids:
            raise ValueError(f"Job {index} duplicates job id {job_id!r}.")
        seen_ids.add(job_id)
        stem = base.safe_name(raw.get("output_name") or raw.get("title"), f"h3_{index:02d}")
        output = Path(
            str(raw.get("output_path") or output_dir / f"{stem}_{job_id}.mp4")
        ).expanduser().resolve()
        if output in seen_outputs:
            raise ValueError(f"Multiple jobs target the same output path: {output}")
        seen_outputs.add(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        resolved.append({
            "id": job_id,
            "title": str(raw.get("title") or f"Video {index}")[:120],
            "prompt": prompt,
            "reference_images": image_paths,
            "duration_seconds": duration,
            "seed": seed,
            "output_path": output,
            "mode": mode,
            "preset": preset,
            "width": width,
            "height": height,
        })
    return resolved


def start_verified_session(
    session: str,
    gpu: str,
    high_mem: bool,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    gpu = gpu.upper()
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU: {gpu}")
    base.start_session(session, gpu, high_mem, progress_path)
    try:
        info = remote_preflight(session)
        warnings = validate_preflight(gpu, info)
        return {"requested_gpu": gpu, **info, "warnings": warnings}
    except Exception:
        try:
            base.stop_session(session)
        finally:
            raise


def run_batch(
    manifest_path: Path,
    *,
    session: str | None,
    gpu: str,
    high_mem: bool,
    stop_on_complete: bool,
    progress_path: Path | None,
    output_dir: Path,
    exec_timeout: float,
    min_cu: float,
    create_session_if_named: bool = False,
) -> dict[str, Any]:
    if not base.NOTEBOOK.is_file():
        raise FileNotFoundError(f"Bundled inference notebook not found: {base.NOTEBOOK}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = resolve_jobs(manifest, output_dir)

    usage_before = require_minimum_cu(min_cu)
    gpu = gpu.upper()
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU: {gpu}")

    owns_session = session is None or create_session_if_named
    if session is None:
        session = f"h3-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session):
        raise ValueError("Invalid Colab session name.")

    progress: dict[str, Any] = {
        "task": "batch",
        "status": "starting",
        "session": session,
        "gpu_requested": gpu,
        "high_mem": bool(high_mem),
        "cu_before": usage_before["balance"],
        "cu_measurement": "measured",
        "jobs": [
            {
                "id": job["id"],
                "title": job["title"],
                "status": "queued",
                "mode": job["mode"],
                "preset": job["preset"],
                "width": job["width"],
                "height": job["height"],
                "duration_seconds": job["duration_seconds"],
                "output": str(job["output_path"]),
            }
            for job in jobs
        ],
        "log_tail": [],
        "updated_at": base.now_iso(),
    }
    base.write_progress(progress_path, progress)

    def log_line(line: str) -> None:
        if not line or len(line) > 320 or "<Picture" in line or "<Subject" in line or "[Chinese]" in line:
            return
        progress["log_tail"] = (progress["log_tail"] + [line])[-30:]
        progress["updated_at"] = base.now_iso()
        base.write_progress(progress_path, progress)

    results: list[dict[str, Any]] = []
    batch_error: str | None = None
    work_root = manifest_path.parent / f"work_{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        if owns_session:
            preflight = start_verified_session(session, gpu, high_mem)
        else:
            preflight = remote_preflight(session)
            preflight["warnings"] = validate_preflight(gpu, preflight)
            preflight["requested_gpu"] = gpu
        progress["preflight"] = preflight
        progress["gpu_actual"] = preflight.get("gpu_name")
        progress["vram_gb"] = preflight.get("vram_gb")
        progress["status"] = "running"
        progress["updated_at"] = base.now_iso()
        base.write_progress(progress_path, progress)

        for index, job in enumerate(jobs):
            current = progress["jobs"][index]
            job_started = time.monotonic()
            current.update({"status": "uploading", "started_at": base.now_iso()})
            progress["current_job"] = job["id"]
            base.write_progress(progress_path, progress)

            remote_images: list[str] = []
            remote_prefix = f"/content/h3_{job['id']}"
            for image_index, image_path in enumerate(job["reference_images"], start=1):
                suffix = image_path.suffix.lower() or ".img"
                remote = f"{remote_prefix}_image_{image_index}{suffix}"
                base.call_colab(
                    ["upload", "--session", session, str(image_path), remote],
                    label=f"upload image {image_index} for {job['title']}",
                    timeout=600,
                    on_line=log_line,
                )
                remote_images.append(remote)

            prompt_path = work_root / f"{job['id']}.txt"
            prompt_path.write_text(job["prompt"], encoding="utf-8")
            remote_prompt = f"{remote_prefix}_prompt.txt"
            remote_output = f"{remote_prefix}_output.mp4"
            base.call_colab(
                ["upload", "--session", session, str(prompt_path), remote_prompt],
                label=f"upload prompt for {job['title']}",
                timeout=120,
                on_line=log_line,
            )

            job_notebook_dir = work_root / job["id"]
            job_notebook_dir.mkdir(parents=True, exist_ok=True)
            notebook_copy = job_notebook_dir / "MiniMax_H3_Turbo_Colab.ipynb"
            patch_notebook(base.NOTEBOOK, notebook_copy)

            env_values = [
                f"H3_INFERENCE_MODE={job['mode']}",
                f"H3_PROMPT_FILE={remote_prompt}",
                f"H3_DURATION_SECONDS={job['duration_seconds']}",
                f"H3_SEED={job['seed']}",
                f"H3_WIDTH={job['width']}",
                f"H3_HEIGHT={job['height']}",
                f"H3_OUTPUT_PREFIX=MiniMax_H3_{job['id'][:20]}",
                f"H3_OUTPUT_PATH={remote_output}",
                f"H3_JOB_TIMEOUT_SECONDS={min(exec_timeout, 7200)}",
            ]
            if job["mode"] == "reference":
                env_values.append(
                    "H3_REFERENCE_IMAGES=" + json.dumps(remote_images, separators=(",", ":"))
                )
            else:
                env_values.append(f"H3_INPUT_IMAGE={remote_images[0]}")

            exec_args = ["exec", "--session", session, "--timeout", str(exec_timeout)]
            for value in env_values:
                exec_args.extend(["--env", value])
            exec_args.extend(["--file", str(notebook_copy)])
            current.update({"status": "generating", "seed": job["seed"]})
            base.write_progress(progress_path, progress)

            base.call_colab(
                exec_args,
                label=f"generate {job['title']}",
                timeout=exec_timeout + 60,
                on_line=log_line,
            )

            current["status"] = "downloading"
            base.write_progress(progress_path, progress)
            base.call_colab(
                ["download", "--session", session, remote_output, str(job["output_path"])],
                label=f"download {job['title']}",
                timeout=900,
                on_line=log_line,
            )
            base.verify_mp4(job["output_path"])

            usage_after_job = safe_usage()
            elapsed = round(time.monotonic() - job_started, 2)
            job_cu = cu_delta(usage_before, usage_after_job)
            current.update({
                "status": "completed",
                "finished_at": base.now_iso(),
                "elapsed_seconds": elapsed,
                "bytes": job["output_path"].stat().st_size,
                "cu_balance_after": usage_after_job["balance"] if usage_after_job else None,
                "cu_consumed_since_batch_start": job_cu,
                "cu_measurement": "measured" if job_cu is not None else "unavailable",
            })
            results.append({
                "id": job["id"],
                "output": str(job["output_path"]),
                "status": "completed",
                "elapsed_seconds": elapsed,
                "mode": job["mode"],
                "preset": job["preset"],
                "width": job["width"],
                "height": job["height"],
                "duration_seconds": job["duration_seconds"],
            })
            progress["updated_at"] = base.now_iso()
            base.write_progress(progress_path, progress)

    except base.ColabTimeoutError as exc:
        batch_error = str(exc)
        if "current_job" in progress:
            current = next(
                (item for item in progress["jobs"] if item["id"] == progress["current_job"]),
                None,
            )
            if current and current["status"] not in {"completed", "failed"}:
                current.update({"status": "failed", "error": batch_error, "finished_at": base.now_iso()})
        for item in progress["jobs"]:
            if item["status"] == "queued":
                item["status"] = "cancelled"
    except Exception as exc:
        batch_error = str(exc)
        if "current_job" in progress:
            current = next(
                (item for item in progress["jobs"] if item["id"] == progress["current_job"]),
                None,
            )
            if current and current["status"] not in {"completed", "failed"}:
                current.update({"status": "failed", "error": batch_error, "finished_at": base.now_iso()})
        for item in progress["jobs"]:
            if item["status"] == "queued":
                item["status"] = "cancelled"
    finally:
        if (owns_session or stop_on_complete) and session:
            progress["status"] = "stopping_session"
            base.write_progress(progress_path, progress)
            try:
                base.stop_session(session)
                progress["session_status"] = "stopped"
            except Exception as exc:
                progress["session_status"] = "stop_failed"
                progress["cleanup_error"] = str(exc)
                batch_error = batch_error or f"Could not stop Colab session {session}: {exc}"
        elif not owns_session:
            progress["session_status"] = "active"
        shutil.rmtree(work_root, ignore_errors=True)

    usage_after = safe_usage()
    consumed = cu_delta(usage_before, usage_after)
    completed = sum(item["status"] == "completed" for item in progress["jobs"])
    failed = sum(item["status"] == "failed" for item in progress["jobs"])
    cancelled = sum(item["status"] == "cancelled" for item in progress["jobs"])
    progress.update({
        "status": "completed" if not batch_error and completed == len(jobs) else "partial" if completed else "failed",
        "completed_count": completed,
        "failed_count": failed,
        "cancelled_count": cancelled,
        "error": batch_error,
        "cu_after": usage_after["balance"] if usage_after else None,
        "cu_consumed": consumed,
        "cu_measurement": "measured" if consumed is not None else "unavailable",
        "updated_at": base.now_iso(),
        "results": results,
    })
    base.write_progress(progress_path, progress)
    return progress


def build_single_manifest(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    return {
        "jobs": [{
            "title": output.stem,
            "prompt_file": str(Path(args.prompt).expanduser().resolve()),
            "reference_images": args.image,
            "duration_seconds": args.duration,
            "output_path": str(output),
            "mode": args.mode,
            "preset": args.preset,
            "width": args.width,
            "height": args.height,
        }]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Jeff MiniMax H3 Colab runner")
    sub = parser.add_subparsers(dest="command", required=True)

    usage_parser = sub.add_parser("usage", help="Read Colab compute-unit balance")
    usage_parser.add_argument("--json", action="store_true")

    start_parser = sub.add_parser("start", help="Create and verify a Colab GPU session")
    start_parser.add_argument("--session", required=True)
    start_parser.add_argument("--gpu", default=os.environ.get("COLAB_GPU", DEFAULT_GPU))
    start_parser.add_argument("--high-mem", action="store_true")
    start_parser.add_argument("--min-cu", type=float, default=float(os.environ.get("COLAB_MIN_CU", DEFAULT_MIN_CU)))

    stop_parser = sub.add_parser("stop", help="Stop a Colab session")
    stop_parser.add_argument("--session", required=True)

    batch_parser = sub.add_parser("batch", help="Render multiple videos on one verified G4 session")
    batch_parser.add_argument("--manifest", type=Path, required=True)
    batch_parser.add_argument("--session")
    batch_parser.add_argument("--gpu", default=os.environ.get("COLAB_GPU", DEFAULT_GPU))
    batch_parser.add_argument("--high-mem", action="store_true")
    batch_parser.add_argument("--stop-on-complete", action="store_true")
    batch_parser.add_argument("--progress", type=Path)
    batch_parser.add_argument("--output-dir", type=Path, default=Path("output"))
    batch_parser.add_argument("--timeout", type=float, default=float(os.environ.get("COLAB_EXEC_TIMEOUT", "3600")))
    batch_parser.add_argument("--min-cu", type=float, default=float(os.environ.get("COLAB_MIN_CU", DEFAULT_MIN_CU)))

    single_parser = sub.add_parser("single", help="Generate one H3 clip")
    single_parser.add_argument("--image", "-i", action="append", required=True)
    single_parser.add_argument("--prompt", "-p", required=True)
    single_parser.add_argument("--output", "-o")
    single_parser.add_argument("--gpu", default=os.environ.get("COLAB_GPU", DEFAULT_GPU))
    single_parser.add_argument("--high-mem", action="store_true")
    single_parser.add_argument("--mode", default=os.environ.get("H3_MODE", "auto"), choices=["auto", "first-frame", "reference"])
    single_parser.add_argument("--preset", default=os.environ.get("H3_PRESET", DEFAULT_PRESET), choices=sorted(PRESETS))
    single_parser.add_argument("--width", type=int)
    single_parser.add_argument("--height", type=int)
    single_parser.add_argument("--duration", type=float, default=float(os.environ.get("H3_DURATION_SECONDS", DEFAULT_DURATION)))
    single_parser.add_argument("--timeout", type=float, default=float(os.environ.get("COLAB_EXEC_TIMEOUT", "3600")))
    single_parser.add_argument("--min-cu", type=float, default=float(os.environ.get("COLAB_MIN_CU", DEFAULT_MIN_CU)))

    args = parser.parse_args()
    try:
        if args.command == "usage":
            result = base.get_usage()
            if args.json:
                print(json.dumps({"ok": True, **result}, ensure_ascii=False))
            else:
                print(f"Current balance: {result['balance']:.2f} compute units")
                print(f"Usage rate: {result['rate_per_hour']:.2f}/hr")
                print(f"Active assignments: {result['active_assignments']}")
            return 0

        if args.command == "start":
            usage = require_minimum_cu(args.min_cu)
            info = start_verified_session(args.session, args.gpu, args.high_mem)
            print(json.dumps({"ok": True, "cu_before": usage["balance"], **info}, ensure_ascii=False))
            return 0

        if args.command == "stop":
            base.stop_session(args.session)
            print(f"Stopped Colab session: {args.session}")
            return 0

        if args.command == "single":
            first = Path(args.image[0]).expanduser().resolve()
            output = (
                Path(args.output).expanduser().resolve()
                if args.output
                else first.with_name(first.stem + "_minimax_h3.mp4")
            )
            manifest = build_single_manifest(args, output)
            temp_manifest = output.parent / f".h3_{uuid.uuid4().hex[:10]}.json"
            temp_manifest.parent.mkdir(parents=True, exist_ok=True)
            temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            try:
                progress = run_batch(
                    temp_manifest,
                    session=os.environ.get("COLAB_SESSION_NAME"),
                    gpu=args.gpu,
                    high_mem=args.high_mem,
                    stop_on_complete=True,
                    progress_path=None,
                    output_dir=output.parent,
                    exec_timeout=args.timeout,
                    min_cu=args.min_cu,
                    create_session_if_named=True,
                )
            finally:
                temp_manifest.unlink(missing_ok=True)
            print(json.dumps(progress, ensure_ascii=False))
            return 0 if progress["status"] == "completed" else 1

        if args.command == "batch":
            progress = run_batch(
                args.manifest.expanduser().resolve(),
                session=args.session,
                gpu=args.gpu,
                high_mem=args.high_mem,
                stop_on_complete=args.stop_on_complete,
                progress_path=args.progress.expanduser().resolve() if args.progress else None,
                output_dir=args.output_dir.expanduser().resolve(),
                exec_timeout=args.timeout,
                min_cu=args.min_cu,
            )
            print(json.dumps(progress, ensure_ascii=False))
            return 0 if progress["status"] == "completed" else 1

    except Exception as exc:
        if getattr(args, "command", None) == "usage" and getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
