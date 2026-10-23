#!/usr/bin/env python3
"""Preflight or compose one reviewed Wan PSY result against its original frames.

Default operation only validates the assets and records a plan. Actual output
requires --compose and matching visual QC. A raw PASS permits composition; an
explicit structural PASS plus compose_review_allowed permits only a preview
whose final quality remains pending. Failed raw quality is never relabelled.
The asset manifest owns the coordinate transform; crop_xywh is never xyxy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any
import wave


def inverse_affine(matrix: list[list[float]]) -> list[list[float]]:
    """Invert the scale-and-letterbox transform written by the asset exporter."""
    if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
        raise ValueError("affine must be 2 by 3")
    sx, cross_x, tx = matrix[0]
    cross_y, sy, ty = matrix[1]
    if cross_x != 0 or cross_y != 0 or sx <= 0 or sy <= 0:
        raise ValueError("expected positive axis scales and letterbox translation")
    return [[1 / sx, 0, -tx / sx], [0, 1 / sy, -ty / sy]]


def validate_manifest(asset: dict[str, Any]) -> dict[str, Any]:
    width, height = asset["source_size"]
    x, y, crop_w, crop_h = asset["crop_xywh"]
    canvas_w, canvas_h = asset["canvas"]
    fit_w, fit_h = asset["fit_size"]
    pad_x, pad_y = asset["pad_xy"]
    if min(width, height, crop_w, crop_h, canvas_w, canvas_h, fit_w, fit_h) <= 0:
        raise ValueError("dimensions must be positive")
    if min(x, y, pad_x, pad_y) < 0 or x + crop_w > width or y + crop_h > height:
        raise ValueError("crop must fit the original image; crop is xywh")
    if pad_x + fit_w > canvas_w or pad_y + fit_h > canvas_h:
        raise ValueError("letterbox content exceeds canvas")
    expected = [[fit_w / crop_w, 0, pad_x - x * fit_w / crop_w],
                [0, fit_h / crop_h, pad_y - y * fit_h / crop_h]]
    actual = asset["affine_source_to_canvas"]
    if any(not math.isclose(a, b, abs_tol=1e-8) for ar, er in zip(actual, expected)
           for a, b in zip(ar, er)):
        raise ValueError("affine disagrees with crop, fit or padding")
    inverse = inverse_affine(actual)
    valid, encoded = asset["valid_frames"], asset["encoded_frames"]
    indices = asset["source_frame_indices"]
    if valid <= 0 or encoded < valid or len(indices) != encoded:
        raise ValueError("frame counts disagree")
    if indices[:valid] != list(range(indices[0], indices[0] + valid)):
        raise ValueError("valid source frames must be consecutive")
    if any(i != indices[valid - 1] for i in indices[valid:]):
        raise ValueError("padding may only repeat the last valid source frame")
    if asset["fps"] != 24:
        raise ValueError("this PSY comparison uses the normalized 24 fps timeline")
    return {"crop_xywh": [x, y, crop_w, crop_h], "inverse_affine": inverse,
            "valid_frames": valid, "source_first_frame": indices[0],
            "source_last_frame": indices[valid - 1], "fps": asset["fps"]}


def require_visual_pass(report: dict[str, Any] | None, raw: Path) -> None:
    inspection_only = bool(report and report.get("compose_review_allowed") is True
                           and report.get("structural_quality") == "PASS"
                           and report.get("visual_quality") in ("REVIEW_REQUIRED", "FAIL")
                           and report.get("final_quality") == "PENDING")
    if report is None or (report.get("visual_quality") != "PASS" and not inspection_only):
        raise ValueError("visual QC is absent or failed; no composite may be written")
    if Path(report.get("raw_server_file", "")).resolve() != raw.resolve():
        raise ValueError("visual QC describes a different raw video")


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-count_frames", "-show_entries",
                             "stream=width,height,r_frame_rate,nb_read_frames",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(float, stream["r_frame_rate"].split("/"))
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": numerator / denominator, "frames": int(stream["nb_read_frames"])}


def audio_path(asset: dict[str, Any], override: Path | None = None) -> tuple[Path, str]:
    if override is not None:
        return override, "cli_override"
    if asset.get("audio_source"):
        return Path(asset["audio_source"]), "asset_audio_source"
    return Path(asset["source"]), "asset_source"


def probe_audio(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                             "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"no audio stream in {path}; supply --audio-source or asset[audio_source]")
    stream = data["streams"][0]
    duration = stream.get("duration") or data.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        raise ValueError(f"audio duration unavailable: {path}")
    return {"codec": stream["codec_name"], "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]), "duration": float(duration)}


def audio_plan(asset: dict[str, Any], plan: dict[str, Any], override: Path | None = None) -> dict[str, Any]:
    path, origin = audio_path(asset, override)
    info = probe_audio(path)
    start = plan["source_first_frame"] / plan["fps"]
    duration = plan["valid_frames"] / plan["fps"]
    if info["duration"] + 1/info["sample_rate"] < start + duration:
        raise ValueError(f"audio ends before required source frame: {path}")
    return {"source": str(path), "origin": origin, "probe": info,
            "start_seconds": start, "duration_seconds": duration,
            "expected_pcm_frames": round(duration*info["sample_rate"])}


def audio_extract_command(audio: dict[str, Any], output: Path) -> list[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", audio["source"],
            "-ss", f"{audio['start_seconds']:.9f}", "-t", f"{audio['duration_seconds']:.9f}",
            "-map", "0:a:0", "-c:a", "pcm_s16le", str(output)]


def extract_audio(audio: dict[str, Any], output: Path) -> dict[str, Any]:
    subprocess.run(audio_extract_command(audio, output), check=True)
    with wave.open(str(output), "rb") as stream:
        decoded = {"pcm_frames": stream.getnframes(), "sample_rate": stream.getframerate(),
                   "channels": stream.getnchannels(), "sample_width": stream.getsampwidth()}
    if (decoded["pcm_frames"] != audio["expected_pcm_frames"]
            or decoded["sample_rate"] != audio["probe"]["sample_rate"]
            or decoded["channels"] != audio["probe"]["channels"]
            or decoded["sample_width"] != 2):
        raise ValueError(f"extracted audio format or exact sample count differs: {decoded}")
    return decoded


def preflight(asset: dict[str, Any], raw: Path, qc: dict[str, Any] | None,
              audio_override: Path | None = None) -> dict[str, Any]:
    plan = validate_manifest(asset)
    source_probe, raw_probe = probe(Path(asset["source"])), probe(raw)
    if [source_probe["width"], source_probe["height"]] != asset["source_size"]:
        raise ValueError("original dimensions differ from manifest")
    if source_probe["frames"] <= plan["source_last_frame"]:
        raise ValueError("original does not contain all mapped frames")
    if [raw_probe["width"], raw_probe["height"]] != asset["canvas"]:
        raise ValueError("raw canvas differs from the source/mask transform")
    if raw_probe["frames"] < plan["valid_frames"]:
        raise ValueError("raw video is shorter than the valid source segment")
    if source_probe["fps"] != 24 or raw_probe["fps"] != 24:
        raise ValueError("source/raw rate differs from the 24 fps manifest")
    try:
        require_visual_pass(qc, raw)
        quality_error = None
    except ValueError as error:
        quality_error = str(error)
    return {**plan, "source": asset["source"], "raw": str(raw),
            "source_probe": source_probe, "raw_probe": raw_probe,
            "visual_quality": None if qc is None else qc.get("visual_quality"),
            "compose_review_only": bool(qc and qc.get("visual_quality") != "PASS"),
            "compose_allowed": quality_error is None, "quality_error": quality_error,
            "semantic_source": asset["semantic_source"],
            "audio": audio_plan(asset, plan, audio_override),
            "preservation": "exact original bytes outside alpha and in original other-person masks before encoding"}


def compose(asset: dict[str, Any], raw: Path, qc: dict[str, Any], output: Path,
            model_path: Path, audio_override: Path | None = None) -> dict[str, Any]:
    require_visual_pass(qc, raw)  # Before allocating a video writer or output file.
    plan = preflight(asset, raw, qc, audio_override)
    if output.exists():
        raise ValueError("output already exists; use a new reviewable filename")
    import sys
    sys.path.insert(0, str(model_path.parent))
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    n, first = plan["valid_frames"], plan["source_first_frame"]
    source_w, source_h = asset["source_size"]
    canvas_w, canvas_h = asset["canvas"]
    inverse = np.float32(plan["inverse_affine"])
    forward = np.float32(asset["affine_source_to_canvas"])
    semantic = Path(asset["semantic_source"])
    with np.load(semantic) as cached:
        key = "target" if "target" in cached.files else "source"
        target = cached[key][first:first+n].astype(np.float32)
        others = cached["others"][first:first+n].astype(np.float32)
    if len(target) != n or target.max() > 1 or others.max() > 1:
        raise ValueError("source semantic mask count or range is invalid")
    def capture(path: str):
        if not hasattr(cv2, "CAP_PROP_N_THREADS"):
            raise RuntimeError("OpenCV decoder thread control unavailable")
        result = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        if not result.isOpened():
            raise ValueError(f"cannot open video with two decoder threads: {path}")
        return result

    raw_cap = capture(str(raw))
    model = YOLO(str(model_path))
    selected, records = [], []
    previous = previous_gray = None
    for frame in range(n):
        raw_ok, generated = raw_cap.read()
        if not raw_ok:
            raise ValueError(f"missing valid frame {frame}")
        source_mask = cv2.resize(target[frame], (source_w, source_h), interpolation=cv2.INTER_LINEAR)
        anchor = cv2.warpAffine(source_mask, forward, (canvas_w, canvas_h))
        result = model.predict(generated, imgsz=max(asset["canvas"]), classes=[0],
                               conf=.12, retina_masks=True, device="cpu", verbose=False)[0]
        if result.masks is None:
            raise ValueError(f"f{frame}: no generated person; refused")
        gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        warped = None
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(gray, previous_gray, None, .5, 3, 19, 3, 5, 1.2, 0)
            gx, gy = np.meshgrid(np.arange(canvas_w, dtype=np.float32), np.arange(canvas_h, dtype=np.float32))
            warped = cv2.remap(previous, gx+flow[:, :, 0], gy+flow[:, :, 1], cv2.INTER_LINEAR)
        candidates = []
        masks = result.masks.data.cpu().numpy()
        for index, mask in enumerate(masks):
            overlap = float(np.minimum(mask, anchor).sum()/max(1, np.maximum(mask, anchor).sum()))
            temporal = 0. if warped is None else float(np.minimum(mask, warped).sum()/max(1, np.maximum(mask, warped).sum()))
            candidates.append({"index": index, "score": 2*overlap+1.5*temporal,
                               "source_iou": overlap, "temporal_iou": temporal})
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        gap = best["score"]-ranked[1]["score"] if len(ranked)>1 else best["score"]
        if best["source_iou"] < .25 or gap < .2 or (warped is not None and best["temporal_iou"] < .18):
            raise ValueError(f"f{frame}: ambiguous generated target; {ranked}")
        previous, previous_gray = masks[best["index"]].copy(), gray
        selected.append(previous)
        records.append({"frame": frame, "source_frame": first+frame, "chosen": best, "gap": gap})
    raw_cap.release()
    # No video output is opened until every valid frame has passed identity selection.
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = output.with_suffix(".source-audio.wav")
    extracted_audio = extract_audio(plan["audio"], audio)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{source_w}x{source_h}", "-r", "24", "-i", "pipe:0", "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "12",
               "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-shortest", "-movflags", "+faststart", str(output)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    x, y, crop_w, crop_h = asset["crop_xywh"]
    checks, saved_alpha = [], []
    source_cap, raw_cap = capture(asset["source"]), capture(str(raw))
    for skipped in range(first):
        if not source_cap.read()[0]:
            raise ValueError(f"missing original skipped frame {skipped}")
    try:
        for i, mask in enumerate(selected):
            ok, original = source_cap.read(); raw_ok, generated = raw_cap.read()
            if not ok or not raw_ok:
                raise ValueError(f"missing source/raw composition frame {i}")
            generated_full = cv2.warpAffine(generated, inverse, (source_w, source_h), flags=cv2.INTER_CUBIC)
            generated_mask = cv2.warpAffine(mask, inverse, (source_w, source_h), flags=cv2.INTER_LINEAR)
            original_mask = cv2.resize(target[i], (source_w, source_h), interpolation=cv2.INTER_LINEAR)
            protect = cv2.resize(np.uint8(others[i] >= .1), (source_w, source_h), interpolation=cv2.INTER_NEAREST)
            protect = cv2.dilate(protect, np.ones((3, 3), np.uint8)) > 0
            alpha = np.maximum(original_mask, generated_mask)
            alpha = cv2.GaussianBlur(cv2.dilate(alpha, np.ones((3,3), np.uint8)), (0,0), .7)
            alpha[protect] = 0
            alpha[:y] = 0; alpha[y+crop_h:] = 0; alpha[:,:x] = 0; alpha[:,x+crop_w:] = 0
            alpha[alpha < .002] = 0
            result = np.uint8(np.clip(generated_full*alpha[...,None]+original*(1-alpha[...,None]), 0, 255))
            result[alpha==0] = original[alpha==0]
            if not np.array_equal(result[alpha==0], original[alpha==0]):
                raise AssertionError(f"f{i}: original outside mask changed")
            if not np.array_equal(result[protect], original[protect]):
                raise AssertionError(f"f{i}: original other person changed")
            writer.stdin.write(result.tobytes())
            checks.append({"frame": i, "source_frame": first+i, "outside_mask_max_change": 0,
                           "other_person_max_change": 0, "protected_pixels": int(protect.sum())})
            saved_alpha.append(np.uint8(cv2.resize(alpha, tuple(asset["semantic_size"]))*255))
        writer.stdin.close()
        error = writer.stderr.read().decode()
        if writer.wait() != 0:
            raise RuntimeError(error)
    except BaseException:
        writer.kill(); writer.wait()
        raise
    finally:
        source_cap.release(); raw_cap.release()
    encoded = probe(output)
    if encoded["frames"] != n or encoded["fps"] != 24:
        raise ValueError("final encoded frame count or rate mismatch")
    report = {**plan, "output": str(output), "encoded": encoded,
              "extracted_audio": extracted_audio,
              "selection": records, "pixel_checks": checks, "visual_qc_report": qc,
              "final_visual_quality": "PENDING",
              "output_role": "protected composite preview; independent final visual QC required"}
    np.savez_compressed(output.with_suffix(".masks.npz"),
                        generated_canvas=np.uint8(np.stack(selected)*255),
                        alpha_semantic_size=np.stack(saved_alpha))
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--audio-source", type=Path,
                        help="original full clip WAV; cut still uses source_first_frame / fps")
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--segmentation-model", type=Path,
                        default=Path("/var/lib/comfyui/gpu5/optimization-runtime/yolo11m-seg.pt"))
    args = parser.parse_args()
    asset = json.loads(args.assets.read_text())
    qc = json.loads(args.quality_report.read_text()) if args.quality_report else None
    report = preflight(asset, args.raw, qc, args.audio_source)
    args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_report.write_text(json.dumps(report, indent=2)+"\n")
    if args.compose:
        if args.output is None:
            parser.error("--compose requires --output")
        compose(asset, args.raw, qc, args.output, args.segmentation_model, args.audio_source)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
