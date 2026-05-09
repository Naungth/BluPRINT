#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

@dataclass
class Condition:
    name: str
    real_dir: Path
    generated_dir: Path
    conditioning_dir: Path

def parse_condition(raw: str) -> Condition:
    # Format: NAME:REAL_DIR:GENERATED_DIR:CONDITIONING_DIR
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --condition value: {raw}\n"
            "Expected format: NAME:REAL_DIR:GENERATED_DIR:CONDITIONING_DIR"
        )
    name, real, generated, conditioning = parts
    return Condition(
        name=name.strip(),
        real_dir=Path(real).expanduser().resolve(),
        generated_dir=Path(generated).expanduser().resolve(),
        conditioning_dir=Path(conditioning).expanduser().resolve(),
    )

def ensure_dirs(cond: Condition) -> None:
    for label, folder in (
        ("real_dir", cond.real_dir),
        ("generated_dir", cond.generated_dir),
        ("conditioning_dir", cond.conditioning_dir),
    ):
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"{cond.name}: {label} not found: {folder}")

def parse_last_float(text: str) -> float:
    fid_match = re.search(r"FID:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if fid_match:
        return float(fid_match.group(1))

    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        raise ValueError(f"Could not parse numeric metric value from output:\n{text}")
    return float(matches[-1])

def run_cmd(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return f"{proc.stdout}\n{proc.stderr}".strip()

def run_fid(real_dir: Path, generated_dir: Path, device: str, batch_size: int) -> float:
    output = run_cmd(
        [
            sys.executable,
            "-m",
            "pytorch_fid",
            str(real_dir),
            str(generated_dir),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
    )
    return parse_last_float(output)

def extract_edges(
    image_path: Path,
    size: int,
    extractor: str,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if extractor == "canny":
        return cv2.Canny(img, canny_low, canny_high)
    if extractor == "sobel":
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if extractor == "laplacian":
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        lap = np.absolute(lap)
        return cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if extractor == "none":
        return img

    raise ValueError(f"Unsupported edge extractor: {extractor}")

def list_images(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

def run_ssim(
    conditioning_dir: Path,
    generated_dir: Path,
    edge_size: int,
    edge_extractor: str,
    canny_low: int,
    canny_high: int,
) -> tuple[float, float, int, int]:
    scores: list[float] = []
    missing = 0
    
    # 1. Build a map of the generated files to easily look them up.
    # This removes the "00001_" prefix and ignores the ".png" extension.
    gen_map = {}
    for g_path in list_images(generated_dir):
        parts = g_path.stem.split("_", 1)
        # If the file has an underscore and the first part is a number (like 00001)
        if len(parts) == 2 and parts[0].isdigit():
            base_name = parts[1]
        else:
            base_name = g_path.stem
        gen_map[base_name] = g_path

    # 2. Loop through conditioning images and find their match
    for cond_path in list_images(conditioning_dir):
        cond_base = cond_path.stem
        
        if cond_base not in gen_map:
            missing += 1
            continue
            
        gen_path = gen_map[cond_base]
        
        cond_edges = extract_edges(
            cond_path,
            edge_size,
            edge_extractor,
            canny_low,
            canny_high,
        )
        gen_edges = extract_edges(
            gen_path,
            edge_size,
            edge_extractor,
            canny_low,
            canny_high,
        )
        score = ssim(cond_edges, gen_edges, data_range=255)
        scores.append(float(score))

    if not scores:
        raise ValueError(
            "SSIM could not run because there were no matching filenames between "
            f"{conditioning_dir} and {generated_dir}"
        )

    return float(np.mean(scores)), float(np.std(scores)), len(scores), missing

def run_cmmd(ref_dir: Path, eval_dir: Path) -> float:
    """
    Computes CMMD using the clip-mmd CLI.
    Requires: pip install clip-mmd
    """
    # The clip-mmd CLI accepts two positional arguments for the directories
    output = run_cmd(["clip-mmd", str(ref_dir), str(eval_dir)])
    return parse_last_float(output)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FID, SSIM, and CMMD for multiple experimental conditions."
    )
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="NAME:REAL_DIR:GENERATED_DIR:CONDITIONING_DIR (repeat exactly for C1/C2).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to pytorch-fid (example: cuda, cpu, cuda:0).",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=1,
        help="Batch size passed to pytorch-fid; use 1 for mixed image sizes.",
    )
    parser.add_argument("--edge-size", type=int, default=512)
    parser.add_argument(
        "--edge-extractor",
        choices=["canny", "sobel", "laplacian", "none"],
        default="canny",
        help=(
            "Preprocessor applied before SSIM. Use the same representation as your "
            "conditioning pipeline whenever possible."
        ),
    )
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument(
        "--output-csv",
        default="metric_results.csv",
        help="Where to write the final comparison table.",
    )
    parser.add_argument(
        "--output-json",
        default="metric_results.json",
        help="Optional machine-readable output.",
    )
    args = parser.parse_args()

    conditions = [parse_condition(raw) for raw in args.condition]
    if len(conditions) != 2:
        raise ValueError(
            f"Expected exactly 2 --condition values (C1 and C2), got {len(conditions)}."
        )
    for cond in conditions:
        ensure_dirs(cond)

    results = []
    for cond in conditions:
        print(f"\n=== Running condition: {cond.name} ===")
        fid = run_fid(cond.real_dir, cond.generated_dir, args.device, args.fid_batch_size)
        print(f"FID: {fid:.6f}")

        ssim_mean, ssim_std, matched_count, missing_count = run_ssim(
            cond.conditioning_dir,
            cond.generated_dir,
            edge_size=args.edge_size,
            edge_extractor=args.edge_extractor,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
        print(
            f"SSIM: mean={ssim_mean:.6f}, std={ssim_std:.6f}, "
            f"matched={matched_count}, missing={missing_count}, "
            f"extractor={args.edge_extractor}"
        )

        cmmd = run_cmmd(cond.real_dir, cond.generated_dir)
        print(f"CMMD: {cmmd:.6f}")

        results.append(
            {
                "condition": cond.name,
                "fid": fid,
                "cmmd": cmmd,
                "ssim_mean": ssim_mean,
                "ssim_std": ssim_std,
                "matched_images": matched_count,
                "missing_generated_images": missing_count,
                "real_dir": str(cond.real_dir),
                "generated_dir": str(cond.generated_dir),
                "conditioning_dir": str(cond.conditioning_dir),
            }
        )

    csv_path = Path(args.output_csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "fid",
                "cmmd",
                "ssim_mean",
                "ssim_std",
                "matched_images",
                "missing_generated_images",
                "real_dir",
                "generated_dir",
                "conditioning_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = Path(args.output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nWrote CSV:  {csv_path}")
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
