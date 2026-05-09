from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
import sys

# Ensure the ControlNet directory is on sys.path so local imports resolve
script_dir = Path(__file__).resolve().parent
controlnet_dir = script_dir / "ControlNet"
if controlnet_dir.exists():
    sys.path.insert(0, str(controlnet_dir))
# keep script dir on path too
sys.path.insert(0, str(script_dir))

import config
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images for each test JSONL record with ControlNet canny."
    )
    parser.add_argument(
        "--jsonl",
        default="../splits/test_10.jsonl",
        help="Path to test JSONL (relative to ControlNet/ by default).",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root for resolving relative source paths (default: script cwd).",
    )
    parser.add_argument(
        "--output-dir",
        default="../C1",
        help="Directory for generated output images.",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=9.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--low-threshold", type=int, default=100)
    parser.add_argument("--high-threshold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=-1, help="Use -1 for random per sample.")
    parser.add_argument(
        "--a-prompt",
        default="best quality, extremely detailed",
        help="Positive prompt suffix.",
    )
    parser.add_argument(
        "--n-prompt",
        default=(
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        ),
        help="Negative prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of records. 0 means all.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint file (.ckpt or .pth). If omitted, built-in candidates are used.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_model_and_sampler(ckpt_path: str | None = None) -> tuple[object, DDIMSampler]:
    # Resolve model config path relative to the ControlNet package if present,
    # otherwise fall back to the script-relative models directory.
    model_config_path = script_dir / "models" / "cldm_v15.yaml"
    if controlnet_dir.exists():
        alt = controlnet_dir / "models" / "cldm_v15.yaml"
        if alt.exists():
            model_config_path = alt
    model = create_model(str(model_config_path)).cpu()

    # Resolve checkpoint path: prefer explicit arg, else fallback candidates
    if ckpt_path:
        checkpoint_path = ckpt_path
    else:
        checkpoint_candidates = [
            "./models/control_sd15_canny.pth",
            "../control_v11p_sd15_canny.pth",
        ]
        checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), checkpoint_candidates[0])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Try repository helper first; fall back to flexible torch loader
    try:
        sd = load_state_dict(checkpoint_path, location="cuda")
        model.load_state_dict(sd)
    except Exception:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        if isinstance(sd, dict):
            # strip common prefixes
            if any(k.startswith("model.") for k in sd.keys()):
                sd = {k.replace("model.", ""): v for k, v in sd.items()}
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            try:
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    model = model.cuda()
    sampler = DDIMSampler(model)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model, sampler


def generate_one(
    model,
    sampler: DDIMSampler,
    apply_canny: CannyDetector,
    input_image_rgb: np.ndarray,
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    image_resolution: int,
    ddim_steps: int,
    strength: float,
    scale: float,
    eta: float,
    low_threshold: int,
    high_threshold: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    with torch.no_grad():
        img = resize_image(HWC3(input_image_rgb), image_resolution)
        h, w, _ = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = int(np.random.randint(0, 65536))
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([f"{prompt}, {a_prompt}"])],
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt])],
        }
        shape = (4, h // 8, w // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = sampler.sample(
            ddim_steps,
            1,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        return x_samples[0], seed


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    jsonl_path = resolve_path(args.jsonl, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    records = load_jsonl(jsonl_path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records to process.")

    print(f"Records to process: {len(records)}")
    print(f"Output directory: {output_dir}")
    model, sampler = load_model_and_sampler(args.ckpt)
    apply_canny = CannyDetector()

    with manifest_path.open("wt", encoding="utf-8") as mf:
        for idx, record in enumerate(records, start=1):
            source_rel = record.get("source")
            prompt = record.get("prompt", "")
            if not source_rel:
                print(f"[{idx}/{len(records)}] missing source, skipping")
                continue

            source_path = resolve_path(source_rel, project_root)
            source_bgr = cv2.imread(str(source_path))
            if source_bgr is None:
                print(f"[{idx}/{len(records)}] could not read source: {source_path}, skipping")
                continue

            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            try:
                generated_rgb, used_seed = generate_one(
                    model=model,
                    sampler=sampler,
                    apply_canny=apply_canny,
                    input_image_rgb=source_rgb,
                    prompt=prompt,
                    a_prompt=args.a_prompt,
                    n_prompt=args.n_prompt,
                    image_resolution=args.image_resolution,
                    ddim_steps=args.ddim_steps,
                    strength=args.strength,
                    scale=args.scale,
                    eta=args.eta,
                    low_threshold=args.low_threshold,
                    high_threshold=args.high_threshold,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[{idx}/{len(records)}] generation failed for {source_rel}: {e}")
                continue

            out_name = f"{idx:05d}_{Path(source_rel).stem}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))

            manifest_record = {
                "index": idx,
                "source": source_rel,
                "prompt": prompt,
                "output": str(out_path.relative_to(project_root)),
                "seed": used_seed,
            }
            mf.write(json.dumps(manifest_record, ensure_ascii=True) + "\n")
            print(f"[{idx}/{len(records)}] saved {out_path.name}")

    print(f"Done. Manifest: {manifest_path}")

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    main()
