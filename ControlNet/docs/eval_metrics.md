# Evaluation Metrics (FID / CMMD / SSIM)

This gives you one repeatable command that runs your two conditions (`C1`, `C2`) and writes a comparison CSV.

## 1) Create a conda environment and install dependencies

From `ControlNet/`:

```bash
conda env create -f environment.yaml
conda activate control
python -m pip install --upgrade pip
python -m pip install -e .
```

This repo already pins `torch` and related packages through `environment.yaml`. The extra `pip install -e .` installs the metric tooling (`pytorch-fid`, `cmmd`, `scikit-image`, etc.) into the same conda environment.

This install now includes CMMD as a required dependency.

## 2) Install CMMD runner script

The evaluation script requires a CMMD command template. One practical setup is:

```bash
git clone https://github.com/sayakpaul/cmmd-pytorch external/cmmd-pytorch
python -m pip install -r external/cmmd-pytorch/requirements.txt
```

## 3) Run both conditions

Run from `ControlNet/`:

```bash
python tools/evaluate_conditions.py \
  --device cuda \
  --edge-extractor canny \
  --condition "C1:/path/to/real_images:/path/to/generated/C1:/path/to/conditioning/C1" \
  --condition "C2:/path/to/real_images:/path/to/generated/C2:/path/to/conditioning/C2" \
  --cmmd-cmd-template "python external/cmmd-pytorch/main.py --ref_dir {ref_dir} --eval_dir {eval_dir}" \
  --output-csv results/metrics.csv \
  --output-json results/metrics.json
```

## Notes

- `FID` and `CMMD`: lower is better.
- `SSIM`: higher is better.
- SSIM applies `--edge-extractor` to both conditioning and generated images before comparison.
- Set `--edge-extractor` to match the representation used in your training conditioning pipeline; mismatched preprocessors can make SSIM artificially low.
- Supported extractors: `canny` (default), `sobel`, `laplacian`, `none`.
- Default preprocessing settings are `edge_size=512`, `canny_low=100`, `canny_high=200` (Canny thresholds only apply when `--edge-extractor canny`).
