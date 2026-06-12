# Skill: modelAssetReadiness

**Use when:** the agent is about to launch a training, inference, or evaluation
script that expects a local model checkpoint, dataset directory, COLMAP sparse
reconstruction, pseudo-mask archive, or preprocessed feature cache — AND you
do not see the asset being downloaded or verified in earlier turns.

## Why this always matters

GitHub enforces a **25 MB per-file limit**. Every repository you encounter
stores large data externally. Datasets (often GBs), pretrained checkpoints
(.pth/.pt/.ckpt/.bin/.safetensors), annotation archives, and pseudo-mask
sets are **never inside /repo**. The executor must download them before the
script runs. If the executor skips this step it will always hit a
`FileNotFoundError` or silently train on an empty scene.

## What to research

1. **Exact HuggingFace repo** — search `<repo_name> dataset HuggingFace` and
   `<repo_name> pretrained checkpoint HuggingFace` to find the canonical HF id
   (e.g. `FudanCVL/Ref-Lerf`, `heshuting555/ReferSplat`).
2. **Google Drive / Baidu Yun mirrors** — the README may link both a Drive id
   and a HF mirror; always prefer HF (no login required inside Docker).
3. **Two-stage checkpoint dependency** — some repos (e.g. language-grounded 3DGS
   variants) require a Stage 1 checkpoint trained on the same scene before Stage 2
   can start. If `train.py` checks for `--checkpoint` and raises on missing file,
   that is a Stage 1 dependency.
4. **Download script** — check if the repo ships `scripts/download_*.sh` or
   similar; if it does, that is the canonical download path.

## What to recommend

Provide a concrete, copy-pasteable block the executor can run immediately:

```bash
# Check if asset exists first
ls /data/<asset_name>

# Download from HuggingFace (datasets)
pip install -q huggingface_hub
huggingface-cli download <hf_id> --repo-type dataset --local-dir /data/<name>

# Download from HuggingFace (model checkpoints)
huggingface-cli download <hf_id> --local-dir /data/models/<name>

# Download from Google Drive
pip install -q gdown
gdown https://drive.google.com/uc?id=<drive_id> -O /data/<name>.zip && unzip /data/<name>.zip -d /data/

# Run a repo-provided download script
bash /repo/scripts/download_data.sh
```

If the two-stage checkpoint is missing, advise the executor to:
1. First train Stage 1 (RGB-only, no `--include_feature`) on the target scene
   with reduced iterations (e.g. `--iterations 1000`) so a checkpoint exists.
2. Then re-run Stage 2 (`--include_feature --checkpoint <path>`).
   State this explicitly if the README documents it.

## If source is unknown

Advise the executor to `web_search "<repo_name> <dataset_name> download"` and
read the top result to find the canonical link. Do not instruct it to create
synthetic or mock data — fabricated data will never match the paper's numbers.

## Tone

Preventive and anticipatory. Flag the missing asset before the script crashes,
not after. Give the full download command, the expected target path, and the
flag to pass to the training script (e.g. `--source_path /data/ref-lerf/ramen`).
