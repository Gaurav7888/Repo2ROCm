# Skill: externalAssetDownload

**Use when:** the agent is in the early planning / dependency-install phase and
has NOT yet downloaded the datasets or pretrained checkpoints that the main
script requires. This is a **proactive** skill — trigger it before the agent
tries to run the first training or evaluation command, not after it crashes.

## The fundamental rule

Large binary assets are **never in the GitHub repository**. GitHub's 25 MB
per-file hard limit means every dataset, checkpoint, annotation archive, and
pseudo-mask set lives on an external host. The plan's
**"EXTERNAL ASSETS REQUIRED"** section lists all detected ones with download
commands. If the executor hasn't acted on that section yet, it will waste turns
recovering from `FileNotFoundError` / `RuntimeError: checkpoint missing` later.

## Trigger signals (any of these in recent turns)

- The agent just finished installing dependencies (pip/apt) and is about to run
  the first script — but no `huggingface-cli download`, `gdown`, `wget`, or
  `bash scripts/download*.sh` appeared in recent commands.
- The plan contains an "EXTERNAL ASSETS REQUIRED" section, but the agent skipped
  over it and jumped straight to `python train.py`.
- A command failed with `FileNotFoundError`, `No such file or directory`, or
  `RuntimeError: checkpoint missing` pointing to a path under `/data/`, `/output/`,
  or `/repo` that should have been downloaded.
- The agent is trying to create synthetic data (random tensors, dummy scenes,
  fake point clouds) instead of downloading the real dataset.

## What to investigate

1. Re-read the plan's "EXTERNAL ASSETS REQUIRED" section (if present) — it
   lists HF ids, Drive links, and download commands.
2. Web-search `<repo_name> dataset download site:huggingface.co` to confirm
   the canonical HF path, especially if the README links Baidu Yun or Google Drive
   (these don't work well inside Docker without accounts/cookies).
3. Check if the repo ships a download script:
   `find /repo -name 'download*.sh' -o -name 'prepare_data*'`

## What to recommend

Provide an ordered download checklist the agent can execute immediately:

```
DOWNLOAD CHECKLIST (run before any train/eval script):

1. ls /data/<dataset_name>          # already on disk? skip.
2. pip install -q huggingface_hub
3. huggingface-cli download <hf_dataset_id> --repo-type dataset \
       --local-dir /data/<dataset_name>
4. huggingface-cli download <hf_model_id> \
       --local-dir /data/models/<checkpoint_name>
5. ls /data/<dataset_name> && ls /data/models/<checkpoint_name>  # verify
```

Adapt the ids and paths from:
- The plan's EXTERNAL ASSETS section, OR
- web_search output, OR
- The repo README (look for lines with `--source_path`, `--checkpoint`,
  or dataset flags like `--colmap_path`, `--data_path`).

## For paper-reproduction runs specifically

The paper's quantitative results (mIoU, PSNR, SSIM, accuracy) depend on the
**exact dataset split and pretrained checkpoint** the authors used.
Synthetic data will never reproduce those numbers.
If the real dataset is gated (login required) or only on Baidu Yun with no HF
mirror, advise the executor to report:
  `echo PAPER_RESULT_NOT_REPRODUCED missing_data: <reason>`
and explain clearly what is missing so a human can fetch it manually.

## Tone

Urgent but practical. The executor may not realise it skipped the download step.
Give it a short diagnosis ("You haven't downloaded the dataset yet — the script
will crash in the next turn"), then the concrete checklist above.
