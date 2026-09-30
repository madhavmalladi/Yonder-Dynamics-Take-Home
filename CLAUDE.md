# CLAUDE.md

Yonder Dynamics AI/ML take-home: train an object detector for **bottle (id 0)** and **mallet (id 1)**. `README.md` is the source of truth for the task, so re-read it when unsure.

## Hard rules

- **Do not edit `sampling/`.** It shows how the dataset was built and is not part of the solution.
- **Treat `Sampled-YD-Object-Detection-2/` as read-only.** Write any preprocessed or relabeled data to a new folder. Deleting the download and fetching it again must still reproduce the results.
- **Never commit the Roboflow API key, `.env`, or the dataset folder.** Read the key from the env var `ROBOFLOW_API_KEY`. Don't hardcode it or paste it into logs or errors, because Roboflow error URLs can contain the key. If a key is ever committed, revoke it in Roboflow and don't rewrite history.
- **Git history is graded.** Make small, frequent commits as work progresses. Never squash, amend, or force-push. Check `git status` before every `git add -A`. Only commit when the user asks.
- **Files over 100 MB are rejected by GitHub.** Commit weights only if they're small (YOLOv8n is about 6 MB, which is fine). Otherwise commit a script that reproduces them.
- **`.gitignore` already ignores `data/`, `sampled/`, `.venv/`, `.env`, and `Sampled-YD-Object-Detection-*/`.** Anything that must be committed can't live in `data/`.
- Keep `requirements.txt` up to date with **pinned versions** of every library used. It currently lists only the sampling deps.

## Dataset facts

- 800 train and 199 valid images. There is **no test split**, so remove the `test:` line from `data.yaml` in a copy if the framework complains. Don't edit the original.
- Labels are in YOLO format: `class x_center y_center width height`, normalized to 0–1.
- There are 685 bottle boxes and 537 mallet boxes, which is fairly balanced.
- Many images are augmented variants (rotation, brightness, exposure) of the same source photos. **This risks train/valid leakage**, so check for near-duplicates across splits and explain the split in METHODOLOGY.md.
- License: CC BY 4.0.

## Required deliverables

1. A training pipeline. YOLO (Ultralytics) is the expected path, and detection is weighted higher than classification.
2. **At least one real preprocessing or augmentation step** beyond resizing, justified by what the raw data actually shows. Default settings don't count.
3. Metrics **per class**: precision, recall, mAP@0.5, mAP@0.5:0.95, and a confusion matrix. Don't rely on "accuracy".
4. An **inference script** called as `python <script> <image_dir> <output_dir>`. For each image it writes a `.txt` with the same name, one detection per line: `class_id x_center y_center width height confidence`, normalized 0–1 with dataset class ids. State the confidence threshold used.
5. **Error analysis** in METHODOLOGY.md: look at specific FPs and FNs and explain *why* they happen.
6. **Real-world video test.** Film a mallet-like object (hammer, rolling pin, etc.) with a phone, extract frames, run the model on them, and report how it does out of distribution.
7. `METHODOLOGY.md`: exact run steps from a fresh clone (tested Python version and OS, the inference command, the env var name) plus bullet-point reasoning and known limitations. The user writes this in their own words.
8. `AI_LOG.md`: one entry per significant AI use, covering what was asked, what was kept or rewritten, what the AI got wrong, and how it was verified.

## Stretch goals (optional)

- Efficiency: parameter count, inference time, the accuracy vs. speed vs. size trade-off for NPU/RKNN edge deployment, and optionally quantization or a smaller model.
- Hard example mining: find low-confidence or wrong training images and propose a data-collection strategy.

## Rubric emphasis

Correctness, legibility, design judgment, handling ambiguity, understanding, **AI verification** (show tests and checks of AI-written code and claims), and stretch goals.

## Environment

- macOS, Apple M3 (8 cores), Python 3.12.4. Use a `.venv` in the repo root.
- There is no CUDA. Try `device="mps"` for Ultralytics and fall back to CPU if MPS is buggy or slow. Colab GPU is also an option.
- Remote: `origin` is https://github.com/madhavmalladi/Yonder-Dynamics-Take-Home.git (public).
