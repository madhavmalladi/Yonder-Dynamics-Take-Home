# PLAN.md

Working plan for the Yonder Dynamics AI/ML take-home. See `CLAUDE.md` for hard rules (files not to edit, secrets, commit habits) and `README.md` for the full spec. Check items off as they're done; this file tracks progress, `METHODOLOGY.md` is the final write-up.

## 0. Setup
- [x] Clone repo, point `origin` at own GitHub, push.
- [x] Create `.venv`, install `requirements.txt`.
- [x] `download_data.py` reads `ROBOFLOW_API_KEY` from `.env`, not hardcoded.
- [x] Run `download_data.py`, confirm `Sampled-YD-Object-Detection-2/` appears (git-ignored).
- [ ] Point editor's interpreter at `.venv/bin/python` to clear linter import errors.
- [x] `pip install ultralytics`, pin version in `requirements.txt`.
- [ ] Commit setup files (not the dataset or `.env`).

## 1. Explore the data
- [ ] Look through a sample of train/valid images: lighting, backgrounds, object scale, angles, occlusion.
- [ ] Count boxes per class, box size distribution, image brightness/contrast stats.
- [ ] Check for near-duplicate/augmented images split across train vs. valid (leakage risk — README notes many images are augmented variants of the same source photos). Use perceptual hashing (`imagehash`) if needed.
- [ ] Write findings into a short script/notebook, commit it.
- [ ] Decide if the given train/valid split is trustworthy or needs re-splitting; document the decision.

## 2. Baseline model
- [ ] Train YOLOv8n out of the box (resize only) at 512x512 for a handful of epochs on `device="mps"` (fall back to CPU/Colab if MPS misbehaves).
- [ ] Record per-class precision, recall, mAP50, mAP50-95, confusion matrix.
- [ ] Commit training script and baseline results.

## 3. Preprocessing / augmentation
- [ ] Based on step 1 findings, pick at least one real preprocessing/augmentation step beyond resizing (e.g., CLAHE/contrast normalization, HSV/brightness augmentation, occlusion), with a stated reason.
- [ ] Retrain with the change, compare metrics against baseline.
- [ ] Document the before/after comparison — this is the justification the README asks for.

## 4. Final training run
- [ ] Pick best config from step 3, train to convergence (~40 epochs or until metrics plateau).
- [ ] Save final weights (commit if small; otherwise commit the training script to reproduce them).
- [ ] Report final per-class metrics.

## 5. Inference script
- [ ] `predict.py <image_dir> <output_dir>`: writes one `.txt` per image, lines as `class_id x_center y_center width height confidence`, normalized 0-1.
- [ ] Pick and state a confidence threshold.
- [ ] Test on a handful of validation images, sanity-check output format against the label format.

## 6. Error analysis
- [ ] Run the model on validation set, collect false positives and false negatives.
- [ ] Look at specific examples, explain likely causes (lighting, occlusion, scale, class confusion, leakage, etc.).
- [ ] Write this up in `METHODOLOGY.md`.

## 7. Real-world video test
- [ ] Film a mallet-like object (hammer, rolling pin, etc.) in a real environment.
- [ ] Extract frames (ffmpeg/OpenCV), run `predict.py` on them.
- [ ] Report how performance compares to the clean validation set.

## 8. Stretch goals (optional, time-permitting)
- [ ] Efficiency: parameter count, measured inference time, discuss accuracy/speed/size trade-offs for NPU/RKNN deployment. Bonus: quantization or smaller architecture.
- [ ] Active learning: identify least-confident/wrong training images, describe or implement a prioritization strategy for new data collection.

## 9. Write-up
- [ ] Fill out `METHODOLOGY.md`: exact run steps from fresh clone, tested Python version/OS, exact inference command, env var name, thought process bullets, known limitations.
- [ ] Fill out `AI_LOG.md` as AI is used throughout, not just at the end.
- [ ] Update `requirements.txt` with every pinned dependency actually used.

## 10. Submission check
- [ ] Fresh clone in a temp dir, follow own `METHODOLOGY.md` instructions end to end, confirm it works.
- [ ] Confirm repo is public (check in incognito window).
- [ ] Confirm no API key, `.env`, or dataset folder was ever committed (`git log -p -- .env` etc.).
- [ ] Confirm full commit history is pushed (no squashing).
- [ ] Submit repo link via the Google Form.
