# HyperScale - DIV2K SRGAN Super-Resolution

HyperScale is an engineering mini-project for single-image super-resolution. Its primary experiment trains a residual SRGAN on the locally supplied DIV2K HR images paired with the corresponding `LR_unknown/X4` images. The existing Pix2Pix restoration implementation is retained for reference and legacy tests, but `app.py` and the SRGAN commands use the new SRGAN pipeline.

**No quality claim is made before evaluating the trained checkpoints.** This project uses the local files already present; it does not download another dataset. `LR_unknown` comes from a non-official source and its degradation method is unknown.

## Verified local data

Default paths in `configs/default.yaml` (relative to the project root):

```text
DIV2K/DIV2K/DIV2K_train_HR
DIV2K/DIV2K/DIV2K_train_LR_unknown/X2
DIV2K/DIV2K/DIV2K_train_LR_unknown/X3
DIV2K/DIV2K/DIV2K_train_LR_unknown/X4
```

The pairing verifier was run against the files in this workspace:

| Scale | HR files | LR files | Matched exact-scale pairs | Unmatched | Duplicates | Invalid/excluded |
|------:|----------:|---------:|--------------------------:|----------:|-----------:|-----------------:|
| X2 | 900 | 900 | 900 | 0 | 0 | 0 |
| X3 | 900 | 900 | 900 | 0 | 0 | 0 |
| X4 | 900 | 900 | 900 | 0 | 0 | 0 |

All scanned training files were PNG. Example names are `0001.png` in HR and `0001x4.png` in LR. Every scanned LR image had dimensions exactly one-half, one-third, or one-quarter of the matching HR image for X2/X3/X4 respectively. The X4 run manifest has **810 train identities and 90 validation identities**, assigned by source image with seed 42. The manifest records HR path, LR path, source ID, dimensions, scale, and split. X2/X3 entries reuse the same source identity splits; they cannot leak across splits.

The extracted archive also has `DIV2K_test_LR_unknown/X2`, `X3`, and `X4`, each with 100 LR inputs (names begin `0901xN.png`). There is no `DIV2K_test_HR` directory, so there are no verified test reference pairs and test PSNR/SSIM cannot be computed. The 90-image split is called **validation**, not test.

## Pair discovery and dataset behavior

Run the verification/pairing report from the project root:

```powershell
python -m data.build_manifest --force
```

The script matches case-insensitive filenames after stripping scale suffixes such as `x4`; it does not pair by sorted order or file counts. It detects ambiguous normalized filename keys, reports unmatched files and unreadable images, and excludes pairs unless the dimensions match the declared scale exactly. It writes:

- `data/splits/div2k_train.json` â€” deterministic all-scale manifest and split assignment.
- `outputs/dataset_pairing_report.json` â€” per-scale matches, unmatched/duplicate/invalid/excluded information.

The project does not modify or regenerate the supplied LR images. It loads samples on demand. Training draws aligned random crops (default 64Ã—64 LR with the corresponding 256Ã—256 HR crop), applying the same mirror/90-degree rotation to both. Validation uses the deterministic center crop. Images are RGB, normalized to `[-1, 1]`. Images smaller than the crop size use the largest square crop available and are logged.

To intentionally rebuild a manifest after changing files, run `python -m data.build_manifest --force`. The seed and 90/10 split are configurable in `configs/default.yaml`. Do not change the validation split after model selection and then report it as an untouched test set.

## Scale support

- **X4 is the configured initial experiment.** Two learned PixelShuffle Ã—2 blocks yield exactly 4Ã— width and height.
- **X2 is implemented** with one PixelShuffle Ã—2 block. Train it separately by setting `data.scale: 2`; checkpoints use an `_X2` filename suffix.
- **X3 is dataset-verified only and not supported by the generator/UI.** It requires a separate upsampling architecture; the code raises an explicit error rather than returning the wrong dimensions.
- The UI only enables scales with a valid corresponding trained checkpoint.

## SRGAN model and losses

The generator contains an initial RGB feature convolution, residual blocks with convolution/BatchNorm/PReLU, a long skip connection, and PixelShuffle upsampling. `Tanh` outputs match the `[-1, 1]` tensor representation.

The discriminator progressively extracts HR-image features and returns a raw real/fake logit. It is trained with `BCEWithLogitsLoss`; no sigmoid is applied before the loss.

Training is deliberately staged:

1. **Generator pretraining:** L1 pixel reconstruction on aligned LR/HR crops. Select the best pretraining checkpoint by validation PSNR.
2. **SRGAN adversarial fine-tuning:** resume the pretrained generator, train the discriminator on real HR and generated SR images, and optimize generator L1 content + weighted adversarial loss. Best SRGAN checkpoint selection uses validation PSNR. Logs additionally include SSIM and separate generator, discriminator, reconstruction, adversarial, and optional perceptual losses.

Optional VGG19 content loss is disabled by default. Enabling it requires a verified, full local VGG19 state dictionary in `training.vgg19_weights_path`; no random VGG weights or automatic network downloads are treated as perceptual features. Without such weights, keep `perceptual_weight: 0`.

## Install and environment

Python 3.11 is recommended; use a Python release compatible with your PyTorch build.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\verify_environment.py
```

CUDA is selected automatically when available. The current verified environment is Python 3.14.7, PyTorch 2.14.1 CPU-only, CUDA unavailable. CPU is useful for tests and small pipeline checks but full DIV2K training may be very slow. Use a compatible CUDA-enabled PyTorch distribution for practical full training.

## Train

### Generator pretraining

```powershell
python -m training.pretrain --config configs/default.yaml
```

For a fast pipeline smoke run on a small subset, use a temporary output/checkpoint configuration or run with small overrides:

```powershell
python -m training.pretrain --config configs/default.yaml --limit-images 8 --epochs 1 --batch-size 2 --lr-crop-size 16 --features 8 --residual-blocks 2 --max-minutes 5
```

### Adversarial SRGAN stage

```powershell
python -m training.train_srgan --config configs/default.yaml
```

Smoke subset:

```powershell
python -m training.train_srgan --config configs/default.yaml --limit-images 8 --epochs 1 --batch-size 2 --lr-crop-size 16 --features 8 --residual-blocks 2 --max-minutes 5
```

Both training stages support `--resume <checkpoint>`, `--limit-images`, and configurable epoch/batch/crop/model-width/time-budget overrides. The staged run does not launch automatically as a multi-hour job. Typical outputs:

```text
checkpoints/generator_pretrain_latest.pth
checkpoints/generator_pretrain_best.pth
checkpoints/generator_pretrain_inference.pth
checkpoints/srgan_latest.pth
checkpoints/srgan_best.pth
checkpoints/srgan_generator_inference.pth
logs/srgan/srgan_training.csv
outputs/srgan/pretrain_epoch_*.png
outputs/srgan/srgan_epoch_*.png
```

X2 checkpoints include `_X2` in the filename. Checkpoints contain model/configuration state, and training checkpoints also contain optimizer, epoch/step, elapsed time, and random-generator state. No old Pix2Pix checkpoint is loaded as SRGAN weights.

**Full configured X4 training command** (810 train pairs, 90 validation pairs; runtime depends on GPU/CPU and is not guaranteed within an hour):

```powershell
python -m training.pretrain --config configs/default.yaml
python -m training.train_srgan --config configs/default.yaml
```

For low memory, reduce `training.batch_size`, `data.lr_crop_size`, or `training.generator_features`. Out-of-memory errors suggest these options. The current default crop is LR 64/HR 256 and model width is 32 features with 8 residual blocks.

### Current trained checkpoint (X4)

Trained on all 810 X4 training pairs on CPU (width 32, 8 residual blocks, 64×64 LR crops): 20 generator-pretraining epochs, then adversarial fine-tuning stopped by the 60-minute budget (L1 + 0.001 adversarial loss, no VGG loss). The generator predicts a residual over the bicubic upsample. Validation on all 90 held-out pairs:

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Bicubic | 20.02 dB | 0.417 |
| Pretrained generator | 20.90 dB | 0.488 |
| SRGAN | 21.11 dB | 0.502 |

The model beats bicubic but is lightly trained; longer training (`--resume`, more epochs, GPU) would improve it further. LR_unknown uses an undocumented degradation, so scores are indicative only.
## Evaluation

The project compares bicubic baseline, pretrained generator, and final SRGAN output on the **same validation pairs**:

```powershell
python -m evaluation.evaluate_srgan --config configs/default.yaml
```

Optional small validation run:

```powershell
python -m evaluation.evaluate_srgan --config configs/default.yaml --limit-images 8
```

Outputs are `outputs/evaluation/srgan_validation.json` and `outputs/evaluation/srgan_validation_grid.png`. The report lists scale, validation pair count, checkpoints/epochs, average inference time, and per-method PSNR/SSIM. PSNR/SSIM are reference metrics, not â€œaccuracyâ€; neither alone captures perceptual quality. The local LR_unknown process may not be equivalent to bicubic degradation. Metrics assume LR and HR remain aligned because scale and dimensions match. There is no paired HR test set in the supplied archive, so no test score is claimed.

## Inference and Streamlit

Reusable Python inference:

```python
from PIL import Image
from inference.super_resolve import super_resolve

result = super_resolve(
    Image.open("low_resolution.png"),
    "checkpoints/srgan_generator_inference.pth",
    expected_scale=4,
)
result.image.save("super_resolved.png")
result.bicubic.save("bicubic_baseline.png")
```

Missing, untrained, incompatible, or wrong-scale checkpoints raise clear errors; inference does not fall back to bicubic as an SRGAN result. Input side is limited to 768 pixels by default to manage memory. Tiled inference is not implemented; larger LR images are rejected with a useful message rather than silently resized and returned at the wrong scale.

Launch from the project root:

```powershell
streamlit run app.py
```

Alternatively, double-click `quickstart.bat` in the project root. It creates a local `.venv` if needed, installs `requirements.txt` into that environment when Streamlit is missing, and starts the app. The first run requires internet access to install dependencies.

The UI previews the LR image, bicubic baseline, and real checkpoint-based SRGAN output, reports exact output dimensions and inference time, and offers PNG download. Only valid trained checkpoint scales are selectable. User uploads are processed in memory.

## Tests and smoke checks

```powershell
python -m data.build_manifest --force
python -m pytest -q
python scripts\verify_environment.py
```

Tests include actual-style filename normalization, duplicate/unmatched/dimension mismatch handling, source-disjoint manifest splits, X4 shape and backward checks, checkpoint/inference errors, metrics, and tiny CPU training. Tests do not require or download DIV2K. A smoke subset verifies execution, not restoration quality.

## Legacy code and project layout

```text
app.py                         SRGAN Streamlit UI
configs/default.yaml           local dataset, X4, and training configuration
data/div2k_dataset.py          pairing validation, manifest, aligned crops
data/build_manifest.py         pairing report CLI
models/sr_generator.py         residual PixelShuffle generator
models/sr_discriminator.py     HR discriminator
models/srgan.py                model wrapper
training/pretrain.py           Stage A pixel-loss generator training
training/train_srgan.py        Stage B adversarial fine-tuning
training/srgan_utils.py        shared training/checkpoint utilities
inference/super_resolve.py     scale-safe SR inference and bicubic baseline
evaluation/evaluate_srgan.py   validation comparison metrics/grid
```

The prior Pix2Pix restoration modules remain in the workspace for backward compatibility and are not used by the SRGAN interface or new training commands.

## Limitations and data use

- The provided LR_unknown degradation algorithm is undocumented; only file-name and exact dimensions were verifiable here.
- There is no HR reference directory for the 100 LR-only test inputs per scale.
- X3 model support is not implemented; X2 and X4 have explicit architectures.
- Training results depend on data, configuration, and compute. No minimum PSNR/SSIM is promised.
- Respect terms applicable to the user's local DIV2K source. The project does not download or redistribute dataset images.
- SR models synthesize details and cannot guarantee ground-truth recovery. Do not present generated details as factual historical evidence.
