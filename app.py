"""RestoreAI DIV2K SRGAN super-resolution interface."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

from inference.super_resolve import load_sr_generator, super_resolve

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_INPUT_SIDE = 768

st.set_page_config(page_title="RestoreAI SRGAN", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_sr_generator(path: str, device_name: str, scale: int):
    return load_sr_generator(path, device_name, expected_scale=scale)


def _read_upload(upload) -> Image.Image:
    if upload.size > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 15 MB upload limit.")
    if Path(upload.name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise ValueError("Upload a JPG, JPEG, PNG, or BMP low-resolution image.")
    try:
        image = Image.open(io.BytesIO(upload.getvalue()))
        if image.width * image.height > 40_000_000:
            raise ValueError("Image dimensions exceed the 40-megapixel safety limit.")
        image.verify()
        image = Image.open(io.BytesIO(upload.getvalue()))
        return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"The uploaded image is invalid: {exc}") from exc


def _checkpoint(scale: int) -> Path:
    suffix = "" if scale == 4 else f"_X{scale}"
    return CHECKPOINT_DIR / f"srgan_generator_inference{suffix}.pth"


st.title("RestoreAI · DIV2K Super-Resolution")
st.markdown("### Increase image resolution with a trained SRGAN generator")
st.write(
    "Upload an LR image, compare bicubic interpolation with a trained residual PixelShuffle SRGAN, "
    "and download the generated high-resolution image."
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

available_scales = []
for candidate in (4, 2):
    path = _checkpoint(candidate)
    if path.is_file():
        try:
            cached_sr_generator(str(path), str(device), candidate)
            available_scales.append(candidate)
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

with st.sidebar:
    st.header("Model status")
    st.write(f"Device: **{device}**")
    if available_scales:
        st.success("Trained SRGAN checkpoint(s): " + ", ".join(f"X{scale}" for scale in available_scales))
        selected = 4 if 4 in available_scales else available_scales[0]
        selected_payload = cached_sr_generator(str(_checkpoint(selected)), str(device), selected)[2]
        if selected_payload.get("training_data_scope") == "subset":
            st.warning(
                f"X{selected} checkpoint was trained on a limited subset "
                f"({selected_payload.get('training_pair_count')} pairs); it is a pipeline/demo checkpoint, "
                "not the completed full-data model."
            )
    else:
        st.warning("No valid trained SRGAN checkpoint found. Complete generator pretraining and SRGAN training first.")
    st.caption("X2 and X4 use scale-specific PixelShuffle models. X3 is not implemented.")
    st.caption(f"Maximum LR input side: {MAX_INPUT_SIDE}px. Tiled inference is not implemented.")

tabs = st.tabs(["Super-resolve", "Model & Evaluation", "About"])

with tabs[0]:
    st.subheader("Super-resolve an image")
    if available_scales:
        scale = st.selectbox(
            "Output scale",
            options=available_scales,
            index=available_scales.index(4) if 4 in available_scales else 0,
            format_func=lambda value: f"X{value}",
        )
    else:
        scale = st.selectbox("Output scale", options=[4], format_func=lambda value: "X4 (checkpoint required)")
    upload = st.file_uploader(
        f"Choose an LR image (JPG, PNG, BMP; maximum 15 MB)",
        type=["jpg", "jpeg", "png", "bmp"],
        max_upload_size=15,
    )
    source = None
    if upload is not None:
        try:
            source = _read_upload(upload)
            st.caption(f"Input: {source.width} × {source.height} pixels")
            st.image(source, caption="Low-resolution input", use_container_width=True)
            if max(source.size) > MAX_INPUT_SIDE:
                st.warning(
                    f"Input exceeds configured {MAX_INPUT_SIDE}px maximum. Resize before upload or adjust "
                    "MAX_INPUT_SIDE in app.py if memory permits."
                )
        except ValueError as exc:
            st.error(str(exc))
    checkpoint = _checkpoint(scale)
    model_ready = scale in available_scales and checkpoint.is_file()
    if not checkpoint.is_file():
        st.info(f"Trained X{scale} checkpoint not found. Train this scale before inference.")
    if st.button(
        f"Run X{scale} super-resolution",
        type="primary",
        disabled=source is None or not model_ready or max(source.size) > MAX_INPUT_SIDE,
    ):
        try:
            with st.spinner("Running SRGAN inference…"):
                result = super_resolve(source, checkpoint, expected_scale=scale, device=device)
            columns = st.columns(3)
            columns[0].image(source, caption=f"LR input · {source.width}×{source.height}", use_container_width=True)
            columns[1].image(
                result.bicubic,
                caption=f"Bicubic baseline · {result.bicubic.width}×{result.bicubic.height}",
                use_container_width=True,
            )
            columns[2].image(
                result.image,
                caption=f"SRGAN output · {result.image.width}×{result.image.height}",
                use_container_width=True,
            )
            st.caption(f"SRGAN inference: {result.elapsed_seconds:.3f}s on {result.device}")
            st.subheader("Zoomed detail (center crop, 1:1 pixels)")
            crop = min(256, result.image.width, result.image.height)
            left = (result.image.width - crop) // 2
            top = (result.image.height - crop) // 2
            box = (left, top, left + crop, top + crop)
            zoom = st.columns(2)
            zoom[0].image(result.bicubic.crop(box), caption="Bicubic crop")
            zoom[1].image(result.image.crop(box), caption="SRGAN crop")
            buffer = io.BytesIO()
            result.image.save(buffer, format="PNG")
            st.download_button(
                "Download SRGAN image (PNG)",
                buffer.getvalue(),
                file_name=f"restoreai_x{scale}.png",
                mime="image/png",
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"Super-resolution failed: {exc}")

with tabs[1]:
    st.header("SRGAN architecture and validation")
    st.markdown(
        """
        - **Generator:** convolutional feature extraction, residual blocks, long skip connection, and one (X2) or two (X4) learned PixelShuffle upsampling blocks.
        - **Discriminator:** progressive convolutional HR-image classifier. Its output is a logit trained with `BCEWithLogitsLoss`.
        - **Training:** generator L1 pretraining first, then adversarial fine-tuning with reconstruction + weighted adversarial loss. Optional VGG19 content loss requires explicit local pretrained weights.
        - **Validation:** compares bicubic, pretraining generator, and final SRGAN on the same deterministic center crops.
        """
    )
    report_path = ROOT / "outputs" / "evaluation" / "srgan_validation.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Validation pairs: {report['pair_count']} · scale {report['scale']}. "
            "These are validation metrics, not test-set results."
        )
        st.json(report)
        grid_path = report_path.parent / "srgan_validation_grid.png"
        if grid_path.is_file():
            st.image(str(grid_path), caption="Bicubic | pretrained generator | SRGAN | HR target")
    else:
        st.info("No SRGAN validation report found. Train both stages, then run `python -m evaluation.evaluate_srgan`.")
    log_path = ROOT / "logs" / "srgan" / "srgan_training.csv"
    if log_path.is_file():
        with log_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            figure, axis = plt.subplots(figsize=(9, 3.5))
            for column in ("generator_loss", "discriminator_loss", "reconstruction_loss", "adversarial_loss"):
                axis.plot(
                    [int(row["global_step"]) for row in rows if row[column]],
                    [float(row[column]) for row in rows if row[column]],
                    label=column.replace("_", " "),
                )
            axis.set_xlabel("Generator update")
            axis.set_ylabel("Loss")
            axis.legend(ncol=2)
            axis.grid(alpha=0.2)
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)

with tabs[2]:
    st.header("Dataset and limitations")
    st.write(
        "This project uses local DIV2K HR images and the matching LR_unknown images. The loader matches normalized "
        "filenames (for example `0001.png` ↔ `0001x4.png`), verifies exact scale dimensions, and creates a fixed "
        "90/10 source-image train/validation split. LR_unknown's degradation method is not documented, so quantitative "
        "scores assume the corresponding HR/LR images remain spatially aligned."
    )
    st.warning(
        "The local archive includes 100 LR test inputs per scale but no test HR references. "
        "Validation scores must not be described as test scores. Upsampling creates plausible detail; it cannot "
        "recover guaranteed ground truth."
    )
    st.caption("Uploads are processed in memory and are not added to the training dataset.")
