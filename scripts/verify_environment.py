"""Report Python, PyTorch, accelerator, and required dependency availability."""

from __future__ import annotations

import importlib
import sys

DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "Pillow": "PIL",
    "OpenCV": "cv2",
    "Streamlit": "streamlit",
    "Matplotlib": "matplotlib",
    "scikit-image": "skimage",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "pytest": "pytest",
}


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {device}")
    except ImportError as exc:
        print(f"PyTorch: MISSING ({exc})")
    print("Dependencies:")
    for label, module in DEPENDENCIES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "available")
            print(f"  {label}: {version}")
        except ImportError:
            print(f"  {label}: MISSING")


if __name__ == "__main__":
    main()
