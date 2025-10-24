# Mamba‑YOLO & vMamba on Windows

*A step‑by‑step, Windows‑first reproduction guide and starter kit.*

---

## 0) Preface

Mamba‑based networks (Mamba, vMamba) are often reproduced smoothly on Linux thanks to a mature open‑source toolchain. On Windows, however, developers frequently hit package conflicts, CUDA/driver mismatches, and extension build hurdles. This README provides a **fully guided Windows workflow** to get **Mamba‑YOLO** training and inference running with **GPU acceleration**, without compiling custom CUDA when possible. It mirrors the tutorial content you referenced and adapts it into a clean, repository‑ready document.

> **Who is this for?** Windows users who want a working environment for Mamba‑YOLO/vMamba + YOLO training with minimal friction.

---

## 1) Highlights

* ✅ **Windows‑validated** setup—commands and versions tested on Windows.
* ✅ **Pre‑packaged artifacts** (wheel/folders) to avoid compiling CUDA ops where possible.
* ✅ **MS COCO 2017 in YOLO format (optional)** for plug‑and‑play training.
* ✅ **VS Code‑first workflow** with a reproducible interpreter and run configuration.
* ✅ **Troubleshooting** for common Windows‑only issues (NumPy 2.x ABI, `selective_scan`, Triton on Windows, etc.).

---

## 2) Repository Structure (suggested)

```
Mamba-YOLO-windows/
├─ attachments/               # Prebuilt wheels and vendor folders for Windows
│  ├─ triton-2.0.0-cp310-cp310-win_amd64.whl
│  ├─ causal-conv1d/          # vendored source (tag v1.1.1)
│  └─ mamba/                  # vendored source (tag v1.1.1)
├─ scripts/
│  ├─ verify_cuda.py          # quick CUDA sanity check
│  ├─ mytest001.py            # minimal inference/test script
│  ├─ mytrain001.py           # simple training launcher (edit your YAML path)
│  └─ mbyolo_train.py         # lightly adapted original trainer
├─ ultralytics/               # framework code (with local modules)
├─ datasets/                  # indexes & samples (no big files tracked)
├─ assets/                    # screenshots, figures
└─ README.md
```

---

## 3) Prerequisites

* **Windows 10/11**, up‑to‑date NVIDIA driver.
* **Anaconda/Miniconda**.
* **CUDA 11.8 runtime** via conda (we will not install a full CUDA Toolkit system‑wide).
* **Python 3.10** recommended.

> If you are brand new to Anaconda on Windows, see: [https://blog.csdn.net/Natsuago/article/details/143081283](https://blog.csdn.net/Natsuago/article/details/143081283)

---

## 4) Environment Setup (GPU, PyTorch, CUDA)

Open **Anaconda Prompt** and create the environment:

```bat
conda create -n mamba python=3.10 -y
conda activate mamba

:: CUDA 11.8 runtime (nvidia channel)
conda install -c nvidia cudatoolkit==11.8 -y

:: Match PyTorch to CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 ^
  --index-url https://download.pytorch.org/whl/cu118

:: Helpful extras
conda install -y packaging
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```

Sanity‑check CUDA:

```python
# scripts/verify_cuda.py
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
```

Expected: `cuda available: True`.

> **Tip**: Pin **NumPy < 2.0** in Windows scientific stacks to avoid ABI issues with native extensions:
> `pip install "numpy<2" -U`

---

## 5) Get the Code

This tutorial assumes two codebases:

* **Original project**: *(add the upstream link)*
  `TODO: https://github.com/…`
* **Windows‑ready tutorial repo** (this project): *(add your repo link)*
  `TODO: https://github.com/…`

If you previously failed to build from upstream on Windows, **start from this Windows‑ready repo** which vendors key dependencies.

---

## 6) Install Windows‑friendly Dependencies

### 6.1 Triton (prebuilt wheel)

Copy the wheel from `attachments/` to a convenient path (or use absolute path), then:

```bat
pip install attachments\triton-2.0.0-cp310-cp310-win_amd64.whl
```

### 6.2 causal-conv1d (tag v1.1.1)

Option A — **Vendored folder (recommended):**

```bat
cd attachments\causal-conv1d
pip install .
```

Option B — fresh clone (if you prefer):

```bat
git clone https://github.com/…/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.1
pip install .
```

### 6.3 mamba (tag v1.1.1)

Option A — **Vendored folder (recommended):**

```bat
cd ..\mamba
pip install .
```

Option B — fresh clone:

```bat
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.1
pip install .
```

> Using the vendored sources avoids compiling extra CUDA ops on Windows.

---

## 7) VS Code Setup & Quick Test

1. Open the folder `Mamba-YOLO-windows/` in **VS Code**.
2. Select interpreter: **Python: Select Interpreter → mamba** (the conda env).
3. Run `scripts/mytest001.py`.

### If you see `AttributeError: 'str' object has no attribute 'contiguous'`

This typically comes from a call to `.contiguous()` on a Python string inside `causal_conv1d_interface.py`. Patch it so only tensors are made contiguous:

```python
# In your site-packages: causal_conv1d/causal_conv1d_interface.py
# (e.g., C:\\Users\\<YOU>\\anaconda3\\envs\\mamba\\lib\\site-packages\\causal_conv1d\\)

def _to_contig(x):
    return x.contiguous() if hasattr(x, "contiguous") else x

# Before (problematic):
# out = causal_conv1d_cuda.causal_conv1d(x.contiguous(), w.contiguous(), padding.contiguous())

# After (safe):
out = causal_conv1d_cuda.causal_conv1d(_to_contig(x), _to_contig(w), _to_contig(padding))
```

Close and reopen the terminal (or run `cls`) and try again. You should see **success** in the test.

---

## 8) Training

### 8.1 Minimal launcher

Edit your dataset YAML path in `scripts/mytrain001.py` (around line ~18) and run:

```bat
python scripts\mytrain001.py --data path\to\your_dataset.yaml --img 640 --epochs 100
```

Example dataset YAML (YOLO format):

```yaml
# datasets/pavement.yaml
path: D:/datasets/pavement
train: images/train
val: images/val
# test: images/test
nc: 6
names: ["crack", "pothole", "patch", "rutting", "manhole", "other"]
```

> If you are new to YOLO custom datasets, it helps to review a YOLO data tutorial before returning here to train with Mamba‑YOLO.

### 8.2 Original (lightly adapted) trainer

You can also run the adapted script which mirrors the upstream training logic:

```bat
python scripts\mbyolo_train.py --data datasets\pavement.yaml --img 896 --epochs 300
```

> **Note**: This repo optionally ships **MS COCO 2017 in YOLO format** (or data indexes) for a plug‑and‑play baseline. Replace `--data` accordingly.

---

## 9) Tips & Known Pitfalls (Windows)

* **NumPy 2.x ABI**: Some compiled wheels built against NumPy 1.x will fail under NumPy 2.x. Pin with `pip install "numpy<2" -U`.
* **`selective_scan` unavailable**: If you see `ImportError: selective_scan is unavailable…`, you are loading code that expects `mamba-ssm>=2.x`. Prefer the **vendored `mamba/`** in this repo, or install `mamba-ssm` and its CUDA extensions per their docs (Linux is easier). On Windows, avoid compiling if possible.
* **CUDA/driver mismatch**: Ensure your NVIDIA driver supports CUDA 11.8. Use the PyTorch wheel matching `cu118`.
* **Triton on Windows**: Use the provided wheel; upstream support for certain Triton builds on Windows is limited.
* **`AttributeError: 'str' … contiguous`**: Apply the `causal-conv1d` patch above so that only tensors call `.contiguous()`.

---

## 10) Results & Benchmarks (placeholders)

Add your own tables/figures once you finish experiments:

```
Model            | imgsz | Epochs | mAP@0.5 | Speed (ms/img, 4090) | Notes
----------------|-------|--------|---------|------------------------|------
Mamba‑YOLO‑T     | 896   | 300    |  —      |  —                     |
Mamba‑YOLO‑S     | 896   | 300    |  —      |  —                     |
```

---

## 11) Acknowledgements

* Thanks to the Zhihu author **行休** for the insightful article that inspired these Windows‑oriented instructions.
* Thanks to the authors and maintainers of **Mamba**, **vMamba**, **Ultralytics/YOLO**, and related libraries.

---

## 12) License

Specify your license here. Example: MIT License.
`TODO: choose a license`

---

## 13) Contacts & Links

* Original upstream repository: `TODO add link`
* This Windows‑ready repo: `TODO add link`
* Issues: please open a GitHub issue with the full console log and environment info (Windows version, driver version, `torch.__version__`, `torch.version.cuda`).
