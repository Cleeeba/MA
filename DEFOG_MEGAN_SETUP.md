# Environment Setup: DeFoG + MEGAN

Sets up a shared Python 3.11 environment for
[DeFoG](https://github.com/the16thpythonist/DeFoG) (Discrete Flow Matching for Graph Generation)
and [MEGAN / graph_attention_student](https://github.com/the16thpythonist/graph_attention_student)
(Multi-Explanation Graph Attention Network) using `uv` and a standard venv.

## Prerequisites

- **Python 3.11** (via pyenv, system package, etc.)
- **[uv](https://docs.astral.sh/uv/)**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **CUDA 12.1** compatible GPU + drivers
- **System libraries** (Ubuntu/Debian):
  ```bash
  sudo apt install libcairo2-dev libgirepository-2.0-dev libgmp-dev libmpfr-dev libmpc-dev
  ```
- DeFoG repository cloned locally (referred to as `./DeFoG` below)

## Version Adjustments from Original Conda Environment

The following packages had to be adjusted from the original `environment.yaml`:

| Package | Original | Adjusted | Reason |
|---------|----------|----------|--------|
| `numpy` | 2.3.1 | **1.26.4** | `graph_attention_student` requires `numpy<2.0.0` |
| `mpmath` | 1.4.1 | **1.3.0** | `sympy==1.14.0` requires `mpmath<1.4` |
| `python-tzdata` | 2025.3 | **removed** | conda-only package name; functionality covered by `tzdata` |
| `graph-tool` | 2.97 | **omitted** | conda-only (not on PyPI). Only needed for SBM dataset evaluation metrics — all molecular datasets and training/sampling work without it. |

Additionally, the MEGAN install (step 3) pulls in its own dependencies (`pycomex`, `visual_graph_datasets`,
`cairosvg`, `polars`, `weasyprint`, etc.) which may adjust some base versions. In testing, `psutil`,
`decorator`, `tqdm`, and `torch-geometric` were adjusted. This is expected and does not break anything.

## 1. Create virtual environment

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

## 2. Install base environment

This installs all packages from the original conda environment (both conda and pip sections).
The `--index-strategy unsafe-best-match` flag is required because PyTorch's package index
overlaps with PyPI for some packages (e.g. pillow, numpy).

```bash
uv pip install \
    "annotated-types==0.7.0" \
    "cffi==2.0.0" \
    "charset-normalizer==3.4.7" \
    "contourpy==1.3.3" \
    "cycler==0.12.1" \
    "filelock==3.25.2" \
    "fonttools==4.62.0" \
    "freetype-py==2.3.0" \
    "fsspec==2026.3.0" \
    "gmpy2==2.3.0" \
    "greenlet==3.3.2" \
    "jinja2==3.1.6" \
    "kiwisolver==1.5.0" \
    "markupsafe==3.0.3" \
    "mpmath==1.3.0" \
    "munkres==1.1.4" \
    "packaging==26.0" \
    "pillow==12.2.0" \
    "platformdirs==4.9.4" \
    "psutil==7.2.2" \
    "pycairo==1.29.0" \
    "pycparser==2.22" \
    "pydantic==2.12.5" \
    "pydantic-core==2.41.5" \
    "pyparsing==3.3.2" \
    "python-dateutil==2.9.0.post0" \
    "pytz==2026.1.post1" \
    "pyyaml==6.0.3" \
    "rdkit==2025.03.3" \
    "reportlab==4.4.10" \
    "rlpycairo==0.4.0" \
    "six==1.17.0" \
    "sqlalchemy==2.0.48" \
    "sympy==1.14.0" \
    "typing-extensions==4.15.0" \
    "typing-inspection==0.4.2" \
    "unicodedata2==17.0.1" \
    "zstandard==0.25.0" \
    "wheel==0.46.3" \
    "aiohappyeyeballs==2.6.1" \
    "aiohttp==3.13.5" \
    "aiosignal==1.4.0" \
    "antlr4-python3-runtime==4.9.3" \
    "asttokens==3.0.1" \
    "attrs==26.1.0" \
    "black==24.3.0" \
    "blessings==1.7" \
    "certifi==2026.2.25" \
    "click==8.3.1" \
    "comm==0.2.3" \
    "debugpy==1.8.20" \
    "decorator==5.2.1" \
    "executing==2.2.1" \
    "frozenlist==1.8.0" \
    "gitdb==4.0.12" \
    "gitpython==3.1.46" \
    "gpustat==0.6.0" \
    "hydra-core==1.3.2" \
    "idna==3.11" \
    "imageio==2.31.1" \
    "ipykernel==6.29.5" \
    "ipython==9.10.1" \
    "ipython-pygments-lexers==1.1.1" \
    "jedi==0.19.2" \
    "joblib==1.5.3" \
    "jupyter-client==8.8.0" \
    "jupyter-core==5.9.1" \
    "lightning-utilities==0.15.3" \
    "matplotlib==3.10.3" \
    "matplotlib-inline==0.2.1" \
    "multidict==6.7.1" \
    "mypy-extensions==1.1.0" \
    "nest-asyncio==1.6.0" \
    "networkx==3.5" \
    "numpy==1.26.4" \
    "nvidia-cublas-cu12==12.1.3.1" \
    "nvidia-cuda-cupti-cu12==12.1.105" \
    "nvidia-cuda-nvrtc-cu12==12.1.105" \
    "nvidia-cuda-runtime-cu12==12.1.105" \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cufft-cu12==11.0.2.54" \
    "nvidia-curand-cu12==10.3.2.106" \
    "nvidia-cusolver-cu12==11.4.5.107" \
    "nvidia-cusparse-cu12==12.1.0.106" \
    "nvidia-ml-py3==7.352.0" \
    "nvidia-nccl-cu12==2.20.5" \
    "nvidia-nvjitlink-cu12==12.9.86" \
    "nvidia-nvtx-cu12==12.1.105" \
    "omegaconf==2.3.0" \
    "overrides==7.3.1" \
    "pandas==2.3.0" \
    "parso==0.8.6" \
    "pathspec==1.0.4" \
    "pexpect==4.9.0" \
    "prompt-toolkit==3.0.52" \
    "propcache==0.4.1" \
    "protobuf==6.33.6" \
    "ptyprocess==0.7.0" \
    "pure-eval==0.2.3" \
    "pygments==2.20.0" \
    "pygsp==0.5.1" \
    "pytorch-lightning==2.0.4" \
    "pyzmq==27.1.0" \
    "requests==2.33.1" \
    "scikit-learn==1.8.0" \
    "scipy==1.16.0" \
    "seaborn==0.13.2" \
    "sentry-sdk==2.57.0" \
    "setproctitle==1.3.7" \
    "setuptools==68.0.0" \
    "smmap==5.0.3" \
    "stack-data==0.6.3" \
    "threadpoolctl==3.6.0" \
    "torch==2.4.0+cu121" \
    "torch-geometric==2.3.1" \
    "torchmetrics==0.11.4" \
    "tornado==6.5.5" \
    "tqdm==4.65.0" \
    "traitlets==5.14.3" \
    "triton==3.0.0" \
    "tzdata==2026.1" \
    "urllib3==2.6.3" \
    "wandb==0.20.1" \
    "wcwidth==0.6.0" \
    "yarl==1.23.0" \
    "pyarrow" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 
    
```

`pyarrow` is added (not in the original env) because it is needed by `polars` for
pandas DataFrame interop used in MEGAN.

## 3. Install MEGAN (graph_attention_student)

```bash
git clone https://github.com/the16thpythonist/graph_attention_student.git
uv pip install -e ./graph_attention_student \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match
```

This pulls in MEGAN's additional dependencies (`pycomex`, `visual_graph_datasets`,
`cairosvg`, `nltk`, `polars`, `weasyprint`, `lightning`, etc.) and will upgrade
`torch-geometric` from 2.3.1 to a compatible version (tested with 2.7.0).

## 4. Verify

```bash
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy; print(f'NumPy {numpy.__version__}')
import graph_attention_student; print('MEGAN OK')
"
```

Expected:
```
PyTorch 2.4.0+cu121, CUDA: True
NumPy 1.26.4
MEGAN OK
```

## Notes

### numpy < 2.0

This is the most significant constraint. `graph_attention_student` requires `numpy<2.0.0`,
which forces the environment to numpy 1.26.x instead of the original 2.3.1. Both DeFoG
and MEGAN work correctly with this version. If numpy 2.x appears in the environment after
setup, something went wrong.

### graph-tool

The original conda environment included `graph-tool`, a C++ graph analysis library that
is **not available on PyPI**. In DeFoG, it is only used in `src/analysis/spectre_utils.py`
for evaluating generated Stochastic Block Model (SBM) graphs via `gt.minimize_blockmodel_dl()`.
It is **not** used during training, sampling, or evaluation of any molecular dataset
(QM9, MOSES, ZINC, Guacamol).

If SBM evaluation is needed in the future, graph-tool must be installed via conda or
compiled from source.
