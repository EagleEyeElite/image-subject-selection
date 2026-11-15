# Automatic Image Segmentation by Guiding SAM with Saliency Maps

This project proposes an automatic image segmentation by combining Saliency Maps with the Segment Anything Model (SAM). Utilizing a pretrained classification model, one can generate predictions and saliency maps using existing XAI methods, which are then employed to guide SAM segmentation.

<!-- Tasks:
- Literature Review: Understand and summarize existing work in XAI and applications of SAM.
- Incorporate saliency maps and predictions to guide SAM.
- Evaluate the approach based on suitable metrics.
- Integrate the approach into Label Studio.
- Write a scientific report on your experimental results.  -->

## Setup (uv)

The project uses [uv](https://docs.astral.sh/uv/); no conda needed. Python 3.12 and all
dependencies (incl. CUDA-enabled PyTorch) are pinned in `pyproject.toml` / `uv.lock`.

```bash
uv sync                               # create .venv with all dependencies
./scripts/download_checkpoints.sh     # SAM vit_b + ResNet50 weights -> checkpoints/
uv run jupyter lab                    # or: uv run python Evaluator.py
```

To use the environment from an existing JupyterHub/Jupyter, register it as a kernel once:

```bash
uv run python -m ipykernel install --user --name automatic-image-segmentation \
    --display-name "Python (automatic-image-segmentation)"
```

**Thread limits:** the local helper `tools/thread-limits` (installed automatically by `uv sync`)
caps `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/... at 32 whenever the venv's Python starts.
Without it NumPy's OpenBLAS crashes on machines with >128 cores (e.g. this JupyterHub).
Set the variables yourself to override.

Add packages with `uv add <pkg>` (or `uv add --dev <pkg>` for tooling).
Gated Hugging Face datasets (e.g. `imagenet-1k`) need `HF_TOKEN` set or `uv run huggingface-cli login`.

## Docker

```bash
cp .env.example .env    # set JUPYTER_TOKEN / HF_TOKEN
docker compose up --build
```

Jupyter Lab is served on port 8888 with GPU access; checkpoints are baked into `/usr/models`.

## Evaluator Pipeline

The evaluator pipeline is a Python script that evaluates the performance of the proposed method on a dataset. The pipeline consists of the following steps:

- Load the dataset
- Load the pretrained classification model
- Generate predictions and saliency maps
- Apply SAM segmentation
- Evaluate the segmentation performance

## Handlers

The handlers are Python scripts that define the logic for certain parts of the pipeline. The handlers are:

- `dataset_handler.py`: Contains functions to load the dataset.
- `model_handler.py`: Contains functions to load the pretrained classification model.
- `heatmap_handler.py`: Contains functions to generate saliency maps.
- `prompt_handler.py`: Contains functions to guide SAM segmentation.
- `segment_eval_handler.py`: Contains functions to evaluate the segmentation performance.

These abstract handlers can be implemented in different ways depending on the specific requirements of the project.
