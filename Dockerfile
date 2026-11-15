# CUDA runtime libs come bundled with the torch wheels (CUDA 13), so a slim
# Python base is enough; the host needs an NVIDIA driver >= 580.
# Run with GPU access (e.g. `docker compose up`, which requests the nvidia device).
FROM python:3.14.10-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CHECKPOINT_DIR=/usr/models \
    PYTHONUNBUFFERED=1

# git: segment-anything is installed from GitHub; curl: checkpoint download;
# libglib2.0-0t64: needed by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates libglib2.0-0t64 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Dependencies first (cached layer). The venv lives in /opt/venv so the
# docker-compose bind mount of the repo onto /usr/src/app doesn't hide it.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY tools/ tools/
RUN uv sync --frozen --no-install-project

# Model checkpoints (SAM ViT-B + ResNet50)
COPY scripts/download_checkpoints.sh scripts/
RUN scripts/download_checkpoints.sh /usr/models

COPY . .
# installs image_subject_selection (editable, so the compose bind mount is live)
RUN uv sync --frozen

EXPOSE 8888

# Jupyter reads the access token from $JUPYTER_TOKEN.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
