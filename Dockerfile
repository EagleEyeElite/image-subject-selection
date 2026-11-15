# CUDA runtime libs come bundled with the torch wheels, so a slim Python base is enough.
# Run with GPU access (e.g. `docker compose up`, which requests the nvidia device).
FROM python:3.12-slim-bookworm

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CHECKPOINT_DIR=/usr/models \
    PYTHONUNBUFFERED=1

# git: segment-anything is installed from GitHub; libglib2.0-0: needed by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.16

WORKDIR /usr/src/app

# Dependencies first (cached layer). The venv lives in /opt/venv so the
# docker-compose bind mount of the repo onto /usr/src/app doesn't hide it.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY tools/ tools/
RUN uv sync --frozen --no-install-project

# Model checkpoints (SAM vit_b + ResNet50)
COPY scripts/download_checkpoints.sh scripts/
RUN scripts/download_checkpoints.sh /usr/models

COPY . .
RUN uv sync --frozen  # installs saliency_sam (editable, so the compose bind mount is live)

EXPOSE 8888

# Jupyter reads the access token from $JUPYTER_TOKEN.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
