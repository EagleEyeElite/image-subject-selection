# pytorch/pytorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size


# Download the AI SAM model file
RUN wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -P /usr/models/
RUN wget "https://download.pytorch.org/models/resnet50-0676ba61.pth" -P /usr/models/

# Install Jupyter
# Copy the requirements.txt file into the container
COPY requirements.txt /usr/src/app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Configure huggingface-cli to use the token non-interactively
ARG HF_TOKEN
RUN python -c "from huggingface_hub import HfFolder; HfFolder.save_token('${HF_TOKEN}')"

# Copy the sample notebook into the container
COPY * /usr/src/app/

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook
# Note: '--allow-root' is often needed in Docker containers
# '--ip=0.0.0.0' makes the notebook accessible from outside the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='token'"]
