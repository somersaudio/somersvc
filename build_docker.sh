#!/bin/bash
# Build and push the SVC GUI training Docker image
# This script runs ON a RunPod pod

set -e

DOCKER_USER="somersaudio"
IMAGE_NAME="svc-gui-training"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:latest"

echo "=== Building SVC GUI Training Docker Image ==="
echo "Image: ${FULL_IMAGE}"

# Create Dockerfile
cat > /tmp/Dockerfile << 'DOCKERFILE'
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install so-vits-svc-fork with pinned deps (no torch upgrade)
RUN pip install --no-cache-dir so-vits-svc-fork --no-deps && \
    pip install --no-cache-dir \
    cm-time click fastapi librosa 'lightning<2.5' matplotlib \
    pebble praat-parselmouth psutil pysimplegui-4-foss pyworld \
    requests 'rich==13.9.4' scipy sounddevice soundfile tensorboard \
    tensorboardx torchcrepe tqdm tqdm-joblib 'transformers<4.46' \
    'numpy<2' 'huggingface-hub<1'

# Pre-download the HuBERT model so it's cached
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('lengyue233/content-vec-best', 'pytorch_model.bin')"

# Verify
RUN python3 -c "import torch; print('PyTorch:', torch.__version__)" && \
    svc --help > /dev/null

WORKDIR /workspace
DOCKERFILE

# Build
echo "Building image (this takes ~5-10 minutes)..."
docker build -t ${FULL_IMAGE} -f /tmp/Dockerfile /tmp/

# Login and push
echo "Pushing to Docker Hub..."
docker push ${FULL_IMAGE}

echo "=== DONE ==="
echo "Image pushed: ${FULL_IMAGE}"
echo "Update your app to use this image name in runpod_client.py"
