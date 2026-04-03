# Custom Docker image for SVC GUI training
# Build: docker build -t svc-gui-training .
# Push: docker tag svc-gui-training <your-dockerhub>/svc-gui-training && docker push <your-dockerhub>/svc-gui-training
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
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" && \
    svc --help > /dev/null

WORKDIR /workspace
