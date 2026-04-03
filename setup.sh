#!/bin/bash
# SVC GUI Setup Script
# Run this once to set up the app on a new machine

echo "================================"
echo "  SomerSVC - Setup"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Install it from python.org"
    exit 1
fi

echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt -q
pip install so-vits-svc-fork -q

# Patch torchaudio for local inference (torchcodec fix)
echo "Patching torchaudio for local compatibility..."
python3 -c "
import torchaudio, pathlib
p = pathlib.Path(torchaudio.__path__[0]) / '_torchcodec.py'
p.write_text('''import soundfile as sf
import torch

def load_with_torchcodec(uri, *args, **kwargs):
    audio, sr = sf.read(str(uri))
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    return torch.tensor(audio, dtype=torch.float32), sr

def save_with_torchcodec(uri, src, sample_rate, *args, **kwargs):
    if src.ndim > 1:
        src = src.T
    sf.write(str(uri), src.cpu().numpy(), sample_rate)
''')
print('torchaudio patched successfully')
"

# Generate SSH key if needed
echo ""
if [ -f "$HOME/.ssh/id_rsa" ]; then
    echo "SSH key already exists at ~/.ssh/id_rsa"
else
    echo "Generating SSH key for RunPod..."
    ssh-keygen -t rsa -b 4096 -f "$HOME/.ssh/id_rsa" -N "" -C "svc-gui"
    echo ""
    echo "SSH key generated!"
fi

echo ""
echo "================================"
echo "  Your SSH Public Key:"
echo "================================"
echo ""
cat "$HOME/.ssh/id_rsa.pub"
echo ""
echo "================================"
echo ""
echo "NEXT STEPS:"
echo "1. Copy the SSH public key above"
echo "2. Go to runpod.io > Settings > SSH Keys"
echo "3. Paste the key and save"
echo "4. Launch the app with:"
echo ""
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
