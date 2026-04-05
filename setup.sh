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

# Download API keys if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Downloading API configuration..."
    curl -sL "https://gist.githubusercontent.com/somersaudio/ad9423ac7f83b3035850afcbd0a2fc9f/raw/.env" -o .env 2>/dev/null
    if [ -f ".env" ] && [ -s ".env" ]; then
        echo "API keys configured!"
    else
        echo "Could not download API keys — you'll need to enter them manually in Settings"
    fi
fi

# Main venv
echo "Setting up main virtual environment..."
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

deactivate

# RVC venv (needs Python 3.10 for fairseq compatibility)
echo ""
echo "Setting up RVC environment..."
if command -v python3.10 &> /dev/null; then
    python3.10 -m venv venv_rvc
    source venv_rvc/bin/activate

    pip install "pip<24.1" -q
    pip install rvc-python --no-deps -q
    pip install torch torchaudio soundfile av "faiss-cpu==1.7.3" pyworld torchcrepe \
        scipy "numpy<=1.23.5" praat-parselmouth "omegaconf==2.0.6" -q
    pip install "fairseq @ git+https://github.com/facebookresearch/fairseq.git@v0.12.2" --no-deps -q
    pip install bitarray regex sacrebleu hydra-core==1.0.7 -q

    deactivate
    echo "RVC environment ready!"
else
    echo "Python 3.10 not found — RVC inference will not be available"
    echo "Install with: brew install python@3.10"
fi

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

# Create desktop app launcher
echo ""
echo "Creating desktop launcher..."
osacompile -o "$HOME/Desktop/SomerSVC.app" -e "
do shell script \"cd \\\"$(pwd)\\\" && source venv/bin/activate && python main.py > /tmp/somersvc.log 2>&1 &\"
" 2>/dev/null

if [ -f "assets/icon.png" ]; then
    mkdir -p /tmp/AppIcon.iconset
    sips -z 16 16 assets/icon.png --out /tmp/AppIcon.iconset/icon_16x16.png 2>/dev/null
    sips -z 32 32 assets/icon.png --out /tmp/AppIcon.iconset/icon_16x16@2x.png 2>/dev/null
    sips -z 32 32 assets/icon.png --out /tmp/AppIcon.iconset/icon_32x32.png 2>/dev/null
    sips -z 64 64 assets/icon.png --out /tmp/AppIcon.iconset/icon_32x32@2x.png 2>/dev/null
    sips -z 128 128 assets/icon.png --out /tmp/AppIcon.iconset/icon_128x128.png 2>/dev/null
    sips -z 256 256 assets/icon.png --out /tmp/AppIcon.iconset/icon_128x128@2x.png 2>/dev/null
    sips -z 256 256 assets/icon.png --out /tmp/AppIcon.iconset/icon_256x256.png 2>/dev/null
    sips -z 512 512 assets/icon.png --out /tmp/AppIcon.iconset/icon_256x256@2x.png 2>/dev/null
    sips -z 512 512 assets/icon.png --out /tmp/AppIcon.iconset/icon_512x512.png 2>/dev/null
    iconutil -c icns /tmp/AppIcon.iconset -o "$HOME/Desktop/SomerSVC.app/Contents/Resources/applet.icns" 2>/dev/null
    touch "$HOME/Desktop/SomerSVC.app"
fi
echo "SomerSVC app created on Desktop!"

echo ""
echo "================================"
echo "  Setup complete!"
echo "================================"
echo ""
echo "Double-click SomerSVC on your Desktop to launch!"
echo ""
