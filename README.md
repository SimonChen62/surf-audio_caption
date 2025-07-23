<h1 align="center">Audio Captioning</h1>
<p align="center">Deep learning model for generating textual descriptions of audio content</p>

## 📋 Overview
This repository contains an implementation of an audio captioning system that generates natural language descriptions for audio clips. The model combines audio feature extraction with sequence-to-sequence learning to produce captions. Pre-trained models are available for immediate inference, or you can train new models using the Clotho dataset.

## ⚙️ Installation
To set up the environment, run the following commands:

```bash
# 1. Install PyTorch (Ensure CUDA is available for GPU support)
pip install torch torchvision torchaudio

# 2. Install librosa for audio processing
pip install librosa

# 3. Install pandas and pathlib for data handling
pip install pandas

# 4. Install soundfile for audio file I/O
pip install soundfile

# 5. Install matplotlib and numpy for visualization and numerical operations
pip install matplotlib numpy

# 6. Install PyQt6 for the GUI interface
pip install PyQt6

# 7. Install pyaudio for real-time audio recording (Optional)
pip install pyaudio

# 8. Install tqdm for progress bars during training
pip install tqdm

# 9. Install wandb for logging (Optional)
pip install wandb

# 10. Install audidata (Custom library)
pip install git+https://github.com/yourusername/audidata.git  # Replace with actual URL

# 11. Install panns_inference for audio tagging
pip install panns_inference  # If available on PyPI, otherwise build from source

# 12. Install transformers for BERT tokenizer
pip install transformers
