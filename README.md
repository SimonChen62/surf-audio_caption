<!DOCTYPE html>
<html>

</head>
<body>

<h1 align="center">Audio Captioning</h1>
<p align="center">Deep learning model for generating textual descriptions of audio content</p>

<h2>📋 Overview</h2>
<p>This repository contains an implementation of an audio captioning system that generates natural language descriptions for audio clips. The model combines audio feature extraction with sequence-to-sequence learning to produce captions.</p>

<div class="note">
    <p><strong>🚀 Quick Start:</strong> Pre-trained models are available in the <code>checkpoints/</code> directory with multiple parameter configurations for immediate inference.</p>
</div>

<div class="warning">
    <p><strong>⚠️ Full Training Note:</strong> If you want to train from scratch, you'll need to manually download the Clotho dataset and use <code>train.py</code>. Otherwise, you can directly use our pre-trained models.</p>
</div>

<h2>⚙️ Installation</h2>
<p>To set up the environment (minimum requirements for running the interactive interface) You can directly download the model parameter file (.pth), and use it directly with inference1 and the UI interface.:</p>

<pre><code># 1. Install PyTorch (Ensure CUDA is available for GPU support)
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
pip install transformers</code></pre>

<h2>🚀 Usage Options</h2>

<h3>Option 1: Using Pre-trained Models (Recommended)</h3>
<p>Use our trained models from the <code>checkpoints/</code> directory:</p>


<h3>Option 2: Training from Scratch</h3>
<p>1. Download and prepare the <a href="https://zenodo.org/record/3490684" target="_blank">Clotho dataset</a></p>
<p>2. Organize your dataset structure:</p>
<pre><code>data/
├── clotho_audio/       # Place audio files here
├── clotho_captions/    # Place caption files here
└── metadata.csv        # Dataset metadata file</code></pre>
<p>3. Run the training script:</p>
<pre><code>python train.py \
  --dataset_dir data \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0001 \
  --save_dir models/</code></pre>




</body>
</html>
