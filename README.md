<!DOCTYPE html>
<html>
<head>
    <title>Audio Captioning Project</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            line-height: 1.4;
        }
        .note {
            background-color: #e7f5ff;
            border-left: 4px solid #4dabf7;
            padding: 12px;
            margin: 15px 0;
            border-radius: 3px;
        }
        .warning {
            background-color: #fff3bf;
            border-left: 4px solid #ffd43b;
            padding: 12px;
            margin: 15px 0;
            border-radius: 3px;
        }
    </style>
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
<p>To set up the environment (minimum requirements for running the interactive interface):</p>

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
<pre><code>python inference.py \
  --model_path checkpoints/model_bert_base.pth \
  --audio_file sample.wav \
  --beam_size 3</code></pre>
<p>Available pre-trained configurations:</p>
<ul>
    <li><code>model_bert_base.pth</code> - Base model with BERT tokenizer</li>
    <li><code>model_large_attn.pth</code> - Larger model with attention mechanism</li>
    <li><code>model_beam5.pth</code> - Optimized for beam search (size=5)</li>
</ul>

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

<h2>🧠 Model Architecture</h2>
<ul>
    <li><strong>Audio Encoder</strong>: PANNs CNN (Pre-trained on AudioSet)</li>
    <li><strong>Text Decoder</strong>: Transformer-based sequence generator</li>
    <li><strong>Feature Fusion</strong>: Attention-based multimodal fusion</li>
</ul>

<h2>📂 Repository Structure</h2>
<pre><code>├── checkpoints/         # Pre-trained models (multiple configurations)
├── data_processing/     # Dataset loading and preprocessing
├── model/               # Model architecture definitions
├── inference.py         # Generate captions from audio
├── train.py             # Training script (use with Clotho dataset)
├── utils/               # Helper functions
└── requirements.txt     # Python dependencies</code></pre>

<h2>📜 License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details</p>

</body>
</html>
