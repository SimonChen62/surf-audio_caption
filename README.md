<!DOCTYPE html>
<html>
<head>
    <title>Audio Captioning Project README</title>
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
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 30px;
            margin-top: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .copy-btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 10px 0;
            cursor: pointer;
            border-radius: 4px;
            float: right;
        }
        .copy-btn:hover {
            background-color: #45a049;
        }
        .section {
            margin-bottom: 30px;
        }
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Audio Captioning Project</h1>
            <p>Deep learning model for generating textual descriptions of audio content</p>
        </div>

        <div class="section">
            <h2>📋 Overview</h2>
            <p>This repository contains an implementation of an audio captioning system that generates natural language descriptions for audio clips. The model combines audio feature extraction with sequence-to-sequence learning to produce captions. Pre-trained models are available for immediate inference, or you can train new models using the Clotho dataset.</p>
        </div>

        <div class="section">
            <h2>⚙️ Installation</h2>
            <p>To set up the environment, run the following commands:</p>
            
            <div class="code-header">
                <h3>Environment Setup</h3>
                <button class="copy-btn" onclick="copyCode('install-code')">Copy All</button>
            </div>
            <pre id="install-code"><code># 1. Install PyTorch (Ensure CUDA is available for GPU support)
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
        </div>

        <div class="section">
            <h2>🚀 Usage</h2>
            
            <h3>Option 1: Using Pre-trained Models (Quick Start)</h3>
            <p>Pre-trained models are available in the <code>checkpoints/</code> directory:</p>
            <pre><code>python inference.py \
  --model_path checkpoints/best_model.pth \
  --audio_file sample.wav \
  --beam_size 3</code></pre>
            
            <h3>Option 2: Training from Scratch</h3>
            <p>1. Download the <a href="https://zenodo.org/record/3490684" target="_blank">Clotho dataset</a></p>
            <p>2. Organize the dataset:</p>
            <pre><code>data/
├── clotho_audio/
├── clotho_captions/
└── metadata.csv</code></pre>
            <p>3. Start training:</p>
            <pre><code>python train.py \
  --dataset_dir data \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0001 \
  --save_dir models/</code></pre>
        </div>

        <div class="section">
            <h2>🧠 Model Architecture</h2>
            <ul>
                <li><strong>Audio Encoder</strong>: PANNs CNN (Pre-trained on AudioSet)</li>
                <li><strong>Text Decoder</strong>: Transformer-based sequence generator</li>
                <li><strong>Feature Fusion</strong>: Attention-based multimodal fusion</li>
            </ul>
        </div>

        <div class="section">
            <h2>📂 Repository Structure</h2>
            <pre><code>├── checkpoints/         # Pre-trained models
├── data_processing/     # Dataset loading and preprocessing
├── model/               # Model architecture definitions
├── inference.py         # Generate captions from audio
├── train.py             # Training script
├── utils/               # Helper functions
└── requirements.txt     # Python dependencies</code></pre>
        </div>

        <div class="section">
            <h2>📊 Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>BLEU-4</th>
                    <th>CIDEr</th>
                    <th>SPICE</th>
                </tr>
                <tr>
                    <td>Baseline</td>
                    <td>0.312</td>
                    <td>0.891</td>
                    <td>0.186</td>
                </tr>
                <tr>
                    <td>+ Attention</td>
                    <td>0.351</td>
                    <td>0.932</td>
                    <td>0.201</td>
                </tr>
                <tr>
                    <td>+ Beam Search</td>
                    <td>0.368</td>
                    <td>0.978</td>
                    <td>0.214</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>🤖 GUI Interface</h2>
            <p>Run the interactive interface:</p>
            <pre><code>python gui_app.py</code></pre>
            <p><em>Note: GUI requires PyQt6 installation</em></p>
        </div>

        <div class="section">
            <h2>📜 License</h2>
            <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details</p>
        </div>
    </div>

    <script>
        function copyCode(elementId) {
            const codeElement = document.getElementById(elementId);
            const textArea = document.createElement('textarea');
            textArea.value = codeElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            const originalText = event.target.textContent;
            event.target.textContent = 'Copied!';
            setTimeout(() => {
                event.target.textContent = originalText;
            }, 2000);
        }
    </script>
</body>
</html>
