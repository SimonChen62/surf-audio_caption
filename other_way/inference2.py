from __future__ import annotations
import argparse

import pandas as pd
from pathlib import Path
import soundfile
import torch
import librosa
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop
from audidata.transforms import Mono

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from train1 import get_audio_encoder, get_llm_decoder, get_audio_latent
# 原来的：
# from train import get_audio_encoder, get_llm_decoder, get_audio_latent

# 修改为：
from models.qwen import Qwen, QwenConfig
from train1 import get_audio_encoder, get_audio_latent
import torch.nn.functional as F

def generate(model, input_seqs, seq_types, max_new_tokens=30, temperature=1.0, top_k=None):
    """
    自回归生成文本
    :param model: Qwen 模型
    :param input_seqs: [audio_latent, text_ids]
    :param seq_types: ["audio", "text"]
    :param max_new_tokens: 最大生成长度
    :param temperature: 温度参数
    :param top_k: top-k 采样
    :return: 生成的 token IDs
    """
    audio_latent, text_ids = input_seqs
    generated = text_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(seqs=[audio_latent, generated], seq_types=seq_types)
            logits = outputs[1][:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

    return [None, generated]

def inference(audio_path: str):
    # 参数设置
    ckpt_path = "checkpoints/train1/Cnn14_Qwen/step1=8000.pth"  # 替换为你自己的路径
    sr = 32000
    device = "cuda"
    max_length = 30
    clip_duration = 10.
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Qwen"
    start_token_id = 101  # CLS token ID
    text_vocab_size = 30522  # BERT tokenizer vocab size

    # 加载音频
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = torch.Tensor(audio[None, None, :]).to(device)

    # 加载音频编码器
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)
    audio_encoder.eval()

    # 加载 Qwen 解码器
    config = QwenConfig(
        block_size=max_length,
        audio_latent_dim=audio_latent_dim,
        vocab_size=text_vocab_size,
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    llm_decoder = Qwen(config).to(device)
    llm_decoder.load_state_dict(torch.load(ckpt_path))
    llm_decoder.eval()

    # 提取音频 embedding
    with torch.no_grad():
        audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)

    # 开始 token
    text_ids = torch.LongTensor([[start_token_id]]).to(device)

    # 生成文本
    result_strings = []
    for _ in range(1):  # 可调整 num_samples
        outputs = generate(
            model=llm_decoder,
            input_seqs=[audio_latent, text_ids],
            seq_types=["audio", "text"],
            max_new_tokens=max_length,
            temperature=1.0,
            top_k=200
        )
        sampled_text_ids = outputs[-1][0].cpu().numpy()
        strings = BertTokenizer(max_length=max_length).tokenizer.decode(
            sampled_text_ids, skip_special_tokens=True
        )
        result_strings.append(strings)

    return "\n".join(result_strings)


def get_clotho_meta(root: str, split: str) -> dict:  #可删
    r"""Load Clotho audio paths and captions."""
    if split == "train":
        meta_csv = Path(root, "clotho_captions_development.csv")
        audios_dir = Path(root, "clotho_audio_development")

    elif split == "test":
        meta_csv = Path(root, "clotho_captions_evaluation.csv")
        audios_dir = Path(root, "clotho_audio_evaluation")

    else:
        raise ValueError(split)

    meta_dict = {
        "audio_name": [],
        "audio_path": [],
        "captions": []
    }

    df = pd.read_csv(meta_csv, sep=',')

    for n in range(len(df)):
        meta_dict["audio_name"].append(df["file_name"][n])
        meta_dict["audio_path"].append(Path(audios_dir, df["file_name"][n]))
        meta_dict["captions"].append([df["caption_{}".format(i)][n] for i in range(1, 6)])

    return meta_dict


def tokens_to_string(tokens, tokenizer):
    return "".join([tokenizer.itos(token) for token in tokens])


if __name__ == "__main__":
    audio_path ="I:/资料/1/surf-2025/mini_audio_caption/clotho/clotho_audio_evaluation/01 A pug struggles to breathe 1_14_2008.wav"
    inference(audio_path)