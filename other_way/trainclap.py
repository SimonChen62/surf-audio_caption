import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop
from audidata.samplers import InfiniteSampler, PseudoRandomSampler
from audidata.transforms import Mono
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import soundfile as sf
import librosa
import tempfile

# 禁用 wandb
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from models.llama import Llama, LlamaConfig

# 导入 CLAP
from msclap import CLAP  


def train(args):
    # 基础参数
    sr = 32000               # 音频采样率
    batch_size = 4           
    num_workers = 4          
    pin_memory = True        
    learning_rate = 1e-4     
    test_every_n_steps = 200 
    save_every_n_steps = 2000
    training_steps = 10000   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 30          
    clip_duration = 10.      
    audio_encoder_name = "CLAP"  # 固定使用 CLAP
    llm_decoder_name = "Llama"   # 固定使用 Llama
    filename = Path(__file__).stem

    # 数据集路径
    root = "I:/资料/1/surf-2025/mini_audio_caption/clotho"

    # 模型命名与保存目录
    model_name = f"{audio_encoder_name}_{llm_decoder_name}"
    ckpts_dir = Path("./checkpoints", filename, model_name)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # 数据预处理
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    target_transform = [
        TextNormalization(),
        BertTokenizer(max_length=max_length)
    ]
    pad_token_id = target_transform[1].tokenizer.pad_token_id
    text_vocab_size = target_transform[1].tokenizer.vocab_size

    # 加载数据集
    train_dataset = Clotho(
        root=root,
        split="train",
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )
    test_dataset = Clotho(
        root=root,
        split="test",
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )

    # 采样器
    train_sampler = InfiniteSampler(train_dataset)
    eval_train_sampler = PseudoRandomSampler(train_dataset)
    eval_test_sampler = PseudoRandomSampler(test_dataset)

    # 数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    eval_train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=eval_train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    eval_test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=eval_test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 加载 CLAP 编码器
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name, device=device)

    # 加载 Llama 解码器
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name,
        audio_latent_dim=1024,  # 确保与 CLAP 输出维度一致
        text_vocab_size=text_vocab_size
    )
    llm_decoder.to(device)

    # 优化器
    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate)

    # 训练循环
    for step, data in enumerate(tqdm(train_dataloader)):
        # 数据移到设备
        audio = data["audio"].to(device)
        text_ids = data["target"].to(device)

        # 获取音频特征
        audio_latent = get_audio_latent(
            model_name=audio_encoder_name,
            model=audio_encoder,
            audio=audio,
            device=device
        )

        # 模型输入
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        # 前向传播
        llm_decoder.train()
        output_seqs = llm_decoder(
            seqs=input_seqs,
            seq_types=seq_types,
            mask=None
        )

        # 计算损失
        loss = caption_loss(
            output_seqs=output_seqs,
            input_seqs=input_seqs,
            ignore_index=pad_token_id
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证与保存
        if step % test_every_n_steps == 0:
            train_loss = validate(
                dataloader=eval_train_dataloader,
                audio_encoder_name=audio_encoder_name,
                audio_encoder=audio_encoder,
                llm_decoder=llm_decoder,
                pad_token_id=pad_token_id,
                device=device
            )
            test_loss = validate(
                dataloader=eval_test_dataloader,
                audio_encoder_name=audio_encoder_name,
                audio_encoder=audio_encoder,
                llm_decoder=llm_decoder,
                pad_token_id=pad_token_id,
                device=device
            )

            print(f"------ Step: {step} ------")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")

        if step % save_every_n_steps == 0:
            ckpt_path = ckpts_dir / f"step={step}.pth"
            torch.save(llm_decoder.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

        if step >= training_steps:
            break


def get_audio_encoder(model_name: str, device: str) -> (nn.Module, int):
    """加载 CLAP 编码器（适配设备）"""
    if model_name == "CLAP":
        # 初始化 CLAP，自动适配设备
        clap = CLAP(version="2023", use_cuda=(device == "cuda"))
        return clap, 1024  # CLAP 输出维度为 1024
    else:
        raise ValueError(f"不支持的音频编码器: {model_name}")


def get_audio_latent(model_name: str, model: CLAP, audio: torch.Tensor, device: str) -> torch.Tensor:
    """生成音频特征，确保维度正确"""
    if model_name != "CLAP":
        raise ValueError(f"不支持的音频编码器: {model_name}")

    batch_size = audio.shape[0]
    latents = []

    for i in range(batch_size):
        # 1. 提取单通道音频数据
        audio_np = audio[i, 0].cpu().numpy()

        # 2. 保存为临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, samplerate=32000)

            # 3. 用 CLAP 提取特征
            with torch.no_grad():
                embeddings = model.get_audio_embeddings([tmp.name], resample=True)
                audio_emb = torch.tensor(embeddings).to(device)
                
                # 验证维度
                print(f"CLAP 输出维度: {audio_emb.shape}")
                if audio_emb.shape[-1] != 1024:
                    raise ValueError(f"期望 CLAP 输出维度为 1024，但得到 {audio_emb.shape[-1]}")
                
                latents.append(audio_emb)

        # 4. 清理临时文件
        os.unlink(tmp.name)

    # 拼接特征并调整维度：(batch_size, 1, 1024)
    latents = torch.cat(latents, dim=0).unsqueeze(1)
    return latents


def get_llm_decoder(model_name: str, audio_latent_dim: int, text_vocab_size: int) -> nn.Module:
    """初始化 Llama 解码器，适配 1024 维度的音频输入"""
    if model_name == "Llama":
        config = LlamaConfig(
            block_size=1024,
            audio_latent_dim=audio_latent_dim,  # 应为 1024
            vocab_size=text_vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)
    else:
        raise ValueError(f"不支持的 LLM 解码器: {model_name}")


def caption_loss(output_seqs: list[torch.Tensor], input_seqs: list[torch.Tensor], ignore_index: int) -> torch.float:
    """计算文本生成损失（CrossEntropy）"""
    _, output_text_logits = output_seqs
    _, target_text_ids = input_seqs
    
    # 关键修改：确保目标标签是 Long 类型
    target_text_ids = target_text_ids.long()
    
    # 裁剪维度以匹配标签
    loss = F.cross_entropy(
        input=output_text_logits[:, :-1, :].flatten(0, 1),
        target=target_text_ids[:, 1:].flatten(0, 1),
        ignore_index=ignore_index
    )
    return loss


def validate(dataloader: DataLoader, audio_encoder_name: str, audio_encoder: CLAP, llm_decoder: nn.Module, pad_token_id: int, device: str, valid_steps=10) -> float:
    """验证模型，计算平均损失"""
    llm_decoder.eval()
    losses = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            audio = data["audio"].to(device)
            text_ids = data["target"].to(device)

            audio_latent = get_audio_latent(
                model_name=audio_encoder_name,
                model=audio_encoder,
                audio=audio,
                device=device
            )

            input_seqs = [audio_latent, text_ids]
            seq_types = ["audio", "text"]

            outputs = llm_decoder(seqs=input_seqs, seq_types=seq_types, mask=None)
            loss = caption_loss(outputs, input_seqs, ignore_index=pad_token_id)
            losses.append(loss.item())

            if step >= valid_steps:
                break

    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)