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
# 新增：HTS-AT 依赖
import sys
from collections import OrderedDict

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from models.llama import Llama, LlamaConfig


# --------------------------
# 新增：HTS-AT 模型加载逻辑
# --------------------------
class HTSAT(nn.Module):
    """简化的 HTS-AT 模型包装类，用于特征提取"""
    def __init__(self, checkpoint_path):
        super().__init__()
        import htsat  # 直接从当前目录导入 htsat.py

        # 初始化 HTS-AT 配置（根据原论文参数）
        self.config = htsat.HTSATConfig()

        # 加载 HTS-AT 模型
        self.model = htsat.HTSAT(self.config)

        # 加载预训练权重（假设已下载到本地）
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict, strict=False)

        # 冻结特征提取部分（仅使用特征，不微调）
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        输入：音频波形 (batch_size, 1, samples)，单通道，采样率32000
        输出：音频特征 (batch_size, embed_dim)
        """
        # HTS-AT 需先转换为梅尔频谱，再输入 transformer
        x = self.model.mel_extractor(x)  # (batch, 1, mel_bins, time_steps)
        x = x.transpose(1, 3)  # (batch, time_steps, mel_bins, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, time_steps, mel_bins)
        
        # 提取 transformer 输出的 [CLS] 特征
        x = self.model.transformer(x)  # (batch, time_steps + 1, embed_dim)，+1 是 [CLS] token
        cls_feat = x[:, 0, :]  # 取 [CLS] 特征作为全局特征
        return cls_feat


# --------------------------
# 核心函数修改
# --------------------------
def train(args):
    # 基础参数（保持不变，仅修改音频编码器名称）
    sr = 32000
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
    audio_encoder_name = "HTS-AT"  # 替换为 HTS-AT
    llm_decoder_name = "Llama"

    filename = Path(__file__).stem
    root = "I:/资料/1/surf-2025/mini_audio_caption/clotho"  # 你的数据集路径

    # 模型保存目录
    model_name = f"{audio_encoder_name}_{llm_decoder_name}"
    ckpts_dir = Path("./checkpoints", filename, model_name)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # 数据预处理（与原代码一致）
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    target_transform = [
        TextNormalization(),
        BertTokenizer(max_length=max_length)
    ]
    pad_token_id = target_transform[1].tokenizer.pad_token_id
    text_vocab_size = target_transform[1].tokenizer.vocab_size

    # 数据集加载（与原代码一致）
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

    # 数据加载器（与原代码一致）
    train_sampler = InfiniteSampler(train_dataset)
    eval_train_sampler = PseudoRandomSampler(train_dataset)
    eval_test_sampler = PseudoRandomSampler(test_dataset)

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

    # 加载 HTS-AT 音频编码器（核心修改）
    audio_encoder, audio_latent_dim = get_audio_encoder(
        model_name=audio_encoder_name,
        checkpoint_path="I:/资料/1/surf-2025/mini_audio_caption/htsat_audioset_pretrained.pth" # 你的 HTS-AT 权重路径
    )
    audio_encoder.to(device)

    # 加载 Llama 解码器（注意：音频特征维度已改为 HTS-AT 的输出维度）
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name,
        audio_latent_dim=audio_latent_dim,
        text_vocab_size=text_vocab_size
    )
    llm_decoder.to(device)

    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate)

    # 训练循环（与原代码一致）
    for step, data in enumerate(tqdm(train_dataloader)):
        audio = data["audio"].to(device)  # (b, 1, t)
        text_ids = data["target"].to(device)

        audio_latent = get_audio_latent(
            model_name=audio_encoder_name,
            model=audio_encoder,
            audio=audio
        )

        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        llm_decoder.train()
        output_seqs = llm_decoder(
            seqs=input_seqs,
            seq_types=seq_types,
            mask=None
        )

        loss = caption_loss(
            output_seqs=output_seqs,
            input_seqs=input_seqs,
            ignore_index=pad_token_id
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % test_every_n_steps == 0:
            train_loss = validate(
                dataloader=eval_train_dataloader,
                audio_encoder_name=audio_encoder_name,
                audio_encoder=audio_encoder,
                llm_decoder=llm_decoder,
                pad_token_id=pad_token_id
            )
            test_loss = validate(
                dataloader=eval_test_dataloader,
                audio_encoder_name=audio_encoder_name,
                audio_encoder=audio_encoder,
                llm_decoder=llm_decoder,
                pad_token_id=pad_token_id
            )

            print(f"------ step: {step} ------")
            print(f"Train loss: {train_loss}")
            print(f"Test loss: {test_loss}")

        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(llm_decoder.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == training_steps:
            break


def get_audio_encoder(model_name: str, checkpoint_path: str) -> (nn.Module, int):
    """加载 HTS-AT 编码器（替换原 Cnn14 加载逻辑）"""
    if model_name == "HTS-AT":
        model = HTSAT(checkpoint_path)
        latent_dim = 768  # 与 HTS-AT 的 embed_dim 一致
        return model, latent_dim
    else:
        raise ValueError(f"不支持的编码器：{model_name}")


def get_audio_latent(model_name: str, model: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    """提取 HTS-AT 的音频特征（替换原 Cnn14 特征提取）"""
    if model_name == "HTS-AT":
        with torch.no_grad():
            model.eval()
            # HTS-AT 输入为 (b, 1, t)，输出为 (b, 768)
            latent = model(audio)  # (b, 768)
            latent = latent[:, None, :]  # 调整为 (b, 1, 768)，与原代码维度格式一致
        return latent
    else:
        raise ValueError(f"不支持的编码器：{model_name}")


# --------------------------
# 以下函数与原代码一致（无需修改）
# --------------------------
def get_llm_decoder(model_name: str, audio_latent_dim: int, text_vocab_size: int) -> nn.Module:
    if model_name == "Llama":
        config = LlamaConfig(
            block_size=1024,
            audio_latent_dim=audio_latent_dim,  # 自动适配 HTS-AT 的 768 维度
            vocab_size=text_vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)
    else:
        raise ValueError(model_name)


def caption_loss(output_seqs: list[torch.Tensor], input_seqs: list[torch.Tensor], ignore_index: int) -> torch.float:
    output_audio_logits, output_text_logtis = output_seqs
    target_audio_latents, target_text_ids = input_seqs

    loss = F.cross_entropy(
        input=output_text_logtis[:, :-1, :].flatten(0, 1),
        target=target_text_ids[:, 1:].flatten(0, 1),
        ignore_index=ignore_index
    )
    return loss


def validate(dataloader: DataLoader, audio_encoder_name: str, audio_encoder: nn.Module, llm_decoder: nn.Module, pad_token_id: int, valid_steps=10) -> float:
    device = next(audio_encoder.parameters()).device
    losses = []
    for step, data in enumerate(dataloader):
        audio = data["audio"].to(device)
        text_ids = data["target"].to(device)
        audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]
        llm_decoder.eval()
        with torch.no_grad():
            outputs = llm_decoder(seqs=input_seqs, seq_types=seq_types, mask=None)
            loss = caption_loss(outputs, input_seqs, ignore_index=pad_token_id)
            losses.append(loss.item())
        if step == valid_steps:
            break
    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)