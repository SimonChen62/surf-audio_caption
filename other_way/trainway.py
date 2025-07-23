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
from panns_inference import AudioTagging
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from models.llama import Llama, LlamaConfig


def train(args):
    # Default parameters
    sr = 32000
    batch_size = 4
    num_workers = 0  # 避免多进程内存溢出
    pin_memory = True
    learning_rate = 5e-5  # 更小的学习率
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 10000
    device = "cuda"
    max_length = 30
    clip_duration = 10.
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"

    filename = Path(__file__).stem

    # Dataset
    root = "I:/资料/1/surf-2025/mini_audio_caption/clotho"

    model_name = f"{audio_encoder_name}_{llm_decoder_name}"
    ckpts_dir = Path("./checkpoints", filename, model_name)
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    target_transform = [
        TextNormalization(),
        BertTokenizer(max_length=max_length)
    ]
    pad_token_id = target_transform[1].tokenizer.pad_token_id
    text_vocab_size = target_transform[1].tokenizer.vocab_size

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

    # Model
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)

    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name,
        audio_latent_dim=audio_latent_dim,
        text_vocab_size=text_vocab_size
    )
    llm_decoder.to(device)

    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate)

    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step, data in enumerate(tqdm(train_dataloader)):
        audio = data["audio"].to(device)
        text_ids = data["target"].to(device)

        if torch.isnan(audio).any() or torch.isinf(audio).any():
            print("Skip batch due to invalid audio data")
            continue

        if torch.isnan(text_ids).any() or torch.isinf(text_ids).any():
            print("Skip batch due to invalid text data")
            continue

        # ✅ 正确控制是否使用混合精度或启用梯度计算
        if use_amp:
            with torch.cuda.amp.autocast():
                audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)
                input_seqs = [audio_latent, text_ids]
                seq_types = ["audio", "text"]
                llm_decoder.train()
                output_seqs = llm_decoder(seqs=input_seqs, seq_types=seq_types, mask=None)
                loss = caption_loss(output_seqs=output_seqs, input_seqs=input_seqs, ignore_index=pad_token_id)
        else:
            with torch.enable_grad():  # ⚠️ 这里必须启用梯度计算
                audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)
                input_seqs = [audio_latent, text_ids]
                seq_types = ["audio", "text"]
                llm_decoder.train()
                output_seqs = llm_decoder(seqs=input_seqs, seq_types=seq_types, mask=None)
                loss = caption_loss(output_seqs=output_seqs, input_seqs=input_seqs, ignore_index=pad_token_id)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Skip update due to invalid loss: {loss.item()}")
            continue

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(llm_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_decoder.parameters(), max_norm=1.0)
            optimizer.step()

        if step % test_every_n_steps == 0:
            train_loss = validate(eval_train_dataloader, audio_encoder_name, audio_encoder, llm_decoder, pad_token_id)
            test_loss = validate(eval_test_dataloader, audio_encoder_name, audio_encoder, llm_decoder, pad_token_id)

            print(f"------ step: {step} ------")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Test loss: {test_loss:.4f}")

        if step % save_every_n_steps == 0:
            ckpt_path = ckpts_dir / f"step={step}.pth"
            torch.save(llm_decoder.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == training_steps:
            break


def get_audio_encoder(model_name: str) -> (nn.Module, int):
    if model_name == "Cnn14":
        model = AudioTagging().model
        latent_dim = 2048
        return model, latent_dim
    else:
        raise ValueError(model_name)


def get_llm_decoder(model_name: str, audio_latent_dim: int, text_vocab_size: int) -> nn.Module:
    if model_name == "Llama":
        config = LlamaConfig(
            block_size=1024,
            audio_latent_dim=audio_latent_dim,
            vocab_size=text_vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)
    else:
        raise ValueError(model_name)


def get_audio_latent(model_name: str, model: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    if model_name == "Cnn14":
        with torch.no_grad():
            model.eval()
            latent = model(audio[:, 0, :])["embedding"]
            latent = latent.unsqueeze(1)  # Add time dimension
        return latent
    else:
        raise ValueError(model_name)


def caption_loss(output_seqs: list[torch.Tensor], input_seqs: list[torch.Tensor], ignore_index: int) -> torch.float:
    _, output_text_logits = output_seqs
    _, target_text_ids = input_seqs

    output_logits = output_text_logits[:, :-1]  # (B, T-1, V)
    targets = target_text_ids[:, 1:]  # (B, T-1)

    loss = F.cross_entropy(
        output_logits.reshape(-1, output_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index
    )

    return loss


def validate(dataloader: DataLoader, audio_encoder_name: str, audio_encoder: nn.Module, llm_decoder: nn.Module,
             pad_token_id: int, valid_steps=10) -> float:
    device = next(audio_encoder.parameters()).device
    losses = []

    with torch.inference_mode():
        for step, data in enumerate(dataloader):
            audio = data["audio"].to(device)
            text_ids = data["target"].to(device)

            audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)

            input_seqs = [audio_latent, text_ids]
            seq_types = ["audio", "text"]

            llm_decoder.eval()
            outputs = llm_decoder(seqs=input_seqs, seq_types=seq_types, mask=None)

            loss = caption_loss(outputs, input_seqs, ignore_index=pad_token_id)

            if not torch.isnan(loss) and not torch.isinf(loss):
                losses.append(loss.item())

            if step == valid_steps:
                break

    return np.mean(losses) if len(losses) > 0 else float('nan')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)