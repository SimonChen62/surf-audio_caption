import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import sys
import logging
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置代理（如果需要）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"

# 文本处理模块
from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer

# Llama模型
from models.llama import Llama, LlamaConfig

# Hugging Face数据集库
try:
    from datasets import load_dataset, Dataset as HFDataset
    from transformers import BertTokenizer as HF_BertTokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("datasets 库未安装，将使用模拟数据")
    DATASETS_AVAILABLE = False

# -----------------------------
# 修改后的数据集类 - 更健壮的实现
# -----------------------------
class AudioCapsFeatureDataset(Dataset):
    def __init__(self, split="train", max_length=30, size=1000):
        self.max_length = max_length
        self.size = size
        self.split = split
        
        # 文本处理
        self.text_normalizer = TextNormalization()
        self.tokenizer = BertTokenizer(max_length=max_length)
        
        # 尝试加载真实数据集
        self.dataset = self._load_real_dataset()
        
        # 如果真实数据集加载失败，创建模拟数据集
        if self.dataset is None:
            logger.warning("无法加载真实数据集，将使用模拟数据")
            self.dataset = self._create_mock_dataset()
        
        logger.info(f"数据集加载完成: {len(self.dataset)} 个样本")

    def _load_real_dataset(self):
        """尝试加载真实数据集"""
        if not DATASETS_AVAILABLE:
            return None
            
        try:
            # 尝试加载 MusicCaps 数据集
            logger.info("尝试加载 google/MusicCaps 数据集...")
            dataset = load_dataset("google/MusicCaps", split=self.split)
            return dataset
        except Exception as e:
            logger.error(f"加载 google/MusicCaps 失败: {str(e)}")
        
        try:
            # 尝试加载 AudioCaps 数据集
            logger.info("尝试加载 khanhctn18/AudioCaps 数据集...")
            dataset = load_dataset("khanhctn18/AudioCaps", split=self.split)
            return dataset
        except Exception as e:
            logger.error(f"加载 khanhctn18/AudioCaps 失败: {str(e)}")
        
        try:
            # 尝试加载另一个替代数据集
            logger.info("尝试加载 ashraq/audioset 数据集...")
            dataset = load_dataset("ashraq/audioset", split=self.split)
            return dataset
        except Exception as e:
            logger.error(f"加载 ashraq/audioset 失败: {str(e)}")
        
        return None

    def _create_mock_dataset(self):
        """创建模拟数据集用于测试"""
        mock_data = []
        captions = [
            "a person speaking",
            "birds chirping in the forest",
            "car engine running",
            "rain falling on the roof",
            "children playing in the park",
            "a dog barking loudly",
            "piano music playing softly",
            "waves crashing on the shore",
            "thunder rumbling in the distance",
            "a crowd cheering at a sports event"
        ]
        
        for i in range(self.size):
            mock_data.append({
                "audio": {"array": np.random.randn(768).tolist()},
                "caption": random.choice(captions)
            })
        
        return HFDataset.from_list(mock_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 获取音频特征
        if "audio" in item and "array" in item["audio"]:
            audio_features = torch.tensor(item["audio"]["array"]).float()
        elif "feature" in item:
            audio_features = torch.tensor(item["feature"]).float()
        else:
            # 生成随机特征作为后备
            audio_features = torch.randn(768).float()
        
        # 获取文本描述
        if "text" in item:
            caption = item["text"]
        elif "caption" in item:
            caption = item["caption"]
        else:
            caption = "audio description"
        
        # 文本处理
        normalized_caption = self.text_normalizer(caption)
        tokenized_caption = self.tokenizer(normalized_caption)
        
        return {
            "audio_features": audio_features,
            "target": tokenized_caption
        }

# -----------------------------
# 训练函数
# -----------------------------
def train(args):
    # 默认参数
    batch_size = 4
    num_workers = 0  # 在Windows上可能需要设置为0
    pin_memory = True
    learning_rate = 1e-4
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 30
    audio_encoder_name = "Precomputed"
    llm_decoder_name = "Llama"
    
    logger.info(f"使用设备: {device}")

    # 模型名称和检查点目录
    filename = Path(__file__).stem
    model_name = f"{audio_encoder_name}_{llm_decoder_name}"
    ckpts_dir = Path("./checkpoints", filename, model_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = AudioCapsFeatureDataset(split="train", max_length=max_length, size=1000)
    test_dataset = AudioCapsFeatureDataset(split="test", max_length=max_length, size=200)
    
    # 采样器
    class InfiniteSampler:
        def __init__(self, dataset):
            self.dataset = dataset
            self.index = 0
            
        def __iter__(self):
            while True:
                yield self.index
                self.index = (self.index + 1) % len(self.dataset)
                
        def __len__(self):
            return float('inf')

    class PseudoRandomSampler:
        def __init__(self, dataset):
            self.dataset = dataset
            self.indices = list(range(len(dataset)))
            np.random.shuffle(self.indices)
            self.index = 0
            
        def __iter__(self):
            while True:
                yield self.indices[self.index]
                self.index = (self.index + 1) % len(self.indices)
                
        def __len__(self):
            return len(self.dataset)
    
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
    
    # 获取音频特征维度
    sample = train_dataset[0]
    audio_feature_dim = sample["audio_features"].shape[0]
    logger.info(f"音频特征维度: {audio_feature_dim}")

    # LLM解码器
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name,
        audio_latent_dim=audio_feature_dim,
        text_vocab_size=train_dataset.tokenizer.tokenizer.vocab_size
    )
    llm_decoder.to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in llm_decoder.parameters())
    logger.info(f"模型参数数量: {total_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate)

    # 训练循环
    logger.info("开始训练...")
    for step, data in enumerate(tqdm(train_dataloader, total=training_steps)):
        if step >= training_steps:
            break
            
        # 获取数据
        audio_features = data["audio_features"].to(device)
        text_ids = data["target"].to(device)
        
        # 直接使用预提取的音频特征
        audio_latent = audio_features.unsqueeze(1)  # 添加序列维度 (batch, 1, feature_dim)
        
        # 输入序列
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
            ignore_index=train_dataset.tokenizer.tokenizer.pad_token_id
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 定期评估
        if step % test_every_n_steps == 0:
            train_loss = validate(
                dataloader=eval_train_dataloader,
                llm_decoder=llm_decoder,
                pad_token_id=train_dataset.tokenizer.tokenizer.pad_token_id,
                device=device
            )
            test_loss = validate(
                dataloader=eval_test_dataloader,
                llm_decoder=llm_decoder,
                pad_token_id=train_dataset.tokenizer.tokenizer.pad_token_id,
                device=device
            )

            logger.info(f"------ 步骤: {step} ------")
            logger.info(f"训练损失: {train_loss:.4f}")
            logger.info(f"测试损失: {test_loss:.4f}")

        # 保存模型
        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(llm_decoder.state_dict(), ckpt_path)
            logger.info(f"模型已保存到 {ckpt_path}")

    # 保存最终模型
    final_path = Path(ckpts_dir, "final_model.pth")
    torch.save(llm_decoder.state_dict(), final_path)
    logger.info(f"训练完成! 最终模型已保存到 {final_path}")

# -----------------------------
# 辅助函数
# -----------------------------
def get_llm_decoder(model_name: str, audio_latent_dim: int, text_vocab_size: int) -> nn.Module:
    """初始化LLM解码器"""
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

def caption_loss(output_seqs: list, input_seqs: list, ignore_index: int) -> torch.Tensor:
    """计算字幕损失"""
    _, output_text_logits = output_seqs
    _, target_text_ids = input_seqs

    loss = F.cross_entropy(
        input=output_text_logits[:, :-1, :].flatten(0, 1),
        target=target_text_ids[:, 1:].flatten(0, 1),
        ignore_index=ignore_index
    )
    return loss

def validate(dataloader, llm_decoder, pad_token_id, device, valid_steps=10) -> float:
    """验证模型"""
    llm_decoder.eval()
    losses = []
    
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            if step >= valid_steps:
                break
                
            # 获取数据
            audio_features = data["audio_features"].to(device)
            text_ids = data["target"].to(device)
            
            # 直接使用预提取的音频特征
            audio_latent = audio_features.unsqueeze(1)
            
            # 输入序列
            input_seqs = [audio_latent, text_ids]
            seq_types = ["audio", "text"]

            # 前向传播
            outputs = llm_decoder(
                seqs=input_seqs,
                seq_types=seq_types,
                mask=None
            )

            # 计算损失
            loss = caption_loss(
                output_seqs=outputs,
                input_seqs=input_seqs,
                ignore_index=pad_token_id
            )
            losses.append(loss.item())
    
    llm_decoder.train()
    return np.mean(losses) if losses else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)