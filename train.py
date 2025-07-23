import argparse #用来解析命令行参数的模块
from pathlib import Path  #路径处理模块 更快捷

import numpy as np  # 数组处理模块 用于数值计算和数组操作
import torch
import torch.nn as nn #神经网络模块 提供构建神经网络的基础组件
import torch.nn.functional as F #无状态函数接口
import torch.optim as optim #优化器 包括adam，sgd
from audidata.datasets import Clotho #音频描述数据集 帮忙自动预处理csv和wav文件
from audidata.io.crops import RandomCrop #音频随机裁剪
from audidata.samplers import InfiniteSampler, PseudoRandomSampler
from audidata.transforms import Mono #音频转换为单声道
from panns_inference import AudioTagging #预训练编码器
from torch.utils.data import DataLoader #数据加载器 用于批量处理数据
from tqdm import tqdm #进度条显示
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
# import wandb

from data.text_normalization import TextNormalization  #文本归一化
from data.text_tokenization import BertTokenizer  #BERT分词器
from models.llama import Llama, LlamaConfig #Llama模型和配置  开源大语言模型 用于生成文本描述


def train(args):
    
    # Default parameters
    sr = 32000  # To be consistent with the encoder 采样率（hz），与音频编码器输入一致 audiotagging模型的输入采样率为32000Hz
    batch_size = 4 #每批次样本数 本来16
    num_workers = 4  #数据加载的工作线程数
    pin_memory = True # 是否将数据加载到固定内存中，以加快数据传输速度
    learning_rate = 1e-4    # 0.0001 学习率 adamw的学习率
    test_every_n_steps = 200 # 每隔多少步进行一次测试
    save_every_n_steps = 2000 # 每隔多少步保存一次模型
    training_steps = 10000 # 训练的总步数
    # wandb_log = True # 是否使用 wandb 记录实验
    device = "cuda" 
    max_length = 30  # 最大文本长度 token数，一种截取度量单位，一个英文单词大约1.3token，中文一个汉字大约1个token
    clip_duration = 10.  # 音频片段时长 按秒
    audio_encoder_name = "Cnn14" #音频编译器名字 
    llm_decoder_name = "Llama" #语言模型解码器名字  用于更好的表达音频内容，更完整 

    filename = Path(__file__).stem #当前文件名 train
    
    # Dataset
    root = "I:/资料/1/surf-2025/mini_audio_caption/clotho" #数据集目录

    # Checkpoints directory  模型保存路径
    model_name = "{}_{}".format(audio_encoder_name, llm_decoder_name) #检查编译器 这个意思是把两个编译器名字用_连接起来，形成一个新的名字
    ckpts_dir = Path("./checkpoints", filename, model_name) #构造一个目录路径，/checkpoint/train/音频编译器_语言模型解码器名字
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True) # #创建目录，如果不存在则创建，parents=True表示创建多级目录，exist_ok=True表示如果目录已存在则不报错
    #设置数据集路径和模型检查点保存路径
    #使用模型名称创建子目录，便于管理不同配置的模型

    # 音频裁剪器，随机裁剪片段
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.) #裁剪音频时长为 clip_duration 秒，end_pad=0.表示不在音频末尾添加填充

    # 字幕转化，归一化文本  用列表存放函数按顺序执行
    target_transform = [
        TextNormalization(),  # 移除标点符号
        BertTokenizer(max_length=max_length)  # 最大token Convert captions to token IDs 转化成token的形式
    ]
    pad_token_id = target_transform[1].tokenizer.pad_token_id  # 获取填充标记的 ID  用于填充到相同长度 序列是token列表/深度学习模型要求输入数据具有相同序列
    text_vocab_size = target_transform[1].tokenizer.vocab_size  # 获取文本词汇表大小  ？
    # Datasets
    train_dataset = Clotho(
        root=root,
        split="train", #使用训练集
        sr=sr,
        crop=crop,
        transform=Mono(), # Mono() 将音频转换为单声道
        target_transform=target_transform # 将字幕转换为 token ID，移除标点
    )
    #测试数据集
    test_dataset = Clotho(
        root=root,
        split="test",
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )

    # 采样器 决定如何在数据集中选择样本
    train_sampler = InfiniteSampler(train_dataset)  # 无限采样器，允许无限次迭代训练数据集
    eval_train_sampler = PseudoRandomSampler(train_dataset) # 随机采样器，随机选择训练数据集中的样本
    eval_test_sampler = PseudoRandomSampler(test_dataset) # 随机采样器，随机选择测试数据集中的样本  用于评估
    
    # 数据加载器，批量batch处理
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

    # 预训练音频编码器，提取特征
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name) #返回预训练的cnn14模型和音频特征维度 是类的实例
    audio_encoder.to(device)

    # LLM decoder解码器，基于音频特征生成文本描述
    llm_decoder = get_llm_decoder( #调用函数
        model_name=llm_decoder_name, 
        audio_latent_dim=audio_latent_dim,  #音频特征维度
        text_vocab_size=text_vocab_size #文本词汇表大小
    )
    llm_decoder.to(device)

    # Optimizer
    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate) #需要和结果label也就是文字对应  优化器选取
    #创建优化器，为后续准备
    # if wandb_log:
        # andb.init(project="mini_audio_caption", name="{}".format(model_name))

    # 训练循环
    for step, data in enumerate(tqdm(train_dataloader)): #enumerate 让你能同时获得“第几个 batch”和“数据本身
        # step: 当前是第几个 batch（0, 1, 2, ...）
        # data: 当前 batch 的数据（如音频和描述）
        # 这里写训练代码 enuermate()是一个迭代器，返回每个batch的索引和数据
        # Move data to device
        audio = data["audio"].to(device)  # shape: (b, c, t_audio) 音频数据 音频波形数据，tensor，形状（batch，通道，音频样本数）
        text_ids = data["target"].to(device)  # shape: (b, t_text) 文本数据 文本token id相当于在数据库的idex，tensor， 形状（batch，文本长度）
        
        # 提取音频嵌入
        audio_latent = get_audio_latent( #用之前定义好的音频编码器对给的音频数据进行特征提取，返回潜在特征
            model_name=audio_encoder_name, 
            model=audio_encoder, audio=audio #调用前面的函数，即创建音频编码器实例对象，并传输数据。即定义了方法和内容
        )
        
        # 结合音频嵌入和文本标识符
        input_seqs = [audio_latent, text_ids] #结合音频特征和文本token id，形成输入序列列表，相当于把音频和对应正确的文本结合
        seq_types = ["audio", "text"] #label

        # Forward
        llm_decoder.train() #设置为训练模式 把这个语言模型设置为训练模式，启用dropout等训练特性
        #前向
        output_seqs = llm_decoder( #逐个token id 预测文本
            seqs=input_seqs, #输入序列
            seq_types=seq_types, #每个序列的类型
            mask=None #没有特殊关注
        ) #有音频输出+每个文本的预测分数 相当于最后一位是n*m维度存储
        # list of output, e.g., [(b, t_audio, audio_dim), (b, t_text, vocab_size)]

        # 计算损失
        loss = caption_loss(
            output_seqs=output_seqs, 
            input_seqs=input_seqs, 
            ignore_index=pad_token_id #忽略padding的token id 用来补齐序列的
        )

        # 反向传播优化
        optimizer.zero_grad()   # Reset all parameter.grad to 0 先清空grad，防止梯度累积 grad是每次重新计算的
        loss.backward()     # Update all parameter.grad 自动计算所有参数的梯度
        optimizer.step()    # Update all parameters based on all parameter.grad 更新新的权重去训练

        # 定期评估模型
        if step % test_every_n_steps == 0: #每两百batch进行一次评估

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

            print("------ step: {} ------".format(step))
            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))

            # if wandb_log:
                # wandb.log(
                    # ata={"train_loss": train_loss, "test_loss": test_loss},
                    # step=step
                # )
        
        # Save model
        if step % save_every_n_steps == 0: #每两千次保存一次模型即权重参数  这个是存档，上面那个是更新应用
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step)) #按照step=step.pth保存
            torch.save(llm_decoder.state_dict(), ckpt_path) #保存模型当前模型的所有参数，权重和偏置等
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break
        

def get_audio_encoder(model_name: str) -> nn.Module: #给他模型名字，继承nn.Module类，返回一个音频编码器模型和音频特征维度
    r"""Load pretrained audio encoder."""
    if model_name == "Cnn14":
        model = AudioTagging().model #是他的一个属性，调用了神经模型
        latent_dim = 2048
        return model, latent_dim #返回预处理数据和音频特征维度

    else:
        raise ValueError(model_name)


def get_llm_decoder(  
    model_name: str, 
    audio_latent_dim: int, 
    text_vocab_size: int
) -> nn.Module: #标注函数返回值的类型 约定返回值是什么类型的结果 就是这个模块
    r"""Initialize LLM decoder."""
    if model_name == "Llama":
        config = LlamaConfig(
            block_size=1024,  # 输入序列的最大长度
            audio_latent_dim=audio_latent_dim, # 音频特征维度
            vocab_size=text_vocab_size, # 文本词汇表大小
            n_layer=12, # LLM的层数  每一层都在提炼信息
            n_head=12, # 注意力头数
            n_embd=768 # 嵌入维度每个token的嵌入维度 让每一个token都有一个768维度的向量表示
        )
        return Llama(config=config) #用这个配置创建llama模型实例，用来预测

    else:
        raise ValueError(model_name)    


def get_audio_latent( #返回音频潜在特征
    model_name: str, 
    model: nn.Module, 
    audio: torch.Tensor #音频波形数据
) -> torch.Tensor:
    r"""Calculate audio latent from an audio.

    Args:
        model_name: str
        model: nn.Module
        audio: (batch_size, channels_num, samples_num)

    Outputs:
        audio_latent: (batch_size, time_steps, emb_dim)
    """
    if model_name == "Cnn14":
        with torch.no_grad():
            model.eval()
            latent = model(audio[:, 0, :])["embedding"]  # (b, d) audio的形状（batch_size, channels_num, samples_num)，这里取第一个通道的音频数据
            #先用model模型对提取的单声道进行前向传播，再embedding提取特征，得到latent的形状为（batch_size, d），d是音频特征维度
            latent = latent[:, None, :]  # (b, t_audio, d) 插入一个新轴，符合后续transformer的输入要求，序列长度默认1
        return latent

    else:
        raise ValueError(model_name)        


def caption_loss(
    output_seqs: list[torch.Tensor], 
    input_seqs: list[torch.Tensor], 
    ignore_index: int
) -> torch.float:
    r"""Calculate caption loss."""

    output_audio_logits, output_text_logtis = output_seqs
    target_audio_latents, target_text_ids = input_seqs

    loss = F.cross_entropy(
        input=output_text_logtis[:, 0 : -1, :].flatten(0, 1),  # shape: (B*T合并, V) 取出n-1个位置的预测分数，不包括最后一个token 再站看变成二维，把batch和序列长度合并
        #用前文取预测后文，因为0和-1是bos和eos没意义，所以相当于正文不分都预测了
        #第三个是个高纬度向量，预测某个词的分数
        target=target_text_ids[:, 1 :].flatten(0, 1),  # shape: (B*T,) 取出第二个到最后一个
        ignore_index=ignore_index
    )

    return loss


def validate(
    dataloader: DataLoader, 
    audio_encoder_name: str,
    audio_encoder: nn.Module, #支持前向传播的神经网络结构
    llm_decoder: nn.Module, 
    pad_token_id: int,
    valid_steps=10
) -> float:  #就是告诉你所需要输入的东西
    r"""Validate the model on part of data.""" #是类似注释，但是程序可见可用doc访问存储

    device = next(audio_encoder.parameters()).device # #获取音频编码器的设备信息，通常是cuda或cpu next()是获取音频编码器的第一个参数的设备信息
    losses = [] #存储每个batch的损失值

    for step, data in enumerate(dataloader):

        # Move data to device
        audio = data["audio"].to(device)  # shape: (b, t_audio)
        text_ids = data["target"].to(device)  # shape: (b, t_text)
        
        # Extract audio embeddings
        audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)
        
        # Combine audio embeddings and text ids
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        # Forward
        llm_decoder.eval()
        outputs = llm_decoder(
            seqs=input_seqs,
            seq_types=seq_types,
            mask=None
        )

        # Loss
        loss = caption_loss(
            output_seqs=outputs, 
            input_seqs=input_seqs, 
            ignore_index=pad_token_id
        )
        losses.append(loss.item()) #加入损失量

        if step == valid_steps:
            break

    return np.mean(losses) #求平均
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train(args)