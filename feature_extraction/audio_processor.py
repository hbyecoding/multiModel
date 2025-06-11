import torch
import logging
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import List, Union, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        """初始化音频处理器"""
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 设置音频参数
        self.target_sample_rate = 16000
        self.max_duration = 30  # 最大处理时长（秒）
    
    def load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            waveform: 音频波形
        """
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            logger.error(f"加载音频时出错 {audio_path}: {str(e)}")
            return None
    
    def process_single_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """处理单个音频
        
        Args:
            waveform: 音频波形
            
        Returns:
            features: 音频特征张量
        """
        if waveform is None:
            return None
        
        # 截取指定长度
        max_length = self.target_sample_rate * self.max_duration
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        
        # 预处理音频
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = torch.mean(outputs.last_hidden_state, dim=1)  # 平均池化
        
        return features
    
    def process(self, audio_paths: List[Union[str, Path]]) -> torch.Tensor:
        """处理音频列表
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            features: 音频特征张量
        """
        features = []
        
        for path in audio_paths:
            # 加载音频
            waveform = self.load_audio(path)
            if waveform is None:
                continue
            
            # 处理音频
            feature = self.process_single_audio(waveform)
            if feature is not None:
                features.append(feature)
        
        if not features:
            raise ValueError("没有成功处理任何音频")
        
        # 堆叠所有特征
        features = torch.cat(features, dim=0)
        return features
    
    def batch_process(self, audio_paths: List[Union[str, Path]], 
                     batch_size: int = 16) -> torch.Tensor:
        """批量处理音频
        
        Args:
            audio_paths: 音频文件路径列表
            batch_size: 批次大小
            
        Returns:
            features: 音频特征张量
        """
        features = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_features = self.process(batch_paths)
            features.append(batch_features)
        
        features = torch.cat(features, dim=0)
        return features
    
    def compute_similarity(self, audio1_path: Union[str, Path],
                         audio2_path: Union[str, Path]) -> float:
        """计算两段音频的相似度
        
        Args:
            audio1_path: 第一段音频路径
            audio2_path: 第二段音频路径
            
        Returns:
            similarity: 相似度分数
        """
        # 加载音频
        waveform1 = self.load_audio(audio1_path)
        waveform2 = self.load_audio(audio2_path)
        
        if waveform1 is None or waveform2 is None:
            return 0.0
        
        # 提取特征
        feature1 = self.process_single_audio(waveform1)
        feature2 = self.process_single_audio(waveform2)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(feature1, feature2, dim=1)
        return similarity.item()