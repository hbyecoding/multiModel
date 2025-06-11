import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os
import json
import torch
import logging
from pathlib import Path
from torch.utils.data import Dataset
from pytube import YouTube
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        pass
        
    def load_market_data(self, file_path: str) -> pd.DataFrame:
        """加载行情数据
        
        Args:
            file_path: 行情数据文件路径
            
        Returns:
            处理后的行情数据DataFrame
        """
        # 读取行情数据
        df = pd.read_csv(file_path)
        
        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 基础数据清洗
        df = df.sort_index()  # 按日期排序
        df = df.fillna(method='ffill')  # 前向填充缺失值
        
        return df
    
    def load_fundamental_data(self, file_path: str) -> pd.DataFrame:
        """加载基本面数据
        
        Args:
            file_path: 基本面数据文件路径
            
        Returns:
            处理后的基本面数据DataFrame
        """
        # 读取基本面数据
        df = pd.read_csv(file_path)
        
        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date', 'stock_code'], inplace=True)
        
        # 基础数据清洗
        df = df.sort_index()
        df = df.fillna(method='ffill')
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标
        
        Args:
            df: 行情数据DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        return df
    
    def calculate_fundamental_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基本面因子
        
        Args:
            df: 基本面数据DataFrame
            
        Returns:
            添加基本面因子后的DataFrame
        """
        # 计算市值因子
        df['LN_MKT_CAP'] = np.log(df['market_cap'])
        
        # 计算估值因子
        df['PE'] = df['close'] / df['eps']
        df['PB'] = df['close'] / df['bps']
        
        # 计算成长因子
        df['REVENUE_YOY'] = df['revenue'].pct_change(periods=4)  # 假设季度数据
        df['PROFIT_YOY'] = df['net_profit'].pct_change(periods=4)
        
        return df
    
    def merge_data(self, market_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """合并行情数据和基本面数据
        
        Args:
            market_data: 行情数据DataFrame
            fundamental_data: 基本面数据DataFrame
            
        Returns:
            合并后的DataFrame
        """
        # 重置索引以便合并
        market_data = market_data.reset_index()
        fundamental_data = fundamental_data.reset_index()
        
        # 按日期和股票代码合并数据
        merged_data = pd.merge(market_data, fundamental_data,
                              on=['date', 'stock_code'],
                              how='inner')
        
        # 重新设置日期索引
        merged_data.set_index('date', inplace=True)
        
        return merged_data

class YouTubeDataset(Dataset):
    def __init__(self, data_dict):
        """初始化YouTube数据集
        
        Args:
            data_dict: 包含数据的字典
        """
        self.titles = data_dict['titles']
        self.thumbnails = data_dict['thumbnails']
        self.audio = data_dict.get('audio', None)
        self.descriptions = data_dict.get('descriptions', None)
        
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        item = {
            'title': self.titles[idx],
            'thumbnail': self.thumbnails[idx]
        }
        
        if self.audio is not None:
            item['audio'] = self.audio[idx]
            
        if self.descriptions is not None:
            item['description'] = self.descriptions[idx]
            
        return item

class YouTubeDataProcessor:
    def __init__(self):
        """初始化YouTube数据处理器"""
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.cache_path = self.data_path / 'cache'
        
        # 确保缓存目录存在
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def download_video_data(self, video_url: str) -> Dict[str, Any]:
        """下载YouTube视频数据
        
        Args:
            video_url: YouTube视频URL
            
        Returns:
            video_data: 视频数据字典
        """
        try:
            # 创建YouTube对象
            yt = YouTube(video_url)
            
            # 获取视频信息
            video_data = {
                'title': yt.title,
                'description': yt.description,
                'thumbnail_url': yt.thumbnail_url,
                'video_id': yt.video_id
            }
            
            # 下载缩略图
            response = requests.get(video_data['thumbnail_url'])
            if response.status_code == 200:
                thumbnail = Image.open(BytesIO(response.content))
                thumbnail_path = self.cache_path / f"{video_data['video_id']}_thumbnail.jpg"
                thumbnail.save(thumbnail_path)
                video_data['thumbnail_path'] = str(thumbnail_path)
            
            # 下载音频
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_path = self.cache_path / f"{video_data['video_id']}_audio.mp3"
            audio_stream.download(filename=str(audio_path))
            video_data['audio_path'] = str(audio_path)
            
            return video_data
            
        except Exception as e:
            logger.error(f"下载视频数据时出错: {str(e)}")
            return None
    
    def load_data(self) -> Dict[str, List]:
        """加载数据
        
        Returns:
            data: 数据字典
        """
        # 从JSON文件加载视频URL列表
        video_urls_path = self.data_path / 'video_urls.json'
        
        if not video_urls_path.exists():
            raise FileNotFoundError(f"找不到视频URL文件: {video_urls_path}")
        
        with open(video_urls_path, 'r') as f:
            video_urls = json.load(f)
        
        # 下载并处理每个视频的数据
        all_data = {
            'titles': [],
            'descriptions': [],
            'thumbnails': [],
            'audio': []
        }
        
        for url in video_urls:
            video_data = self.download_video_data(url)
            if video_data:
                all_data['titles'].append(video_data['title'])
                all_data['descriptions'].append(video_data['description'])
                all_data['thumbnails'].append(video_data['thumbnail_path'])
                all_data['audio'].append(video_data['audio_path'])
        
        return all_data
    
    def build_datasets(self, 
                      text_features: torch.Tensor,
                      image_features: torch.Tensor,
                      audio_features: torch.Tensor,
                      val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """构建训练和验证数据集
        
        Args:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 音频特征
            val_split: 验证集比例
            
        Returns:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        # 计算分割点
        total_samples = len(text_features)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        
        # 构建数据字典
        train_data = {
            'titles': text_features[:train_size],
            'thumbnails': image_features[:train_size],
            'audio': audio_features[:train_size]
        }
        
        val_data = {
            'titles': text_features[train_size:],
            'thumbnails': image_features[train_size:],
            'audio': audio_features[train_size:]
        }
        
        # 创建数据集
        train_dataset = YouTubeDataset(train_data)
        val_dataset = YouTubeDataset(val_data)
        
        return train_dataset, val_dataset
    
    def save_features(self, features: Dict[str, torch.Tensor], name: str):
        """保存特征
        
        Args:
            features: 特征字典
            name: 特征名称
        """
        save_path = self.cache_path / f"{name}_features.pt"
        torch.save(features, save_path)
        logger.info(f"特征已保存到: {save_path}")
    
    def load_features(self, name: str) -> Dict[str, torch.Tensor]:
        """加载特征
        
        Args:
            name: 特征名称
            
        Returns:
            features: 特征字典
        """
        load_path = self.cache_path / f"{name}_features.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"找不到特征文件: {load_path}")
            
        features = torch.load(load_path)
        logger.info(f"已加载特征: {load_path}")
        return features