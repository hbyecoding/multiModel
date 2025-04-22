import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from .clip_model import CLIPModel
from .whisper_model import WhisperModel
from .llama_model import LlamaModel
import cv2
import numpy as np
from textblob import TextBlob
import librosa

class MultimodalFusion:
    def __init__(self):
        """初始化多模态融合模型"""
        self.clip_model = CLIPModel()
        self.whisper_model = WhisperModel()
        self.llama_model = LlamaModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_face_ratio(self, image: np.ndarray) -> float:
        """分析图像中人脸占比
        
        Args:
            image: 输入图像
            
        Returns:
            人脸占比
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        total_face_area = sum(w * h for (x, y, w, h) in faces)
        image_area = image.shape[0] * image.shape[1]
        
        return total_face_area / image_area if image_area > 0 else 0.0
    
    def analyze_emotion_match(self, title: str, audio: np.ndarray) -> float:
        """分析标题情感与音频情绪匹配度
        
        Args:
            title: 视频标题
            audio: 音频数据
            
        Returns:
            情感匹配度
        """
        # 分析标题情感
        blob = TextBlob(title)
        title_polarity = blob.sentiment.polarity
        
        # 分析音频情绪
        audio_features = librosa.feature.mfcc(y=audio)
        audio_energy = librosa.feature.rms(y=audio)
        audio_tempo = librosa.beat.tempo(y=audio)[0]
        
        # 简单的情绪映射规则
        audio_mood = (np.mean(audio_energy) + audio_tempo/120) / 2
        audio_polarity = (audio_mood - 0.5) * 2
        
        # 计算匹配度
        match_score = 1 - abs(title_polarity - audio_polarity) / 2
        return match_score
    
    def evaluate_title_diversity(self, titles: List[str]) -> float:
        """评估标题多样性
        
        Args:
            titles: 标题列表
            
        Returns:
            多样性分数
        """
        unique_words = set()
        total_words = 0
        
        for title in titles:
            words = title.split()
            unique_words.update(words)
            total_words += len(words)
        
        return len(unique_words) / total_words if total_words > 0 else 0.0
    
    def fuse_features(self, title: str, image: np.ndarray, audio: np.ndarray) -> Dict[str, Any]:
        """融合多模态特征
        
        Args:
            title: 视频标题
            image: 封面图像
            audio: 音频数据
            
        Returns:
            融合结果
        """
        # 特征提取
        text_features = self.llama_model.analyze_content(title)
        image_text_similarity = self.clip_model.compute_similarity(title, image)
        audio_features = self.whisper_model.transcribe(audio)
        
        # 计算评估指标
        face_ratio = self.analyze_face_ratio(image)
        emotion_match = self.analyze_emotion_match(title, audio)
        
        # 生成多样化标题
        diverse_titles = [
            self.llama_model.optimize_title(title, ["engaging", "creative"])["optimized_title"],
            self.llama_model.optimize_title(title, ["emotional", "dramatic"])["optimized_title"],
            self.llama_model.optimize_title(title, ["informative", "professional"])["optimized_title"]
        ]
        diversity_score = self.evaluate_title_diversity(diverse_titles)
        
        return {
            "metrics": {
                "face_ratio": face_ratio,
                "ctr_boost": face_ratio > 0.3,  # 人脸占比>30%时CTR提升
                "emotion_match_score": emotion_match,
                "completion_rate_impact": emotion_match > 0.85,  # 情感匹配度高时完播率提升
                "title_diversity_score": diversity_score,
                "diversity_improvement": diversity_score > 0.7  # 标题多样性提升
            },
            "features": {
                "text_analysis": text_features,
                "image_text_similarity": image_text_similarity,
                "audio_transcription": audio_features
            },
            "diverse_titles": diverse_titles
        }