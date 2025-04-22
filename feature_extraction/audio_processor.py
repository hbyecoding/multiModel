import torch
import whisper
from typing import Dict, Any
import numpy as np
from pydub import AudioSegment
import librosa

class AudioProcessor:
    def __init__(self, model_name: str = 'base'):
        """初始化音频处理器
        
        Args:
            model_name: Whisper模型名称
        """
        self.model = whisper.load_model(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """音频转录
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录结果
            
        Raises:
            FileNotFoundError: 当音频文件不存在时
            RuntimeError: 当音频文件格式不支持或损坏时
        """
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
        try:
            # 预处理音频文件
            audio = AudioSegment.from_file(audio_path)
            # 转换为WAV格式
            temp_path = audio_path + ".wav"
            audio.export(temp_path, format="wav")
            # 使用转换后的WAV文件进行转录
            result = self.model.transcribe(temp_path)
            # 删除临时文件
            os.remove(temp_path)
            return result
        except Exception as e:
            raise RuntimeError(f"音频处理失败: {str(e)}")

    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征数组
        """
        # 加载音频文件
        y, sr = librosa.load(audio_path)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 提取频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # 提取色谱图特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        return np.concatenate([mfcc.flatten(), 
                              spectral_centroids.flatten(),
                              chroma.flatten()])
    
    def align_transcription(self, audio_path: str) -> Dict[str, Any]:
        """对齐音频转录结果
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            对齐结果
        """
        # 转录音频
        result = self.transcribe(audio_path)
        
        # 提取时间戳信息
        segments = result['segments']
        aligned_text = []
        
        for segment in segments:
            aligned_text.append({
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
                'confidence': segment['confidence']
            })
        
        return {
            'aligned_text': aligned_text,
            'language': result['language'],
            'duration': result['segments'][-1]['end']
        }
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """分析音频内容
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频分析结果
        """
        # 提取音频特征
        features = self.extract_features(audio_path)
        
        # 获取音频基本信息
        audio = AudioSegment.from_file(audio_path)
        
        # 计算特征统计量
        stats = {
            'feature_mean': np.mean(features),
            'feature_std': np.std(features),
            'feature_norm': np.linalg.norm(features)
        }
        
        return {
            'features': features,
            'stats': stats,
            'duration': len(audio) / 1000.0,  # 转换为秒
            'sample_rate': audio.frame_rate,
            'channels': audio.channels
        }