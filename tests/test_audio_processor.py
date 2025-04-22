import os
import sys
import unittest
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.audio_processor = AudioProcessor()
        # 使用一个示例音频文件路径
        self.test_audio_path = os.path.join(
            os.path.dirname(__file__),
            'test_data',
            'test_audio.mp3'
        )
        
    def test_transcribe(self):
        result = self.audio_processor.transcribe(self.test_audio_path)
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        
    def test_extract_features(self):
        features = self.audio_processor.extract_features(self.test_audio_path)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(features.size > 0)
        
    def test_align_transcription(self):
        result = self.audio_processor.align_transcription(self.test_audio_path)
        self.assertIsInstance(result, dict)
        self.assertIn('aligned_text', result)
        self.assertIn('language', result)
        self.assertIn('duration', result)
        
    def test_analyze_audio(self):
        result = self.audio_processor.analyze_audio(self.test_audio_path)
        self.assertIsInstance(result, dict)
        self.assertIn('features', result)
        self.assertIn('stats', result)
        self.assertIn('duration', result)
        self.assertIn('sample_rate', result)
        self.assertIn('channels', result)
        
if __name__ == '__main__':
    unittest.main()