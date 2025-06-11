import unittest
import torch
from pathlib import Path
from feature_extraction import TextProcessor, ImageProcessor, AudioProcessor

class TestProcessors(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        
        # 测试数据
        self.test_texts = [
            "This is a test video",
            "Another test video description"
        ]
        
    def test_text_processor(self):
        """测试文本处理器"""
        features = self.text_processor.process(self.test_texts)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features), len(self.test_texts))
        
    def test_batch_processing(self):
        """测试批处理功能"""
        batch_size = 2
        features = self.text_processor.batch_process(self.test_texts, batch_size)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features), len(self.test_texts))

if __name__ == '__main__':
    unittest.main() 