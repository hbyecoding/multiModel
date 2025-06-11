import os
import torch
import logging
from pathlib import Path
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    CLIPProcessor, 
    CLIPModel,
    TrainingArguments
)
from models.clip_trainer import CLIPTrainer
from models.llama_trainer import LLAMATrainer
from data_processor import YouTubeDataProcessor
from feature_extraction.text_processor import TextProcessor
from feature_extraction.image_processor import ImageProcessor
from feature_extraction.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeMultiModalTrainer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.models_path = self.base_path / 'models'
        self.output_path = self.base_path / 'training' / 'outputs'
        
        # 确保必要的目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化处理器
        self.data_processor = YouTubeDataProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

    def setup_clip_model(self):
        """设置和初始化CLIP模型"""
        logger.info("初始化CLIP模型...")
        
        # 加载CLIP模型和处理器
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 初始化训练器
        self.clip_trainer = CLIPTrainer(
            model=model,
            processor=processor,
            device=self.device
        )
        
        logger.info("CLIP模型设置完成")

    def setup_llama_model(self):
        """设置和初始化LLAMA2模型"""
        logger.info("初始化LLAMA2模型...")
        
        # 加载LLAMA2模型和分词器
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        
        # 初始化训练器
        self.llama_trainer = LLAMATrainer(
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )
        
        logger.info("LLAMA2模型设置完成")

    def train_clip(self, train_data, val_data, training_args):
        """训练CLIP模型"""
        logger.info("开始CLIP模型训练...")
        
        # 设置训练参数
        args = TrainingArguments(
            output_dir=str(self.output_path / "clip"),
            num_train_epochs=training_args.get('num_epochs', 3),
            per_device_train_batch_size=training_args.get('batch_size', 32),
            learning_rate=training_args.get('learning_rate', 5e-5),
            logging_dir=str(self.output_path / "clip" / "logs"),
        )
        
        # 开始训练
        self.clip_trainer.train(
            train_data=train_data,
            val_data=val_data,
            training_args=args
        )
        
        logger.info("CLIP模型训练完成")

    def train_llama(self, train_data, val_data, training_args):
        """训练LLAMA2模型"""
        logger.info("开始LLAMA2模型训练...")
        
        # 设置训练参数
        args = TrainingArguments(
            output_dir=str(self.output_path / "llama"),
            num_train_epochs=training_args.get('num_epochs', 3),
            per_device_train_batch_size=training_args.get('batch_size', 4),
            learning_rate=training_args.get('learning_rate', 1e-5),
            logging_dir=str(self.output_path / "llama" / "logs"),
            fp16=True,  # 使用混合精度训练
            gradient_accumulation_steps=4
        )
        
        # 开始训练
        self.llama_trainer.train(
            train_data=train_data,
            val_data=val_data,
            training_args=args
        )
        
        logger.info("LLAMA2模型训练完成")

    def process_youtube_data(self):
        """处理YouTube数据"""
        logger.info("开始处理YouTube数据...")
        
        # 加载和预处理数据
        raw_data = self.data_processor.load_data()
        
        # 提取特征
        text_features = self.text_processor.process(raw_data['titles'])
        image_features = self.image_processor.process(raw_data['thumbnails'])
        audio_features = self.audio_processor.process(raw_data['audio'])
        
        # 构建训练数据集
        train_data, val_data = self.data_processor.build_datasets(
            text_features,
            image_features,
            audio_features
        )
        
        logger.info("YouTube数据处理完成")
        return train_data, val_data

def main():
    # 训练参数
    clip_training_args = {
        'num_epochs': 5,
        'batch_size': 32,
        'learning_rate': 5e-5
    }
    
    llama_training_args = {
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 1e-5
    }
    
    try:
        # 初始化训练器
        trainer = YouTubeMultiModalTrainer()
        
        # 处理数据
        train_data, val_data = trainer.process_youtube_data()
        
        # 设置并训练CLIP模型
        trainer.setup_clip_model()
        trainer.train_clip(train_data, val_data, clip_training_args)
        
        # 设置并训练LLAMA2模型
        trainer.setup_llama_model()
        trainer.train_llama(train_data, val_data, llama_training_args)
        
        logger.info("所有模型训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()