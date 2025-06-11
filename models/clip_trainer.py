import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
import logging

logger = logging.getLogger(__name__)

class CLIPTrainer:
    def __init__(self, model, processor, device):
        """初始化CLIP训练器
        
        Args:
            model: CLIP模型实例
            processor: CLIP处理器实例
            device: 训练设备
        """
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def compute_loss(self, image_features, text_features):
        """计算对比损失
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            
        Returns:
            total_loss: 总损失
        """
        # 计算相似度矩阵
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(len(image_features), device=self.device)
        
        # 计算双向对比损失
        loss_i2t = self.criterion(logits_per_image, labels)
        loss_t2i = self.criterion(logits_per_text, labels)
        
        # 总损失为两个方向损失的平均
        total_loss = (loss_i2t + loss_t2i) / 2
        return total_loss
    
    def train_epoch(self, train_loader, optimizer):
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            
        Returns:
            epoch_loss: 当前epoch的平均损失
        """
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 获取图像和文本数据
            images = batch['images'].to(self.device)
            texts = batch['texts']
            
            # 处理输入
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # 计算损失
            loss = self.compute_loss(image_features, text_features)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_loader)
        return epoch_loss
    
    def evaluate(self, val_loader):
        """评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            val_loss: 验证损失
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                texts = batch['texts']
                
                inputs = self.processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                loss = self.compute_loss(image_features, text_features)
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        return val_loss
    
    def train(self, train_data, val_data, training_args):
        """训练模型
        
        Args:
            train_data: 训练数据集
            val_data: 验证数据集
            training_args: 训练参数
        """
        # 创建数据加载器
        train_loader = DataLoader(
            train_data,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=training_args.per_device_train_batch_size
        )
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate
        )
        
        # 训练循环
        for epoch in range(int(training_args.num_train_epochs)):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # 评估
            val_loss = self.evaluate(val_loader)
            
            # 记录进度
            logger.info(
                f"Epoch {epoch+1}/{training_args.num_train_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
        # 保存模型
        self.model.save_pretrained(training_args.output_dir)
        self.processor.save_pretrained(training_args.output_dir) 