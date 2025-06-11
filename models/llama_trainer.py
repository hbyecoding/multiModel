import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LLAMATrainer:
    def __init__(self, model, tokenizer, device):
        """初始化LLAMA训练器
        
        Args:
            model: LLAMA模型实例
            tokenizer: LLAMA分词器实例
            device: 训练设备
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 设置特殊token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_inputs(self, batch):
        """准备模型输入
        
        Args:
            batch: 数据批次
            
        Returns:
            model_inputs: 模型输入
        """
        # 获取输入文本和标签
        input_texts = batch['input_texts']
        target_texts = batch['target_texts']
        
        # 编码输入
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 编码目标
        labels = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 设置labels
        inputs['labels'] = labels['input_ids']
        
        return inputs
    
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
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # 准备输入
            inputs = self.prepare_inputs(batch)
            
            # 前向传播
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = total_loss / len(train_loader)
        return epoch_loss
    
    def evaluate(self, val_loader):
        """评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            val_loss: 验证损失
            perplexity: 困惑度
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = self.prepare_inputs(batch)
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(val_loss))
        
        return val_loss, perplexity.item()
    
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
        best_val_loss = float('inf')
        
        for epoch in range(int(training_args.num_train_epochs)):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # 评估
            val_loss, perplexity = self.evaluate(val_loader)
            
            # 记录进度
            logger.info(
                f"Epoch {epoch+1}/{training_args.num_train_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Perplexity: {perplexity:.2f}"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(training_args.output_dir)
                self.tokenizer.save_pretrained(training_args.output_dir)
                logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    def generate_text(self, prompt, max_length=100):
        """生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            
        Returns:
            generated_text: 生成的文本
        """
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text 