import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Any
from textblob import TextBlob
import numpy as np
from collections import Counter

class LlamaModel:
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-hf'):
        """初始化LLaMA模型
        
        Args:
            model_name: 模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)
        
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """生成回复
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            
        Returns:
            生成的回复
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """分析内容
        
        Args:
            text: 输入文本
            
        Returns:
            内容分析结果
        """
        # 构建分析提示
        prompt = f"""请分析以下内容的主题、关键词和情感倾向：
        
        {text}
        
        分析结果："""
        
        # 生成分析结果
        analysis = self.generate_response(prompt)
        
        return {
            'raw_text': text,
            'analysis': analysis,
            'length': len(text)
        }
    
    def optimize_title(self, title: str, keywords: List[str]) -> Dict[str, Any]:
        """优化视频标题
        
        Args:
            title: 原始标题
            keywords: 关键词列表
            
        Returns:
            优化结果
        """
        # 构建优化提示
        prompt = f"""请基于以下信息优化YouTube视频标题：
        
        原始标题：{title}
        目标关键词：{', '.join(keywords)}
        
        优化后的标题："""
        
        # 生成优化标题
        optimized_title = self.generate_response(prompt)
        
        return {
            'original_title': title,
            'optimized_title': optimized_title,
            'keywords': keywords
        }
    
    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练步骤
        
        Args:
            input_ids: 输入ID
            labels: 标签
            
        Returns:
            训练损失
        """
        outputs = self.model(
            input_ids=input_ids,
            labels=labels
        )
        
        return {
            'loss': outputs.loss
        }
    
    def analyze_title_metrics(self, title: str) -> Dict[str, Any]:
        """分析标题指标
        
        Args:
            title: 输入标题
            
        Returns:
            标题分析指标
        """
        # 情感分析
        blob = TextBlob(title)
        sentiment = blob.sentiment
        
        # 词汇多样性
        words = title.split()
        unique_words = set(words)
        word_freq = Counter(words)
        
        # 计算指标
        metrics = {
            'sentiment_polarity': sentiment.polarity,  # 情感极性
            'sentiment_subjectivity': sentiment.subjectivity,  # 主观性
            'word_count': len(words),  # 词数
            'unique_word_ratio': len(unique_words) / len(words) if words else 0,  # 独特词比例
            'avg_word_freq': sum(word_freq.values()) / len(word_freq) if word_freq else 0  # 平均词频
        }
        
        return metrics
    
    def generate_diverse_titles(self, base_title: str, num_variants: int = 3) -> List[str]:
        """生成多样化标题变体
        
        Args:
            base_title: 基础标题
            num_variants: 变体数量
            
        Returns:
            标题变体列表
        """
        prompts = [
            f"请用更吸引人的方式重写这个标题，保持核心含义：{base_title}",
            f"请用更专业的语言重写这个标题：{base_title}",
            f"请用更有情感共鸣的方式重写这个标题：{base_title}"
        ]
        
        variants = []
        for prompt in prompts[:num_variants]:
            variant = self.generate_response(prompt)
            variants.append(variant)
            
        return variants