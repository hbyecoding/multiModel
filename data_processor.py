import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime

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