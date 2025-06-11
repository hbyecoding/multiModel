import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from strategy import QuantStrategy
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_path = Path(__file__).parent / 'data'
        self.market_data_path = self.base_path / 'market_data'
        self.fundamental_data_path = self.base_path / 'fundamental_data'
        self.news_data_path = self.base_path / 'news_data'
        self.social_data_path = self.base_path / 'social_media_data'

    def download_market_data(self, symbols, start_date, end_date):
        """下载市场数据"""
        logger.info(f"开始下载市场数据: {symbols}")
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    df.to_csv(self.market_data_path / f"{symbol}_market_data.csv")
                    logger.info(f"成功下载 {symbol} 的市场数据")
            except Exception as e:
                logger.error(f"下载 {symbol} 数据时出错: {str(e)}")

    def download_fundamental_data(self, symbols):
        """下载基本面数据"""
        logger.info("开始下载基本面数据")
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                # 获取财务报表数据
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cash_flow
                
                # 合并财务数据
                fundamental_data = {
                    'financials': financials.to_dict(),
                    'balance_sheet': balance_sheet.to_dict(),
                    'cash_flow': cash_flow.to_dict()
                }
                
                # 保存数据
                with open(self.fundamental_data_path / f"{symbol}_fundamental.json", 'w') as f:
                    json.dump(fundamental_data, f)
                logger.info(f"成功下载 {symbol} 的基本面数据")
            except Exception as e:
                logger.error(f"下载 {symbol} 基本面数据时出错: {str(e)}")

def plot_performance(results, save_path):
    """绘制策略表现图表"""
    plt.figure(figsize=(15, 10))
    
    # 绘制累积收益曲线
    plt.subplot(2, 1, 1)
    results['cumulative_returns'].plot()
    plt.title('策略累积收益')
    plt.grid(True)
    
    # 绘制每日收益分布
    plt.subplot(2, 1, 2)
    results['daily_returns'].hist(bins=50)
    plt.title('每日收益分布')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_performance_metrics(results):
    """打印策略表现指标"""
    logger.info('\n策略表现指标:')
    logger.info(f"年化收益率: {results['annual_return']*100:.2f}%")
    logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤: {results['max_drawdown']*100:.2f}%")

def main():
    # 创建数据收集器
    collector = DataCollector()
    
    # 设置参数
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # 示例股票
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 下载数据
    collector.download_market_data(symbols, start_date, end_date)
    collector.download_fundamental_data(symbols)
    
    # 策略参数
    params = {
        'market_data_path': str(collector.market_data_path),
        'fundamental_data_path': str(collector.fundamental_data_path),
        'top_n': 3,  # 选股数量
        'holding_period': 20,  # 持仓周期
        'initial_capital': 1000000.0  # 初始资金
    }
    
    # 初始化策略
    strategy = QuantStrategy()
    
    try:
        # 运行策略
        logger.info('正在运行策略...')
        results = strategy.run_strategy(**params)
        
        # 输出策略表现
        print_performance_metrics(results)
        
        # 绘制策略表现图表
        plot_path = Path(__file__).parent / 'evaluation' / 'strategy_performance.png'
        plot_performance(results, plot_path)
        logger.info(f'\n策略表现图表已保存为 {plot_path}')
        
    except Exception as e:
        logger.error(f"策略运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()