import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from strategy import QuantStrategy

def plot_performance(results):
    """绘制策略表现图表
    
    Args:
        results: 策略回测结果
    """
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
    plt.savefig('strategy_performance.png')
    plt.close()

def print_performance_metrics(results):
    """打印策略表现指标
    
    Args:
        results: 策略回测结果
    """
    print('\n策略表现指标:')
    print(f"年化收益率: {results['annual_return']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")

def main():
    # 策略参数
    params = {
        'market_data_path': 'market_data.csv',  # 行情数据文件路径
        'fundamental_data_path': 'fundamental_data.csv',  # 基本面数据文件路径
        'top_n': 20,  # 选股数量
        'holding_period': 20,  # 持仓周期
        'initial_capital': 1000000.0  # 初始资金
    }
    
    # 初始化策略
    strategy = QuantStrategy()
    
    # 运行策略
    print('正在运行策略...')
    results = strategy.run_strategy(**params)
    
    # 输出策略表现
    print_performance_metrics(results)
    
    # 绘制策略表现图表
    plot_performance(results)
    print('\n策略表现图表已保存为 strategy_performance.png')

if __name__ == '__main__':
    main()