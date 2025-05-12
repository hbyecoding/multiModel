import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
from data_processor import DataProcessor

class QuantStrategy:
    def __init__(self):
        """初始化量化策略"""
        self.data_processor = DataProcessor()
        
    def calculate_factor_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子得分
        
        Args:
            df: 包含因子数据的DataFrame
            
        Returns:
            添加因子得分后的DataFrame
        """
        # 技术面因子得分
        df['TECH_SCORE'] = 0
        # 均线多头排列得分
        df.loc[(df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']), 'TECH_SCORE'] += 1
        # MACD金叉得分
        df.loc[df['MACD'] > df['Signal_Line'], 'TECH_SCORE'] += 1
        # RSI超卖反弹得分
        df.loc[(df['RSI'] > 30) & (df['RSI'] < 70), 'TECH_SCORE'] += 1
        
        # 基本面因子得分
        # 市值因子 - 偏好中小市值
        df['SIZE_SCORE'] = -df['LN_MKT_CAP'].rank(pct=True)
        
        # 价值因子 - 偏好低估值
        df['VALUE_SCORE'] = -(df['PE'].rank(pct=True) + df['PB'].rank(pct=True)) / 2
        
        # 成长因子 - 偏好高成长
        df['GROWTH_SCORE'] = (df['REVENUE_YOY'].rank(pct=True) + 
                            df['PROFIT_YOY'].rank(pct=True)) / 2
        
        # 综合得分
        df['TOTAL_SCORE'] = (df['TECH_SCORE'].rank(pct=True) +
                            df['SIZE_SCORE'] +
                            df['VALUE_SCORE'] +
                            df['GROWTH_SCORE']) / 4
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, 
                        top_n: int = 20,
                        holding_period: int = 20) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            df: 包含因子得分的DataFrame
            top_n: 选股数量
            holding_period: 持仓周期（交易日）
            
        Returns:
            包含交易信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        
        # 按日期循环
        for date in df.index.unique():
            # 获取当日数据
            daily_data = df.loc[date]
            
            # 选择得分最高的N只股票
            top_stocks = daily_data.nlargest(top_n, 'TOTAL_SCORE')
            
            # 生成交易信号
            signals.loc[date, top_stocks.index] = 1
        
        # 填充其他位置为0
        signals = signals.fillna(0)
        
        return signals
    
    def calculate_positions(self, signals: pd.DataFrame,
                          initial_capital: float = 1000000.0) -> pd.DataFrame:
        """计算持仓仓位
        
        Args:
            signals: 交易信号DataFrame
            initial_capital: 初始资金
            
        Returns:
            持仓仓位DataFrame
        """
        # 计算每只股票的目标持仓金额
        stock_count = (signals == 1).sum(axis=1)
        position_sizes = initial_capital / stock_count
        
        # 生成持仓矩阵
        positions = signals.multiply(position_sizes, axis=0)
        
        return positions
    
    def backtest(self, positions: pd.DataFrame, 
                 price_data: pd.DataFrame,
                 transaction_cost: float = 0.003) -> Dict[str, Any]:
        """回测策略表现
        
        Args:
            positions: 持仓仓位DataFrame
            price_data: 价格数据DataFrame
            transaction_cost: 交易成本率
            
        Returns:
            回测结果统计
        """
        # 计算每日收益率
        daily_returns = price_data.pct_change()
        portfolio_returns = (positions.shift(1) * daily_returns).sum(axis=1)
        
        # 计算换手率和交易成本
        position_changes = positions.diff().abs()
        turnover = position_changes.sum(axis=1) / positions.sum(axis=1)
        transaction_costs = turnover * transaction_cost
        
        # 计算净收益率
        net_returns = portfolio_returns - transaction_costs
        
        # 计算累积收益
        cumulative_returns = (1 + net_returns).cumprod()
        
        # 计算年化收益率
        annual_return = (cumulative_returns.iloc[-1] ** 
                        (252 / len(cumulative_returns)) - 1)
        
        # 计算夏普比率
        daily_rf = 0.03 / 252  # 假设无风险利率为3%
        excess_returns = net_returns - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # 计算最大回撤
        cumulative_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / cumulative_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns,
            'daily_returns': net_returns
        }
    
    def run_strategy(self, 
                     market_data_path: str,
                     fundamental_data_path: str,
                     top_n: int = 20,
                     holding_period: int = 20,
                     initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """运行完整的策略流程
        
        Args:
            market_data_path: 行情数据文件路径
            fundamental_data_path: 基本面数据文件路径
            top_n: 选股数量
            holding_period: 持仓周期
            initial_capital: 初始资金
            
        Returns:
            策略回测结果
        """
        # 加载和处理数据
        market_data = self.data_processor.load_market_data(market_data_path)
        fundamental_data = self.data_processor.load_fundamental_data(fundamental_data_path)
        
        # 计算技术指标和基本面因子
        market_data = self.data_processor.calculate_technical_indicators(market_data)
        fundamental_data = self.data_processor.calculate_fundamental_factors(fundamental_data)
        
        # 合并数据
        df = self.data_processor.merge_data(market_data, fundamental_data)
        
        # 计算因子得分
        df = self.calculate_factor_scores(df)
        
        # 生成交易信号
        signals = self.generate_signals(df, top_n, holding_period)
        
        # 计算持仓
        positions = self.calculate_positions(signals, initial_capital)
        
        # 回测策略
        results = self.backtest(positions, df['close'].unstack())
        
        return results