# 多模态量化交易系统

这是一个基于多模态数据的量化交易系统，集成了市场数据、基本面数据、新闻数据和社交媒体数据的分析功能。

## 系统架构

```
multiModel/
├── data/                      # 数据目录
│   ├── market_data/          # 市场数据
│   ├── fundamental_data/     # 基本面数据
│   ├── news_data/           # 新闻数据
│   └── social_media_data/   # 社交媒体数据
├── models/                   # 模型目录
├── training/                 # 训练脚本
├── evaluation/              # 评估结果
├── feature_extraction/      # 特征工程
├── tests/                   # 测试用例
├── strategy.py             # 策略实现
├── data_processor.py       # 数据处理
├── data_setup.py          # 数据设置
└── main.py                # 主程序
```

## 功能特点

1. **多源数据集成**
   - 市场数据：股票价格、交易量等
   - 基本面数据：财务报表、公司信息等
   - 新闻数据：公司新闻、行业新闻等
   - 社交媒体数据：社交媒体情感分析

2. **自动化数据收集**
   - 使用 yfinance 获取市场和基本面数据
   - 支持自动更新和数据同步

3. **策略实现**
   - 多因子选股策略
   - 技术指标分析
   - 基本面分析
   - 动态仓位管理

4. **性能评估**
   - 回测系统
   - 性能指标计算
   - 可视化分析

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 设置数据源：
   ```python
   # 在 main.py 中配置股票列表
   symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
   ```

2. 运行系统：
   ```bash
   python main.py
   ```

3. 查看结果：
   - 策略表现图表将保存在 evaluation 目录
   - 日志信息将实时显示在控制台

## 配置参数

主要参数在 main.py 中配置：

- `top_n`：选股数量
- `holding_period`：持仓周期
- `initial_capital`：初始资金

## 开发计划

- [ ] 添加深度学习模型支持
- [ ] 集成更多数据源
- [ ] 优化策略性能
- [ ] 添加实时交易支持

## 贡献指南

欢迎提交 Pull Request 或提出 Issue。

## 许可证

MIT License
