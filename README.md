# ALPHAVANTAGE x JAINWIN

A multi-strategy algorithmic trading system developed for the Alphavantage Quantitative Hackathon, organized by Untrade x Quant Club. ZELTA leverages reinforcement learning, machine learning, and advanced technical analysis to create robust trading models, specifically tailored for BTC/USDT and ETH/USDT markets.

---

## 🧠 Overview of Strategies

### 📌 Strategy 1 - Q-Learning Reinforcement Agent

This strategy employs a model-free Q-Learning agent to dynamically learn optimal trading actions in a simulated trading environment using discrete state spaces derived from indicators like RSI, EMA, Aroon, and price change. The agent learns from interaction—without prior market knowledge—updating a Q-table through a reward-driven learning loop. It incorporates stop-loss logic for both long and short positions and adapts over time to maximize cumulative returns while managing risk via exploration-exploitation balance.

---

### 📌 Strategy 2 - Enhanced Technical Strategy

A pure technical analysis-based rule engine that combines momentum, volatility, and trend-following indicators for disciplined long and short entries. With strict entry filters like EMA crossover, RSI zones, ADX strength, and VWAP validation, it avoids noise and false breakouts. The strategy uses ATR-based SL/TP and trailing stop-loss to adaptively lock in profits and dynamically manage exits. It is ideal for structured and rule-based intraday or swing trading environments.

---

### 📌 Strategy 3 - ML Enhanced Hybrid BTC Strategy

A hybrid approach combining technical analysis with machine learning models—Random Forest, Decision Trees, and Neural Networks—to detect local tops/bottoms and confirm momentum shifts. It uses a wide feature set (MACD, ATR, Bollinger Bands, VWAP, etc.) and applies rigorous overfitting prevention techniques like model consensus filters, fallback logic, and SMA trend confirmation. The strategy only trades when technical and ML signals align, ensuring a high-confidence signal pipeline.

---

### 📌 Strategy 4 - BTC-ETH Regime-Adaptive Pair Strategy

This strategy exploits BTC-ETH correlation by using BTC signals to trade ETH in a pair-trading style. It integrates the Hurst exponent for regime detection, volatility filters via ATR, and indicator confirmation from RSI, Supertrend, and Bollinger Bands. The system adapts to trending vs. mean-reverting environments and dynamically adjusts entries and exits based on BTC regime shifts, making it a robust solution for arbitraging correlated assets.

---

## 📊 Backtesting & Performance

Each strategy has been rigorously backtested with metrics including:
- Sharpe & Sortino Ratios
- Win Rate
- Max Drawdown
- Time To Recover 
- Profitability Metrics
- Holding Periods
- Benchmark Comparisons

For full results check the report for each strategy.

