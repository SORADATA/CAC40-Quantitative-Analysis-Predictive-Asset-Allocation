# 📊 CAC40 Portfolio Optimization: ML-Based Stock Selection & Asset Allocation

> **Master 2 - Portfolio Management | Université de Lorraine**  
> Advanced quantitative analysis combining Machine Learning, clustering, and Modern Portfolio Theory for CAC40 stocks

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 🎯 Project Overview

This project develops an **intelligent portfolio management system** for CAC40 stocks by combining:

1. **Unsupervised Learning** (clustering) to identify stock behavioral patterns
2. **Supervised Learning** (XGBoost) to predict returns and directions
3. **Quantitative Finance** (Mean-Variance Optimization) to maximize risk-adjusted returns

### Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Classification AUC** | **95.29%** | Excellent directional prediction (up/down) |
| **Clustering Quality** | Silhouette Score > 0.6 | Well-separated momentum groups |
| **Portfolio Sharpe** | Target > 1.0 | Risk-adjusted outperformance vs. benchmark |

---

## 🚀 Methodology

### 1. **Feature Engineering & Data Pipeline**


**Key Features (16 total)**:
- Technical: RSI, MACD, ATR, Bollinger Bands
- Volume: Euro Volume, Garman-Klass Volatility
- Returns: 1M, 2M, 3M, 6M rolling returns
- Factors: Fama-French 5-Factor Model (Mkt-RF, SMB, HML, RMW, CMA)
- Cluster: Behavioral group assignment

---

### 2. **Stock Selection via Machine Learning**

#### **Phase A: Clustering (K-Means)**

Segment stocks into 4 behavioral groups based on RSI momentum:

| Cluster | RSI Range | Profile | Strategy |
|---------|-----------|---------|----------|
| **0** | ~30 | Oversold | Contrarian buy |
| **1** | ~45 | Neutral-Low | Hold/Avoid |
| **2** | ~55 | Neutral-High | Monitor |
| **3** | ~70 | **Momentum** 🔥 | **Target for portfolio** |

**Result**: Cluster 3 stocks show **+292% avg return** vs. overall market (+10.08% vs 2.57%)

---

#### **Phase B: XGBoost Classification** (Direction Prediction)

Predict if `return_1m > 0` (stock goes up/down):



