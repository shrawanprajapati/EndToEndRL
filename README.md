# QuantPilot AI 🤖📈

> **An Intelligent Reinforcement Learning Trading Platform for Cryptocurrency Markets**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter)](https://flutter.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-PPO-orange)](https://stable-baselines3.readthedocs.io)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-purple)](https://gymnasium.farama.org)
[![IITISoC](https://img.shields.io/badge/IITISoC-2026-red)](https://iiti.ac.in)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Team Details](#-team-details)
3. [Problem Statement](#-problem-statement)
4. [System Architecture](#-system-architecture)
5. [Technology Stack](#-technology-stack)
6. [Project Structure](#-project-structure)
7. [Core Components](#-core-components)
   - [Reinforcement Learning Engine](#71-reinforcement-learning-engine)
   - [Trading Environment (MDP)](#72-trading-environment-mdp)
   - [PPO Agent Training](#73-ppo-agent-training)
   - [FastAPI Backend](#74-fastapi-backend)
   - [Flutter Frontend](#75-flutter-frontend)
   - [Backtesting Engine](#76-backtesting-engine)
   - [Sentiment Analysis Pipeline](#77-sentiment-analysis-pipeline)
   - [LLM Analyst Module](#78-llm-analyst-module)
8. [Feature Engineering](#-feature-engineering)
9. [API Reference](#-api-reference)
10. [Validation Results](#-validation-results)
11. [Frontend Dashboard](#-frontend-dashboard)
12. [Installation & Setup](#-installation--setup)
13. [Usage Guide](#-usage-guide)
14. [Known Issues & Roadmap](#-known-issues--roadmap)
15. [Contributing](#-contributing)

---

## 🚀 Project Overview

**QuantPilot AI** is a production-grade, full-stack algorithmic trading platform built on top of Model-Free Reinforcement Learning (RL). Unlike traditional systems that rely on hard-coded indicator thresholds or static statistical rules, QuantPilot AI trains an intelligent policy network to learn dynamic risk-reward balancing mechanisms natively from historical market data.

The platform is an end-to-end vertical slice — data ingestion, feature engineering, custom Gymnasium simulation environment, PPO agent training via Stable-Baselines3, a FastAPI orchestration backend, and a polished Flutter dashboard — all wired together into a single working product.

### Key Highlights

| Metric | Value |
|--------|-------|
| **Agent Mean Return** (20 episodes) | **+2.03%** |
| **Buy & Hold Return** (same period) | **-11.07%** |
| **Net Alpha Generated** | **+13.09%** |
| **Agent Win Rate** | **80.0%** |
| **Beat Rate vs Passive** | **85.0%** |
| **Mean Sharpe Ratio** | **0.9351** |
| **Mean Calmar Ratio** | **9.3630** |
| **Mean Max Drawdown** | **0.89%** |

---

## 👥 Team Details

**Institution:** Indian Institute of Technology (IIT) Indore
**Domain:** Finance & Analytics
**Project ID:** FA-ADV-01
**Competition:** IITISoC 2026

| Name | Roll Number | GitHub | Key Skills |
|------|-------------|--------|-----------|
| **Shrawan Kumar Prajapati** *(Lead)* | 250008033 | [@shrawanprajapati](https://github.com/shrawanprajapati) | Flutter, Python, Pandas/NumPy, LangGraph, SQL, APIs, LLM, Git |
| Aryan Bhupariya | 250008004 | [@Aryanbhupariya](https://github.com/Aryanbhupariya) | Python, Pandas, NumPy, Reinforcement Learning, Financial Data Analysis |
| Adwit Tiwari | 250004004 | [@Adwit](https://github.com/Adwit) | Python, Dart, Flutter, ReactNative, Node.js, Firebase, SQL, RAG, APIs, Finance & Trading |
| Smita Singh | 250008035 | [@smitasingh25](https://github.com/smitasingh25) | Python, NumPy, Pandas, APIs, RAG |
| Atharva Kulkarni | 250004008 | [@AtharvaKulkarni-17](https://github.com/AtharvaKulkarni-17) | Python, Dart, C++, Pandas, Matplotlib |

---

## 🎯 Problem Statement

Cryptocurrency markets are **highly volatile, non-stationary, and noisy**. Traditional trading strategies — static indicator rules or supervised models — frequently fail to adapt when:

- **Market regimes shift** (e.g., transitioning from a trending bull market to choppy, sideways, high-volatility action)
- **False breakout signals** proliferate from lagging indicators
- **Structural statistical assumptions** (stationarity, normality) break down under extreme macro conditions

Reinforcement learning offers a reward-driven paradigm where an agent learns to take sequential actions based on long-term objectives, without assuming any fixed market structure. QuantPilot AI exploits this by:

- Mapping historical market vectors into non-linear observation spaces via Proximal Policy Optimization (PPO)
- Teaching the policy to **learn defensive positioning** natively rather than encoding stop-loss rules by hand
- Integrating **regime detection** (Hidden Markov Model) to automatically adapt risk posture when the market structure is unclassifiable

---

## 🏗 System Architecture

The system is organized into three fully decoupled structural tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     FLUTTER CLIENT (UI)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Market   │  │ News     │  │ Live Sim │  │ Backtest     │   │
│  │ Tab      │  │ Tab      │  │ Tab      │  │ Tab          │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
└───────┼─────────────┼──────────────┼───────────────┼───────────┘
        │             │              │               │
   WS /candles  REST /regime   WS /stream    REST /run_backtest
        │             │              │               │
┌───────▼─────────────▼──────────────▼───────────────▼───────────┐
│                    FASTAPI BACKEND                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Asynchronous Core Routers (asyncio)           │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  TradingService   │                        │
│                    └─────────┬─────────┘                        │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     RL / QUANT ENGINE                            │
│  ┌─────────────────────────┐   ┌──────────────────────────┐    │
│  │   TradingEnvironment    │   │   VectorBT Backtester    │    │
│  │   (Gymnasium Custom)    │   │   (Classical Strategies) │    │
│  └─────────────┬───────────┘   └──────────────────────────┘    │
│                │                                                 │
│  ┌─────────────▼───────────┐                                    │
│  │  PPO Model /            │                                    │
│  │  StableBaselines3       │                                    │
│  └─────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Transport Specification

| Endpoint | Protocol | Direction | Purpose |
|----------|----------|-----------|---------|
| `/api/v1/ws/candles` | WebSocket | Server → Client | Async live market tick streaming to chart canvas |
| `/api/v1/stream` | WebSocket | Bi-directional | Live allocation & portfolio telemetry |
| `/api/v1/run_backtest` | REST HTTP | Request/Response | Batch historical evaluation with strategy params |
| `/api/v1/agent/chat` | REST HTTP | Request/Response | LLM contextual analytical framing via Gemini |
| `/api/v1/regime` | REST HTTP | Response | Current HMM market regime classification |

---

## 🛠 Technology Stack

### Backend & AI
| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web Framework | FastAPI + Uvicorn |
| Async Runtime | asyncio / WebSockets |
| RL Framework | Stable-Baselines3 (PPO) |
| RL Environment | Gymnasium (Custom) |
| Backtesting | VectorBT |
| Data Processing | Pandas, NumPy |
| Volatility Modelling | ARCH / GARCH |
| Regime Detection | Hidden Markov Model (hmmlearn) |
| LLM Integration | Google Gemini API |
| Data Source | Binance API (live), CSV (historical) |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | Flutter (Dart) |
| State Management | Riverpod (code-generation) |
| Charting | Custom Canvas Painters |
| HTTP Client | Dio |
| WebSocket | dart:io WebSocket |

### DevOps & Tooling
| Layer | Technology |
|-------|-----------|
| Version Control | Git / GitHub |
| Training Monitoring | TensorBoard |
| Environment | Python venv / conda |

---

## 📁 Project Structure

```
QuantPilotAI/
│
├── RLModel/                        # Core RL Engine
│   ├── data/
│   │   ├── raw/                    # Raw OHLCV CSV files (BTC/USDT)
│   │   └── processed/
│   │       ├── train.csv           # ~4.5 years training data
│   │       └── test.csv            # Final 6 months holdout
│   │
│   ├── models/
│   │   ├── trading_env.py          # Custom Gymnasium environment
│   │   ├── train_final.py          # PPO training entry point
│   │   ├── evaluate.py             # 20-episode validation harness
│   │   ├── trading_service.py      # Model inference service
│   │   └── saved/
│   │       ├── ppo_trading_model.zip   # Trained PPO weights
│   │       └── eval_results.json       # Last evaluation output
│   │
│   └── backtest/
│       └── backtest.py             # VectorBT + RL backtest router
│
├── backend/
│   └── main.py                     # FastAPI server, routers, WebSocket handlers
│
├── rl_trading_ui/                  # Flutter client application
│   ├── lib/
│   │   ├── main.dart
│   │   ├── tabs/
│   │   │   ├── market_tab.dart     # Candlestick chart + overlay indicators
│   │   │   ├── news_tab.dart       # Sentiment feed aggregator
│   │   │   ├── live_sim_tab.dart   # Virtual portfolio live simulation
│   │   │   ├── backtest_tab.dart   # Backtest parameter UI + results
│   │   │   └── analyst_tab.dart    # LLM-driven diagnostic assistant
│   │   └── providers/
│   │       └── state_providers.dart  # Riverpod state definitions
│   └── pubspec.yaml
│
├── requirements.txt
└── README.md
```

---

## 🔧 Core Components

### 7.1 Reinforcement Learning Engine

The RL engine is the heart of QuantPilot AI. It uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3 with a custom multi-layer perceptron actor-critic architecture trained entirely on historical BTC/USDT OHLCV data.

**Training Data Split:**
- **Train set:** ~4.5 years of historical BTC/USDT data
- **Test set:** Final 6 months (strict holdout — never seen during training)

**PPO Clipped Objective:**
```
L^CLIP(θ) = E_t [ min( r_t(θ)·Â_t,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]

where:
  r_t(θ)  = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   (probability ratio)
  ε       = 0.2                                   (clip parameter)
  Â_t     = advantage estimate at time t
```

The clip parameter (ε = 0.2) prevents the network weights from changing too drastically during any single training update — critical for stability on highly volatile financial data.

---

### 7.2 Trading Environment (MDP)

The custom Gymnasium environment formalizes the trading problem as a **Markov Decision Process** ⟨S, A, P, R, γ⟩:

#### Observation Space (S)
```python
s_t = [ X_t, Pos_t, UnrealizedROI_t, CashRatio_t ]

X_t = [
    δ_EMA20,          # Normalized EMA20 delta
    RSI_t,            # Relative Strength Index (14-period)
    MACD_signal,      # MACD signal line
    BB_position_t,    # (Close - LowerBand) / (UpperBand - LowerBand)
]
```

#### Action Space (A)
```
A = {
    0 → Liquidate / Short   (reduce/close position),
    1 → Hold / Maintain     (keep current allocation),
    2 → Buy / Long          (increase allocation)
}
```

#### Step Logic (`trading_env.py`)
```python
def step(self, action):
    current_price = self.price_series[self.current_step]

    # Apply transaction cost on action change (friction penalty)
    fee_penalty = 0.0
    if action != self.last_action:
        fee_penalty = self.capital_balance * self.transaction_fee_rate
        self.capital_balance -= fee_penalty

    # Execute portfolio rebalancing
    self._execute_portfolio_rebalancing(action, current_price)

    # Advance and check terminal conditions
    self.current_step += 1
    terminated = self.current_step >= len(self.price_series) - 1
    if self.portfolio_value <= self.liquidation_floor:
        terminated = True

    reward = self._calculate_step_reward(fee_penalty)
    return self._get_observation(), reward, terminated, False, self._get_info()
```

#### Reward Function
```
R_t = ΔPortfolioValue_t  −  (α × Drawdown_t)  −  β · 𝟙[a_t ≠ a_{t-1}]

where:
  α  = drawdown aversion scalar (scales down reward during severe drawdowns)
  β  = turnover friction penalty (discourages unnecessary action switching)
```

---

### 7.3 PPO Agent Training

**Training entry point:** `RLModel/models/train_final.py`

Key training configuration:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: TradingEnvironment(train_df)])

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,          # ε for PPO clipping
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=500_000)
model.save("models/saved/ppo_trading_model")
```

Monitor training in TensorBoard:
```bash
tensorboard --logdir ./logs/
```

---

### 7.4 FastAPI Backend

**Entry point:** `backend/main.py`

The backend uses asyncio coroutines and WebSocket handlers to serve the Flutter client without blocking model inference threads.

```python
# Example WebSocket candle streaming endpoint
@app.websocket("/api/v1/ws/candles")
async def websocket_candles(websocket: WebSocket):
    await websocket.accept()
    while True:
        candle_data = await trading_service.get_latest_candle()
        await websocket.send_json(candle_data)
        await asyncio.sleep(1)

# Example REST backtest endpoint
@app.post("/api/v1/run_backtest")
async def run_backtest(params: BacktestParams):
    if params.strategy == "rl_model":
        return await trading_service.run_rl_backtest(params)
    else:
        return await trading_service.run_vectorbt_backtest(params)
```

The `TradingService` coordinator extracts row vectors from localized `.csv` frames or live WebSocket streams and prepares observation vectors for the underlying AI model.

---

### 7.5 Flutter Frontend

State management uses **Riverpod** with code-generated providers:

```dart
// Core state providers
final backtestFromDateProvider = StateProvider<DateTime>(
    (ref) => DateTime(2024, 1, 1));
final backtestToDateProvider = StateProvider<DateTime>(
    (ref) => DateTime(2026, 6, 1));
final activeStrategyProvider = StateProvider<String>(
    (ref) => 'rl_model');

class BacktestSelector extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final strategy = ref.watch(activeStrategyProvider);
    return DropdownButton<String>(
      value: strategy,
      onChanged: (val) =>
          ref.read(activeStrategyProvider.notifier).state = val!,
      items: ['sma', 'ema', 'rsi', 'macd', 'rl_model'].map(...),
    );
  }
}
```

**Design Language:** Dark-mode glassmorphism
- Primary background: Jet-black (`#0A0A0A`) with midnight-blue accents
- Active status: Vibrant teal (`#00FAC6`)
- Risk alerts: Amber
- Drawdown indicators: Crimson
- Card panels: Semi-transparent with backdrop blur filters

---

### 7.6 Backtesting Engine

The platform uses a **hybrid dual-engine** approach:

```
Incoming Request: { "strategy": "rl_model" or "sma/ema/rsi/macd" }
                              │
              ┌───────────────┴───────────────┐
              │                               │
     strategy == "rl_model"          strategy == classical
              │                               │
              ▼                               ▼
   [RL Custom Loop Engine]         [VectorBT Matrix Engine]
   Sequential Step Traversal       Array-based Vectorized Math
   O(N) Time Complexity            O(1) Execution
   Simulates Memory & State        Static Signal Evaluation
   Tracks Transaction Costs        Fast Batch Processing
```

**Classical strategies supported:** `sma`, `ema`, `rsi`, `macd`

---

### 7.7 Sentiment Analysis Pipeline

The News Tab aggregates live articles from **CoinDesk, CoinTelegraph, Bloomberg, and Reuters** and passes them through an NLP inference pipeline:

**Bullish example:** High institutional inflow coverage → **67% positive score** → Market classified as BULLISH
**Bearish example:** Mixed/negative coverage → **30% positive score** → Market classified as BEARISH → automatic risk-posture tightening

Each article is individually labelled **Positive / Neutral / Negative**, and the aggregate score drives the system's overall directional conviction.

---

### 7.8 LLM Analyst Module

**Tab:** `analyst_tab.dart` → REST `/api/v1/agent/chat` → Gemini LLM

When the Hidden Markov Model registers an **"Unknown"** market regime, the Analyst automatically generates a risk-averse protocol:

> *"The HMM Market Regime is currently Unknown. Our plan is to operate under a heightened state of caution and reduced directional conviction... the system defaults to a risk-averse posture..."*

Automated response priority matrix:
1. **Reduced Exposure** — Maintain positions significantly below the 80% maximum cap
2. **Capital Preservation Focus** — Smaller position sizes, tighter stop-loss controls
3. **Intensified Monitoring** — Increase analysis frequency across all data streams

---

## 📊 Feature Engineering

The following technical indicators are computed and used as state inputs and chart overlays:

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| EMA20 | 20-period | Short-term trend tracking |
| EMA50 | 50-period | Medium-term trend tracking |
| SMA200 | 200-period | Long-term trend / support-resistance |
| RSI | 14-period | Momentum / overbought-oversold |
| MACD | (12, 26, 9) | Trend momentum + signal crossover |
| Bollinger Bands | (20, 2σ) | Volatility channels + band position |
| VWAP | Daily | Volume-weighted fair value |
| ATR | 14-period | True volatility range |
| StochRSI | 14-period | Stochastic oscillator of RSI |
| GARCH Volatility | Rolling | Statistical regime volatility estimation |
| HMM Regime | Hidden states | Market regime classification |

---

## 🌐 API Reference

### WebSocket Endpoints

**`WS /api/v1/ws/candles`**
```json
// Server → Client (every tick)
{
  "timestamp": 1719830400,
  "open": 62150.0,
  "high": 62480.0,
  "low": 61900.0,
  "close": 62310.0,
  "volume": 1523.45
}
```

**`WS /api/v1/stream`**
```json
// Bi-directional live simulation telemetry
{
  "allocation": 0.31,
  "portfolio_value": 9424.98,
  "drawdown": 0.0579,
  "trades": 0,
  "latency_ms": 0
}
```

### REST Endpoints

**`POST /api/v1/run_backtest`**
```json
// Request
{
  "strategy": "rl_model",
  "dataset": "TEST",
  "from_date": "2024-01-01",
  "to_date": "2026-06-01"
}

// Response
{
  "total_return": -0.0909,
  "sharpe_ratio": -2.48,
  "calmar_ratio": -2.79,
  "max_drawdown": 0.1053,
  "win_rate": 0.0,
  "trades": 725
}
```

**`POST /api/v1/agent/chat`**
```json
// Request
{ "message": "What is our plan for this HMM Market Regime?" }

// Response
{ "response": "The HMM Market Regime is currently Unknown..." }
```

**`GET /api/v1/regime`**
```json
{
  "regime": "Unknown",
  "volatility": 0.0150,
  "volatility_label": "LOW",
  "drawdown": -0.6860,
  "max_exposure": 0.80
}
```

---

## ✅ Validation Results

Full evaluation output from `evaluate.py` across 20 held-out test episodes:

```
[EVALUATION] Loading dataset: data/processed/test.csv
[EVALUATION] Running 20 episodes...

Episode  1/20  return=  2.82%  buy&hold=-23.31%  alpha=+26.13%  max_drawdown= 0.80%
Episode  2/20  return=  2.95%  buy&hold=-23.49%  alpha=+26.44%  max_drawdown= 0.80%
Episode  3/20  return=  3.04%  buy&hold=-13.06%  alpha=+16.10%  max_drawdown= 0.80%
Episode  4/20  return=  2.95%  buy&hold=-18.47%  alpha=+21.42%  max_drawdown= 0.80%
Episode  5/20  return=  3.10%  buy&hold= -3.19%  alpha= +6.29%  max_drawdown= 0.93%
Episode  6/20  return= -0.98%  buy&hold=  8.49%  alpha= -9.47%  max_drawdown= 0.98%
Episode  7/20  return= -0.98%  buy&hold= 15.44%  alpha=-16.42%  max_drawdown= 0.98%
Episode  8/20  return=  3.15%  buy&hold=-10.52%  alpha=+13.67%  max_drawdown= 0.80%
Episode  9/20  return=  3.25%  buy&hold= -2.17%  alpha= +5.41%  max_drawdown= 0.80%
Episode 10/20  return=  0.21%  buy&hold= -0.49%  alpha= +0.70%  max_drawdown= 1.10%
...
Episode 20/20  return=  2.95%  buy&hold=-18.68%  alpha=+21.63%  max_drawdown= 0.80%

-------- Backtest summary --------
Episodes evaluated        : 20
Mean return               : 2.03%
Std of returns            : 1.66%
Best episode              : 3.25%
Worst episode             : -1.27%
Win rate (agent > 0)      : 80.0%
Mean max drawdown         : 0.89%
Worst-case drawdown       : 1.71%
Mean Sharpe ratio         : 0.9351
Mean Sortino ratio        : 0.4631
Mean Calmar ratio         : 9.3630
Mean Omega ratio          : 1.8126

-------- Alpha vs buy & hold -----
Mean buy-and-hold return  : -11.07%
Mean alpha                : +13.09%
Std of alpha              : 12.96%
Best alpha                : +26.44%
Worst alpha               : -16.42%
Beat rate (alpha > 0)     : 85.0%
  -> Beat rate answers: 'did the agent outperform passive holding?'
     A positive mean alpha in a down market = the agent is working.

[SUCCESS] Saved -> models/saved/eval_results.json
```

### Alpha vs Passive Benchmark

```
  Agent Mean Return   ████████████████████  +2.03%
  Buy & Hold Return   ████████████████████████████████  -11.07%
                      Net Alpha Generated  =  +13.09%
```

> The agent achieves **85.0% beat rate** against passive holding — proving the policy learned defensive positioning in unfavorable macro regimes rather than simply riding the market upward.

---

## 🖥 Frontend Dashboard

### Tabs Overview

| Tab | File | Purpose |
|-----|------|---------|
| **Market** | `market_tab.dart` | Live candlestick chart with toggle-able indicator overlays, regime badge, GARCH volatility, drawdown, max exposure |
| **Live Sim** | `live_sim_tab.dart` | Virtual $10,000 sandbox portfolio — tracks real-time allocation decisions, equity curve, drawdown, latency |
| **Backtest** | `backtest_tab.dart` | Configure date range + strategy, run backtest, view return/Sharpe/Calmar/win-rate results |
| **News** | `news_tab.dart` | Aggregated live crypto news feed with per-article sentiment labels and aggregate sentiment index |
| **Analyst** | `analyst_tab.dart` | LLM-powered natural-language diagnostic — explains regime posture, risk levels, and recommended actions |

### Market Tab Telemetry Bar

| Field | Description |
|-------|-------------|
| **DRAWDOWN** | Active distance from historical equity peak (e.g. -68.60%) |
| **VOLATILITY** | GARCH-estimated regime: LOW / MEDIUM / HIGH |
| **MAX EXPOSURE** | Hard software ceiling on capital deployment (default 80%) |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Flutter SDK 3.x
- Node.js (optional, for tooling)

### 1. Clone the Repository
```bash
git clone https://github.com/shrawanprajapati/quantpilot-ai.git
cd quantpilot-ai
```

### 2. Set Up the Python Environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt` includes:**
```
fastapi
uvicorn[standard]
stable-baselines3[extra]
gymnasium
vectorbt
pandas
numpy
arch                  # GARCH volatility
hmmlearn              # HMM regime detection
google-generativeai   # Gemini LLM
python-binance        # Binance live data
```

### 3. Prepare Data
```bash
# Place raw OHLCV CSV in:
#   RLModel/data/raw/btcusdt_daily.csv
# Then preprocess:
python RLModel/data/preprocess.py
```

### 4. Train the RL Agent
```bash
cd RLModel/models
python train_final.py
# Monitor with: tensorboard --logdir ./logs/
```

### 5. Run the Validation Harness
```bash
python evaluate.py
# Results saved to: models/saved/eval_results.json
```

### 6. Start the FastAPI Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Run the Flutter Frontend
```bash
cd rl_trading_ui
flutter pub get
flutter run -d chrome          # Web
flutter run -d windows         # Desktop
```

The app will connect to `localhost:8000` by default.

---

## 📖 Usage Guide

### Running a Backtest
1. Open the **Backtest** tab
2. Select **Dataset** (TRAIN / TEST) and **Strategy** (rl_model / sma / ema / rsi / macd)
3. Set **From Date** and **To Date**
4. Click **RUN BACKTEST**
5. Results display: Total Return, Sharpe, Calmar, Max Drawdown, Win Rate, Trade Count

### Using the Live Simulation
1. Open the **Live Sim** tab
2. The model connects to the live WebSocket stream
3. Watch allocation decisions, equity curve, and drawdown update in real time
4. Portfolio starts at a virtual $10,000

### Querying the Analyst
1. Open the **Analyst** tab
2. Type a question (e.g. *"What is our plan for this HMM Market Regime?"*)
3. The LLM responds with a structured risk assessment

### Toggling Chart Indicators
1. Open the **Market** tab
2. Tap any indicator chip (EMA20, EMA50, SMA200, RSI, MACD, etc.)
3. The overlay renders / removes from the candlestick canvas in real time

---

## 🐛 Known Issues & Roadmap

### Current Known Issues

| ID | Component | Issue | Status |
|----|-----------|-------|--------|
| **TSK-001** | API Core | Observation scaling mismatch between standalone `evaluate.py` and `backtest.py` API router causes degraded in-app backtest performance (-9.09% vs +2.03% standalone) | 🔧 In Progress |
| **TSK-002** | UI Canvas | Chart canvas lag during high-frequency WebSocket tick updates | 🔧 Planned |
| **TSK-003** | Analyst Tab | LLM module failures during external API overload or rate-limiting | 🔧 Planned |

> **TSK-001 Root Cause:** The `/run_backtest` router passes unnormalized raw OHLCV columns directly to the policy network. During training, the environment normalizes observations via rolling z-scores before inference. The mismatch produces noisy action distributions, leading to extremely high trade turnover (725 trades), excessive fee churn, and a negative return profile. **Fix:** inject the same normalization pipeline into the backtest API router.

### Immediate Next Steps (Post Mid-Term)

- [ ] **Fix TSK-001** — normalize observations inside the backtest API router to match training-time preprocessing
- [ ] **Fix TSK-002** — migrate chart rendering to custom Flutter canvas painters for higher throughput
- [ ] **Fix TSK-003** — integrate a local fallback LLM for offline/overload scenarios
- [ ] **Multi-asset support** — extend the environment and portfolio tracker beyond BTC to a basket of crypto assets
- [ ] **Hyperparameter tuning** — systematic grid/random search over PPO hyperparameters and reward function weights
- [ ] **Live API deployment** — wire the Live Deploy toggle to a real Binance paper-trading account
- [ ] **User authentication** — add account management for startup-ready multi-user deployment
- [ ] **Docker packaging** — containerize the entire stack for one-command deployment
- [ ] **CI/CD pipeline** — automated test suite, model regression checks, and deployment hooks

### Vision: Startup-Grade Product

The longer-term roadmap positions QuantPilot AI as a **retail-accessible AI trading intelligence platform**:

1. **Model Quality** — continuous retraining, regime-adaptive hyperparameters, multi-timeframe observation stacking
2. **Product Polish** — onboarding flows, alert notifications, performance reports, mobile-first responsive UI
3. **Live Trading Safety Rails** — kill switches, daily loss limits, human-override buttons, audit logs
4. **Observability** — full telemetry dashboards, model drift monitoring, latency tracking
5. **Scalability** — multi-tenant backend, async job queues, GPU inference acceleration

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main`

Please ensure all new Python code passes `flake8` linting and all new Dart code passes `flutter analyze` before opening a PR.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io) — RL algorithm implementations
- [Gymnasium (Farama Foundation)](https://gymnasium.farama.org) — custom environment API
- [VectorBT](https://vectorbt.dev) — blazing-fast vectorized backtesting
- [FastAPI](https://fastapi.tiangolo.com) — async Python web framework
- [Flutter](https://flutter.dev) — cross-platform client framework
- Science & Technology Council, IIT Indore — IITISoC 2026

---

<div align="center">
  <strong>Built with ❤️ at IIT Indore · IITISoC 2026 · Finance & Analytics Domain</strong><br/>
  <em>Shrawan Kumar Prajapati · Aryan Bhupariya · Adwit Tiwari · Smita Singh · Atharva Kulkarni</em>
</div>
