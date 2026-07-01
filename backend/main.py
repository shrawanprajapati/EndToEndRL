from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import os
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from google import genai
from google.genai import types

client = genai.Client()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.services.trading_service import TradingService

app = FastAPI(title="QuantPilot API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
DATA_PATH = os.path.join(ROOT_DIR, "RLModel", "data", "processed", "featured_data.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "RLModel", "models", "saved", "backtest_report.json")
API_SECRET = os.environ.get("QUANTPILOT_SECRET", "changeme")
sys.path.insert(0, os.path.join(ROOT_DIR, "RLModel"))

from backtesting import get_strategies, run_vectorbt_backtest

# --- Models ---
class RiskLimits(BaseModel):
    max_drawdown: float
    max_position: float
    stop_loss: float
    take_profit: float

class ChatRequest(BaseModel):
    message: str
    session_id: str

class VectorBacktestRequest(BaseModel):
    mode: str = "test"
    strategy: str = "all"

# --- State ---
trading_service = TradingService()
STATE = {
    "emergency_stop": False,
    "risk_limits": {
        "max_drawdown": 15.0,
        "max_position": 0.8,
        "stop_loss": 8.0,
        "take_profit": 20.0
    }
}

df_cache = None

def load_data():
    global df_cache
    if os.path.exists(DATA_PATH):
        df_cache = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df_cache)} rows of feature data.")

@app.on_event("startup")
async def startup_event():
    load_data()
    trading_service.load()

# --- Auth Dependency ---
async def verify_secret(x_api_secret: str = Header(None)):
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Secret")

# --- Core Endpoints ---

@app.get("/")
async def root():
    return {
        "service": "QuantPilot backend",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "server_file": "backend/main.py",
        "model_ready": trading_service.is_ready,
        "feature_rows": 0 if df_cache is None else len(df_cache),
    }

@app.get("/api/v1/regime")
async def get_regime():
    market_dd = 0.0
    if df_cache is not None and not df_cache.empty:
        market_dd = float(df_cache.iloc[-1].get("drawdown_72h", 0.0))

    if "latest_tick" in STATE:
        tick = STATE["latest_tick"]
        return {
            "timestamp": str(tick.get("timestamp")),
            "hmm_regime": 1.0 if tick.get("hmm_regime") == "Bullish" else (-1.0 if tick.get("hmm_regime") == "Bearish" else 0.0),
            "hmm_regime_label": str(tick.get("hmm_regime", "Unknown")),
            "garch_volatility": float(tick.get("garch_volatility", 0.015)),
            "drawdown_72h": market_dd
        }
        
    if df_cache is None or df_cache.empty:
        raise HTTPException(status_code=404, detail="No data available")
    latest = df_cache.iloc[-1]
    return {
        "timestamp": str(latest.get("timestamp")),
        "hmm_regime": float(latest.get("hmm_regime", 0)),
        "hmm_regime_label": "Bullish" if latest.get("hmm_regime", 0) > 0.5 else ("Bearish" if latest.get("hmm_regime", 0) < -0.5 else "Sideways"),
        "garch_volatility": float(latest.get("garch_vol_raw", 0.015)),
        "drawdown_72h": float(latest.get("drawdown_72h", 0))
    }

@app.get("/api/v1/backtest")
async def get_backtest():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            return json.load(f)
    return {"message": "No backtest report available"}

@app.get("/api/v1/backtesting/strategies")
async def get_vectorbt_strategies():
    try:
        strategies = get_strategies()
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing Python dependency: {e.name}") from e
    
    # Add 'rl_model' to the list of available strategies sent to Flutter
    return {"modes": ["train", "test", "full"], "strategies": ["all", "rl_model"] + list(strategies.keys())}

@app.get("/api/v1/backtesting/comparison")
async def get_vectorbt_comparison():
    comparison_path = os.path.join(ROOT_DIR, "RLModel", "backtesting", "reports", "strategy_comparison.csv")
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Strategy comparison report not found.")
    df = pd.read_csv(comparison_path).replace([np.inf, -np.inf], np.nan)
    return {"comparison": df.where(pd.notnull(df), None).to_dict(orient="records")}

@app.post("/api/v1/run_backtest", dependencies=[Depends(verify_secret)])
async def run_vectorbt_strategy_backtest(request: VectorBacktestRequest):
    try:
        strategies = get_strategies()
        if request.mode not in {"train", "test", "full"}:
            raise HTTPException(status_code=400, detail="mode must be one of train, test, or full.")
            
        # ALLOW rl_model to pass the safety validation check
        if request.strategy != "all" and request.strategy not in strategies and request.strategy != "rl_model":
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_vectorbt_backtest(mode=request.mode, strategy=request.strategy),
        )
    except HTTPException:
        raise
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing Python dependency: {e.name}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    rows = pd.DataFrame(result["comparison"]).replace([np.inf, -np.inf], np.nan)
    result["comparison"] = rows.where(pd.notnull(rows), None).to_dict(orient="records")
    return result

@app.get("/api/v1/checkpoints")
async def get_checkpoints():
    ckpt_path = os.path.join(ROOT_DIR, "RLModel", "models", "saved", "*.zip")
    files = glob.glob(ckpt_path)
    checkpoints = []
    for f in files:
        name = os.path.basename(f).replace(".zip", "")
        checkpoints.append({"name": name, "timestamp": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()})
    return {"checkpoints": checkpoints}

@app.get("/api/v1/features/latest")
async def get_latest_features(n: int = 10):
    if df_cache is None or df_cache.empty:
        return []
    latest = df_cache.tail(n).to_dict(orient="records")
    return latest

@app.get("/api/v1/feature_importance")
async def get_feature_importance():
    # Placeholder for SHAP calculations
    return {
        "importance": [
            {"feature": "rsi_14", "importance": 0.25},
            {"feature": "macd_line", "importance": 0.18},
            {"feature": "garch_vol", "importance": 0.15}
        ]
    }

@app.get("/api/v1/news")
async def get_news():
    # Return mock realistic crypto news data
    import random
    sources = ["CoinDesk", "CoinTelegraph", "Bloomberg", "Reuters", "Decrypt"]
    titles = [
        "Bitcoin Spot ETFs See Record Inflows as Institutions Pile In",
        "Ethereum Network Upgrade Successfully Deployed on Testnet",
        "Regulatory Clarity Sparks Bullish Sentiment in Crypto Markets",
        "Major Retailer Announces Acceptance of Crypto Payments",
        "Market Analysts Predict Upcoming Volatility Amid Macroeconomic Shifts",
        "DeFi Protocols Reach New All-Time Highs in Total Value Locked",
        "Central Banks Explore Integration of Digital Currencies",
        "Mining Difficulty Adjusts Following Network Hashrate Surge"
    ]
    news_items = []
    for i in range(10):
        source = random.choice(sources)
        title = random.choice(titles)
        sentiment_val = random.random()
        if sentiment_val > 0.6:
            sentiment = "positive"
        elif sentiment_val < 0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        news_items.append({
            "id": str(i + 1),
            "source": source,
            "title": title,
            "sentiment": sentiment,
            "url": "https://example.com/news",
            "published_at": (datetime.now() - pd.Timedelta(minutes=random.randint(5, 300))).isoformat()
        })
        
    return {"news": sorted(news_items, key=lambda x: x["published_at"], reverse=True)}

# --- Agent Endpoints ---

@app.get("/api/v1/agent/report/latest")
async def get_latest_report():
    return {"verdict": "BUY", "score": 75, "reasoning": "Strong momentum detected in RSI and MACD."}

@app.get("/api/v1/agent/report/history")
async def get_report_history():
    return {"history": []}

@app.post("/api/v1/agent/report", dependencies=[Depends(verify_secret)])
async def trigger_report():
    # 1. Gather historical/live data context matrices
    current_price = 0.0
    regime_label = "Unknown"
    volatility = 0.015
    drawdown = 0.0
    
    if "latest_tick" in STATE:
        tick = STATE["latest_tick"]
        current_price = float(tick.get("price", 0.0))
        regime_label = str(tick.get("hmm_regime", "Unknown"))
        volatility = float(tick.get("garch_volatility", 0.015))
        drawdown = float(tick.get("drawdown", 0.0))
    elif df_cache is not None and not df_cache.empty:
        latest_row = df_cache.iloc[-1]
        current_price = float(latest_row.get("close", 0.0))
        hmm_val = latest_row.get("hmm_regime", 0)
        regime_label = "Bullish" if hmm_val > 0.5 else ("Bearish" if hmm_val < -0.5 else "Sideways")
        volatility = float(latest_row.get("garch_vol_raw", 0.015))
        drawdown = float(latest_row.get("drawdown_72h", 0.0))

    # 2. Frame systemic parameters
    system_instruction = """
    You are the Lead Risk Officer and Quant Auditor for QuantPilot AI.
    Your task is to generate a comprehensive, executive trading intelligence report. 
    Break your report down into 3 clean Markdown sections:
    1. MARKET REGIME & ENVIRONMENT AUDIT
    2. RISK METRICS & DRAWDOWN ASSESSMENT
    3. STRATEGIC ALLOCATION & ROADMAP
    Be highly professional, completely technical, and use markdown lists or bullet points.
    """
    
    user_prompt = f"""
    Compile a comprehensive audit report given these live model metrics:
    - Target Asset: BTC/USDT
    - Terminal Price: ${current_price:,.2f}
    - Hidden Markov Model State: {regime_label}
    - GARCH Volatility Factor: {volatility:.5f}
    - 72h Realized Peak-to-Trough Drawdown: {drawdown * 100:.2f}%
    - Safety Position Cap: {STATE['risk_limits']['max_position']*100}%
    """

    try:
        # 3. Request structured synthesis report from Gemini
        response = client.models.generate_content(
            model='gemini-2.0-flash', # Corrected model version
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        
        # Return a structured dictionary for the Flutter UI
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "verdict": "BUY" if regime_label == "Bullish" else "EXIT",
            "report_content": response.text
        }
    except Exception as e:
        print(f"DEBUG ERROR: {e}") # Print to your terminal to see exactly what failed
        raise HTTPException(status_code=500, detail=f"Report Generation Failed: {str(e)}")

@app.post("/api/v1/agent/chat")
async def agent_chat(request: ChatRequest):
    # 1. Fetch real-time feature states from our system cache matrix to seed the context window
    current_price = 0.0
    regime_label = "Unknown"
    volatility = 0.015
    drawdown = 0.0
    
    if "latest_tick" in STATE:
        tick = STATE["latest_tick"]
        current_price = float(tick.get("price", 0.0))
        regime_label = str(tick.get("hmm_regime", "Unknown"))
        volatility = float(tick.get("garch_volatility", 0.015))
        drawdown = float(tick.get("drawdown", 0.0))
    elif df_cache is not None and not df_cache.empty:
        latest_row = df_cache.iloc[-1]
        current_price = float(latest_row.get("close", 0.0))
        hmm_val = latest_row.get("hmm_regime", 0)
        regime_label = "Bullish" if hmm_val > 0.5 else ("Bearish" if hmm_val < -0.5 else "Sideways")
        volatility = float(latest_row.get("garch_vol_raw", 0.015))
        drawdown = float(latest_row.get("drawdown_72h", 0.0))
    else:
        # Fallback tracking parameters if cache stream is bootstrapping
        regime_label = "Sideways"

    max_pos = STATE['risk_limits']['max_position']

    # 2. Build out engineering system context prompts combining live telemetry constraints
    system_instruction = f"""
    You are the QuantPilot AI Pro Analyst, an advanced algorithmic trading system copilot.
    You evaluate cryptocurrency parameters by unifying your financial knowledge base with live mathematical telemetry from a Reinforcement Learning (RL) execution model.
    
    Current Live Telemetry State:
    - Current Asset Price: ${current_price:,.2f}
    - Hidden Markov Model (HMM) Market Regime: {regime_label}
    - GARCH Volatility Factor: {volatility:.5f}
    - 72-Hour Continuous Portfolio Drawdown: {drawdown * 100:.2f}%
    - Safety Boundaries: {max_pos * 100}% maximum exposure cap (Risk parameters active)
    
    Instructions:
    Answer the user's prompt by directly synthesizing the telemetry values with your market intelligence. 
    Explain why the model is configured this way, maintain an executive tone, and avoid hand-waving or meta-commentary.
    """

    try:
        # 3. Request analysis using the official, recommended SDK generation model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.25, # Lower temp ensures predictable structural analysis
            )
        )
        
        # 4. Programmatically map semantic badges to match state layouts
        data_verdict = "NEUTRAL"
        data_score = 50
        
        if regime_label == "Bullish" and volatility < 0.02:
            data_verdict = "BUY"
            data_score = 80
        elif regime_label == "Bearish" or drawdown > 0.10:
            data_verdict = "EXIT"
            data_score = 15

        # 5. Package into the downstream data matrix needed by the Flutter Client
        return {
            "answer": response.text,
            "suggested_questions": [
                "How does the GARCH volatility impact our exposure?",
                "What is our plan for this HMM Market Regime?"
            ],
            "data_confidence": "LIVE",
            "current_verdict": data_verdict,
            "current_score": data_score
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini processing exception: {str(e)}"
        ) from e

@app.post("/api/v1/agent/chat/reset")
async def reset_chat(request: ChatRequest):
    return {"status": "success"}

# --- WebSockets ---

@app.websocket("/api/v1/ws/candles")
async def ws_candles(websocket: WebSocket):
    await websocket.accept()
    try:
        if df_cache is not None:
            # Stream actual historical candles first
            for _, row in df_cache.tail(100).iterrows():
                await websocket.send_json({
                    "timestamp": str(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                })
        while True:
            # Then follow with "live" ticks
            await asyncio.sleep(60) # Simulate 1m candles
    except WebSocketDisconnect:
        pass

@app.websocket("/api/v1/stream")
async def stream_data(websocket: WebSocket):
    await websocket.accept()
    
    async def send_tick(data):
        try:
            if STATE["emergency_stop"]:
                data["action"] = 0.0
                data["status"] = "EMERGENCY_STOP"
            data["agent_verdict"] = "BUY" if data["action"] > 0.5 else "NEUTRAL"
            data["agent_score"] = int(data["action"] * 100)
            data["type"] = "tick"
            STATE["latest_tick"] = data
            await websocket.send_json(data)
        except Exception:
            pass

    try:
        await trading_service.run_simulation_loop(send_tick)
    except WebSocketDisconnect:
        pass

@app.websocket("/api/v1/ws/agent_stream")
async def agent_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(10)
            await websocket.send_json({"type": "agent_insight", "text": "Market showing signs of trend reversal."})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
