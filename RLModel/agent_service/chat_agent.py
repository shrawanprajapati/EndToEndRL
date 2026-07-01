import os
import time
import json
import re
import asyncio
import httpx
from google import genai
from google.genai import types

AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "chat_audit.log"
)

GEMINI_MODEL = "gemini-2.0-flash"


class ChatAgent:
    def __init__(
        self,
        api_base_url: str = "http://127.0.0.1:8000",
        gemini_api_key: str = None,
        max_history_turns: int = 10,
    ):
        self.api_base_url = api_base_url
        self.gemini_api_key = gemini_api_key or os.environ.get("AIzaSyC8pfeeVFqmkswHmDlpyd7_jxTiqtBMxCk", "")
        self.max_history_turns = max_history_turns
        # Gemini history format: list of types.Content objects
        self.conversation_history: list = []
        self.last_context_bundle: dict = {}
        self.last_context_time: float = 0.0
        self.context_ttl: float = 30.0

    # ── Context Fetching ─────────────────────────────────────────────────────

    async def get_live_context(self) -> dict:
        now = time.time()
        if now - self.last_context_time < self.context_ttl and self.last_context_bundle:
            bundle = dict(self.last_context_bundle)
            bundle["freshness"] = "CACHED"
            return bundle

        async def _fetch(client: httpx.AsyncClient, endpoint: str):
            try:
                url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
                headers = {"x-api-secret": os.environ.get("QUANTPILOT_SECRET", "changeme")}
                resp = await client.get(url, headers=headers, timeout=4.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                print(f"[ChatAgent] Error fetching {endpoint}: {e}")
            return None

        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    _fetch(client, "api/v1/regime"),
                    _fetch(client, "api/v1/backtest"),
                    _fetch(client, "api/v1/feature_importance"),
                    _fetch(client, "health"),
                    _fetch(client, "api/v1/agent/report/latest"),
                    _fetch(client, "api/v1/features/latest?n=1"),
                    return_exceptions=True,
                )
        except Exception as e:
            print(f"[ChatAgent] Gather error: {e}")
            results = [None] * 6

        def _safe(r):
            return r if isinstance(r, dict) else {}

        regime_data     = _safe(results[0])
        backtest_data   = _safe(results[1])
        importance_data = _safe(results[2])
        health_data     = _safe(results[3])
        report_data     = _safe(results[4])

        features_raw = results[5]
        if isinstance(features_raw, list) and features_raw:
            features_data = features_raw[0]
        elif isinstance(features_raw, dict):
            features_data = features_raw
        else:
            features_data = {}

        is_successful = any(isinstance(r, (dict, list)) for r in results)

        bundle = {
            "regime":             regime_data.get("hmm_regime_label", "Unknown"),
            "hmm_regime":         regime_data.get("hmm_regime", 0.0),
            "price":              features_data.get("close", "N/A"),
            "garch_vol":          features_data.get("garch_vol", "N/A"),
            "backtest":           backtest_data,
            "feature_importance": importance_data,
            "health":             health_data,
            "report":             report_data,
            "latest_feature":     features_data,
        }

        if is_successful:
            bundle["freshness"] = "LIVE"
            self.last_context_bundle = bundle
            self.last_context_time = now
        else:
            if self.last_context_bundle:
                bundle = dict(self.last_context_bundle)
                bundle["freshness"] = "STALE"
            else:
                bundle["freshness"] = "STALE"

        return bundle

    # ── System Prompt ─────────────────────────────────────────────────────────

    def build_system_prompt(self, context: dict) -> str:
        return f"""You are the "QuantPilot Analyst", an expert quantitative trading AI analyst.
Your job is to explain the reinforcement learning model's trading decisions, market regimes,
features, and risk limits to the user in clear, technical terms.

CURRENT REAL-TIME CONTEXT:
- Close Price: {context.get("price", "N/A")}
- HMM Regime: {context.get("regime", "N/A")} (raw={context.get("hmm_regime", "N/A")})
- GARCH Volatility: {context.get("garch_vol", "N/A")}
- Health / Emergency Stop: {json.dumps(context.get("health", {}))}
- Backtest Summary: {json.dumps(context.get("backtest", {}))}
- SHAP Feature Importance: {json.dumps(context.get("feature_importance", {}))}
- Latest Feature Row: {json.dumps(context.get("latest_feature", {}))}
- Report Narrative: {json.dumps(context.get("report", {}))}

RULES:
1. Ground every answer strictly in the data above. Never fabricate numbers.
2. Stay focused on this RL model — avoid generic crypto advice.
3. SHAP feature glossary:
   - rsi_14: RSI momentum (oversold < 30, overbought > 70)
   - macd_line: MACD trend signal
   - frac_diff_0_5: Fractionally differenced price (memory-preserving stationarity)
   - cvd_24h: Cumulative Volume Delta (buy/sell pressure)
   - garch_vol: GARCH-estimated 1h volatility
4. At the very end of your response append EXACTLY this line (no extra newlines before it):
   You might also ask: [Question 1] / [Question 2] / [Question 3]
"""

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat(self, user_message: str) -> dict:
        start = time.time()

        # Trim history to max turns (each turn = 2 Content items: user + model)
        max_items = self.max_history_turns * 2
        if len(self.conversation_history) > max_items:
            self.conversation_history = self.conversation_history[-max_items:]

        context = await self.get_live_context()
        system_prompt = self.build_system_prompt(context)

        assistant_response = ""
        suggested: list[str] = []

        if not self.gemini_api_key:
            assistant_response = (
                "Running in local mode — GEMINI_API_KEY not set. "
                f"Current regime: {context.get('regime')}, "
                f"volatility: {context.get('garch_vol')}.\n\n"
                "You might also ask: What is the current regime? / "
                "What does GARCH volatility mean? / How is the model performing?"
            )
        else:
            try:
                client = genai.Client(api_key=self.gemini_api_key)

                # Build contents: prior history + new user message
                contents = list(self.conversation_history) + [
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_message)]
                    )
                ]

                def _call():
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.4,
                            max_output_tokens=1024,
                        ),
                    )
                    return response.text

                assistant_response = await asyncio.wait_for(
                    asyncio.to_thread(_call),
                    timeout=30.0,
                )

            except Exception as e:
                print(f"[ChatAgent] Gemini error: {e}")
                assistant_response = (
                    f"I encountered an issue reaching Gemini ({type(e).__name__}). "
                    f"Current regime is {context.get('regime')}.\n\n"
                    "You might also ask: What is the current regime? / "
                    "How is the model performing? / What does SHAP show?"
                )

        # Store turn in Gemini Content format
        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )
        self.conversation_history.append(
            types.Content(role="model", parts=[types.Part(text=assistant_response)])
        )

        # Parse suggested questions from trailing line
        match = re.search(r"You might also ask:\s*(.+)", assistant_response, re.IGNORECASE)
        if match:
            raw = match.group(1)
            suggested = [q.strip() for q in re.split(r"\s*/\s*", raw) if q.strip()][:3]

        latency = round(time.time() - start, 3)

        # Audit log
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_message": user_message,
                    "response_length": len(assistant_response),
                    "latency_sec": latency,
                    "freshness": context.get("freshness"),
                    "model": GEMINI_MODEL,
                }) + "\n")
        except Exception as e:
            print(f"[ChatAgent] Audit log error: {e}")

        return {
            "answer": assistant_response,
            "suggested_questions": suggested,
            "data_confidence": context.get("freshness", "STALE"),
            "current_verdict": context.get("regime", "Unknown"),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def reset_conversation(self):
        self.conversation_history = []

    def get_conversation_summary(self) -> dict:
        return {
            "turns": len(self.conversation_history) // 2,
            "last_turn": (
                self.conversation_history[-1].parts[0].text
                if self.conversation_history else None
            ),
            "model": GEMINI_MODEL,
        }
