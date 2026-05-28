"""
Short Trade Screener Bot
Scans all Binance USDT-M perpetuals every 4 hours and sends Telegram alerts
when weighted signal score >= 12/17.

Signals:
  Bollinger Band upper (4H)   — weight 4
  OBV Divergence 1H           — weight 4
  OBV Divergence 4H           — weight 4
  RSI 4H overbought/diverge   — weight 3
  Funding Rate elevated       — weight 2
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MIN_SCORE        = 12       # alert threshold out of 17
SCAN_INTERVAL    = 4 * 3600 # 4 hours in seconds
TOP_N_VOLUME     = None     # None = all USDT-M perpetuals

BINANCE_BASE     = "https://fapi.binance.com"

# Signal weights
W_BB      = 4
W_OBV_1H  = 4
W_OBV_4H  = 4
W_RSI_4H  = 3
W_FUNDING = 2
MAX_SCORE = W_BB + W_OBV_1H + W_OBV_4H + W_RSI_4H + W_FUNDING  # 17

# Indicator params
BB_PERIOD    = 20
BB_STD       = 2.0
RSI_PERIOD   = 14
RSI_OB       = 70          # overbought threshold
OBV_PIVOTS   = 2           # number of swing points to compare for divergence
FUNDING_THRESH = 0.0001    # 0.01% per 8h (positive = longs paying)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ─── BINANCE HELPERS ─────────────────────────────────────────────────────────

async def get_all_usdt_perpetuals(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo")
    resp.raise_for_status()
    data = resp.json()
    symbols = [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    ]
    return sorted(symbols)


async def get_klines(client: httpx.AsyncClient, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    resp = await client.get(
        f"{BINANCE_BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit}
    )
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


async def get_funding_rate(client: httpx.AsyncClient, symbol: str) -> float:
    resp = await client.get(
        f"{BINANCE_BASE}/fapi/v1/premiumIndex",
        params={"symbol": symbol}
    )
    resp.raise_for_status()
    return float(resp.json()["lastFundingRate"])


# ─── INDICATORS ──────────────────────────────────────────────────────────────

def calc_bollinger(df: pd.DataFrame) -> dict:
    """Returns True if current close >= upper Bollinger Band."""
    close = df["close"]
    sma   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    current_close = close.iloc[-1]
    current_upper = upper.iloc[-1]
    triggered = current_close >= current_upper
    return {
        "signal": triggered,
        "close":  round(current_close, 6),
        "upper":  round(current_upper, 6),
    }


def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    obv = (direction * df["volume"]).cumsum()
    return obv


def find_swing_highs(series: pd.Series, n: int = 2) -> pd.Series:
    """Returns a series with swing high values (NaN elsewhere)."""
    highs = pd.Series(np.nan, index=series.index)
    for i in range(n, len(series) - n):
        window = series.iloc[i - n: i + n + 1]
        if series.iloc[i] == window.max():
            highs.iloc[i] = series.iloc[i]
    return highs


def bearish_divergence(price: pd.Series, indicator: pd.Series, n_pivots: int = OBV_PIVOTS) -> bool:
    """
    Returns True if price is making higher highs while indicator makes lower highs
    over the last n_pivots swing highs.
    """
    price_highs = find_swing_highs(price).dropna()
    ind_highs   = find_swing_highs(indicator).dropna()

    # Align on common indices
    common = price_highs.index.intersection(ind_highs.index)
    if len(common) < 2:
        return False

    price_highs = price_highs[common].iloc[-n_pivots:]
    ind_highs   = ind_highs[common].iloc[-n_pivots:]

    if len(price_highs) < 2:
        return False

    price_trend = price_highs.iloc[-1] > price_highs.iloc[0]   # price HH
    ind_trend   = ind_highs.iloc[-1]   < ind_highs.iloc[0]     # indicator LH
    return bool(price_trend and ind_trend)


def check_rsi_signal(df: pd.DataFrame) -> dict:
    """RSI overbought OR bearish RSI divergence on the given timeframe."""
    rsi = calc_rsi(df["close"])
    overbought  = rsi.iloc[-1] > RSI_OB
    divergence  = bearish_divergence(df["close"], rsi)
    return {
        "signal":     overbought or divergence,
        "overbought": overbought,
        "divergence": divergence,
        "rsi_value":  round(rsi.iloc[-1], 1),
    }


def check_obv_divergence(df: pd.DataFrame) -> dict:
    obv  = calc_obv(df)
    div  = bearish_divergence(df["close"], obv)
    return {"signal": div}


# ─── SCAN ONE SYMBOL ─────────────────────────────────────────────────────────

async def scan_symbol(client: httpx.AsyncClient, symbol: str) -> dict | None:
    try:
        # Fetch data concurrently
        df_1h, df_4h, funding = await asyncio.gather(
            get_klines(client, symbol, "1h", 72),
            get_klines(client, symbol, "4h", 18),
            get_funding_rate(client, symbol),
        )

        bb     = calc_bollinger(df_4h)
        obv_1h = check_obv_divergence(df_1h)
        obv_4h = check_obv_divergence(df_4h)
        rsi    = check_rsi_signal(df_4h)
        fund   = {"signal": funding > FUNDING_THRESH, "rate": funding}

        score = (
            (W_BB      if bb["signal"]     else 0) +
            (W_OBV_1H  if obv_1h["signal"] else 0) +
            (W_OBV_4H  if obv_4h["signal"] else 0) +
            (W_RSI_4H  if rsi["signal"]    else 0) +
            (W_FUNDING if fund["signal"]   else 0)
        )

        return {
            "symbol":  symbol,
            "score":   score,
            "price":   bb["close"],
            "bb":      bb,
            "obv_1h":  obv_1h,
            "obv_4h":  obv_4h,
            "rsi":     rsi,
            "funding": fund,
        }

    except Exception as e:
        log.warning(f"  {symbol}: skipped — {e}")
        return None


# ─── FORMAT TELEGRAM MESSAGE ─────────────────────────────────────────────────

def format_alert(results: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"🔴 *Short Trade Screener* — {now}",
        f"Found *{len(results)}* candidate(s) with score ≥ {MIN_SCORE}/{MAX_SCORE}\n",
    ]

    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        s = r["score"]
        pct = round(s / MAX_SCORE * 100)
        verdict = (
            "Strong setup"   if pct >= 88 else
            "Good setup"     if pct >= 70 else
            "Moderate setup"
        )

        bb_line    = "✅" if r["bb"]["signal"]     else "❌"
        obv1h_line = "✅" if r["obv_1h"]["signal"] else "❌"
        obv4h_line = "✅" if r["obv_4h"]["signal"] else "❌"
        rsi_line   = "✅" if r["rsi"]["signal"]    else "❌"
        fund_line  = "✅" if r["funding"]["signal"] else "❌"

        rsi_detail = ""
        if r["rsi"]["overbought"]:
            rsi_detail = f" (RSI {r['rsi']['rsi_value']} > {RSI_OB})"
        elif r["rsi"]["divergence"]:
            rsi_detail = f" (divergence, RSI {r['rsi']['rsi_value']})"

        fund_detail = f" ({r['funding']['rate']*100:.4f}%)"

        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"*{r['symbol']}*  |  Score: *{s}/{MAX_SCORE}* ({pct}%)  |  _{verdict}_",
            f"Price: `{r['price']}`",
            f"",
            f"{bb_line} Bollinger Band upper touch (4H)  [w:4]",
            f"{obv1h_line} OBV divergence (1H)  [w:4]",
            f"{obv4h_line} OBV divergence (4H)  [w:4]",
            f"{rsi_line} RSI 4H{rsi_detail}  [w:3]",
            f"{fund_line} Funding rate elevated{fund_detail}  [w:2]",
            f"",
            f"👉 [Binance Chart](https://www.binance.com/en/futures/{r['symbol']})",
        ]

    lines.append("\n_⚠️ Screener only — always apply your own analysis before entering._")
    return "\n".join(lines)


def format_no_results(total_symbols: int, duration_secs: float, next_scan: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"🔍 *Short Trade Screener* — {now}\n\n"
        f"No candidates found with score ≥ {MIN_SCORE}/{MAX_SCORE}\n\n"
        f"📊 Scanned *{total_symbols}* symbols in {duration_secs:.0f}s\n"
        f"⏰ Next scan at *{next_scan} UTC*"
    )


def format_alert_with_heartbeat(results: list[dict], total_symbols: int, duration_secs: float, next_scan: str) -> str:
    msg = format_alert(results)
    msg += (
        f"\n\n📊 Scanned *{total_symbols}* symbols in {duration_secs:.0f}s"
        f"\n⏰ Next scan at *{next_scan} UTC*"
    )
    return msg


def format_startup() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    next_scan = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")
    return (
        f"✅ *Short Trade Screener is live* — {now}\n\n"
        f"Scanning all USDT\\-M perpetuals every 4 hours\\.\n\n"
        f"*Settings*\n"
        f"• Alert threshold: {MIN_SCORE}/{MAX_SCORE}\n"
        f"• Signals: BB \\(4H\\), OBV \\(1H\\), OBV \\(4H\\), RSI \\(4H\\), Funding Rate\n\n"
        f"⏰ First scan starting now\\. Next at *{next_scan} UTC*"
    )


def format_error(error: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"⚠️ *Screener error* — {now}\n\n"
        f"`{error[:200]}`\n\n"
        f"_Will retry at next scheduled scan._"
    )


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

async def run_scan(bot: Bot):
    log.info("Starting scan…")
    scan_start = time.monotonic()

    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(timeout=20, limits=limits) as client:
        symbols = await get_all_usdt_perpetuals(client)
        log.info(f"Scanning {len(symbols)} USDT-M perpetuals…")

        # Scan in batches to avoid rate limits
        BATCH = 20
        results = []
        for i in range(0, len(symbols), BATCH):
            batch = symbols[i:i + BATCH]
            tasks = [scan_symbol(client, sym) for sym in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r and r["score"] >= MIN_SCORE])
            log.info(f"  Scanned {min(i + BATCH, len(symbols))}/{len(symbols)}…")
            await asyncio.sleep(0.5)  # gentle rate limit pause

    duration = time.monotonic() - scan_start
    next_scan = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")

    log.info(f"Scan complete in {duration:.0f}s. {len(results)} candidate(s) found.")

    if results:
        msg = format_alert_with_heartbeat(results, len(symbols), duration, next_scan)
    else:
        msg = format_no_results(len(symbols), duration, next_scan)

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=msg,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )
    log.info("Telegram message sent.")


async def main():
    log.info(f"Short Trade Screener started. Scanning every {SCAN_INTERVAL // 3600}h, threshold {MIN_SCORE}/{MAX_SCORE}.")
    bot = Bot(token=TELEGRAM_TOKEN)

    # Send startup confirmation
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=format_startup(),
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
        )
    except Exception as e:
        log.error(f"Could not send startup message: {e}")

    while True:
        try:
            await run_scan(bot)
        except Exception as e:
            log.error(f"Scan failed: {e}", exc_info=True)
            try:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=format_error(str(e)),
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True,
                )
            except Exception:
                pass
        log.info(f"Sleeping {SCAN_INTERVAL // 3600}h until next scan…")
        await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
