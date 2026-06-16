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
from aiohttp import web
from telegram import Bot
from telegram.constants import ParseMode

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MIN_SCORE        = 10       # alert threshold out of 17
SCAN_INTERVAL    = 4 * 3600 # 4 hours in seconds
TOP_N_VOLUME     = None     # None = all USDT-M perpetuals

BINANCE_BASE     = "https://fapi.binance.com"

# Signal weights
W_BB      = 4
W_OBV_1H  = 4
W_OBV_4H  = 4
W_RSI_4H  = 3
W_CVD_4H  = 3
W_FUNDING = 2
MAX_SCORE = W_BB + W_OBV_1H + W_OBV_4H + W_RSI_4H + W_CVD_4H + W_FUNDING  # 20

# Indicator params
BB_PERIOD      = 20
BB_STD         = 2.0
BB_LOOKBACK    = 5           # candles to look back for a recent upper band tag
RSI_PERIOD     = 14
RSI_OB         = 65          # overbought threshold
OBV_PIVOTS     = 2           # number of swing points to compare for divergence
FUNDING_THRESH = 0.0001      # 0.01% per 8h (positive = longs paying)

# Pullback filter (hard gate — must pass before signals are checked)
RSI_1D_PERIOD    = 6     # RSI period for daily timeframe
RSI_1D_THRESHOLD = 60   # minimum RSI(6) on 1D to pass the filter
PULLBACK_MIN_RETRACE = 0.05  # price must be at least 5% below the swing high
PULLBACK_MAX_RETRACE = 0.20  # price must be no more than 20% below the swing high

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
    """
    Returns True if any of the last BB_LOOKBACK candles closed at or above
    the upper Bollinger Band — confirming recent overextension before the retrace.
    """
    close = df["close"]
    sma   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std

    # Check last BB_LOOKBACK candles for an upper band tag
    recent_close = close.iloc[-BB_LOOKBACK:]
    recent_upper = upper.iloc[-BB_LOOKBACK:]
    tagged = (recent_close >= recent_upper).any()

    current_close = close.iloc[-1]
    current_upper = upper.iloc[-1]

    return {
        "signal":  tagged,
        "close":   round(current_close, 6),
        "upper":   round(current_upper, 6),
        "gap_pct": round((current_upper - current_close) / current_upper * 100, 2),
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


def calc_cvd(df: pd.DataFrame) -> pd.Series:
    """
    True CVD using taker buy/sell volume from klines.
    taker_sell = total_volume - taker_buy_base
    delta per candle = taker_buy_base - taker_sell = 2 * taker_buy_base - volume
    CVD = cumulative sum of deltas
    """
    delta = 2 * df["taker_buy_base"] - df["volume"]
    return delta.cumsum()


def check_cvd_divergence(df: pd.DataFrame) -> dict:
    cvd = calc_cvd(df)
    div = bearish_divergence(df["close"], cvd)
    return {"signal": div}


def find_swing_highs(series: pd.Series, n: int = 2) -> pd.Series:
    """
    Returns swing high values. Uses n=2 lookahead normally, but also
    includes the most recent candle as a candidate if it's a local high
    with just 1 candle of lookahead — prevents missing fresh highs at
    the edge of the data.
    """
    highs = pd.Series(np.nan, index=series.index)
    length = len(series)
    for i in range(n, length):
        lookback = series.iloc[i - n: i]
        # Full n-candle lookahead if available, else at least 1
        lookahead_end = min(i + n + 1, length)
        lookahead = series.iloc[i + 1: lookahead_end]
        if len(lookahead) == 0:
            continue
        if series.iloc[i] >= lookback.max() and series.iloc[i] >= lookahead.max():
            highs.iloc[i] = series.iloc[i]
    return highs


def slope_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 20) -> bool:
    """
    Fallback: compares linear slope of price vs indicator over last N candles.
    Returns True if price slope is positive while indicator slope is negative.
    """
    if len(price) < lookback:
        return False
    p = price.iloc[-lookback:].values
    ind = indicator.iloc[-lookback:].values
    x = np.arange(lookback)
    price_slope = np.polyfit(x, p, 1)[0]
    ind_slope   = np.polyfit(x, ind, 1)[0]
    return bool(price_slope > 0 and ind_slope < 0)


def bearish_divergence(price: pd.Series, indicator: pd.Series, n_pivots: int = OBV_PIVOTS) -> bool:
    """
    Returns True if bearish divergence detected via pivot-point method OR
    slope-based fallback. Pivot method: price HH while indicator LH.
    Slope fallback: price trending up while indicator trending down.
    """
    # Primary: pivot point method
    price_highs = find_swing_highs(price).dropna()
    ind_highs   = find_swing_highs(indicator).dropna()

    pivot_divergence = False
    if len(price_highs) >= 2 and len(ind_highs) >= 2:
        # Use last n_pivots price swing highs, find nearest indicator highs
        ph = price_highs.iloc[-n_pivots:]
        ih = ind_highs.iloc[-n_pivots:]
        if len(ph) >= 2 and len(ih) >= 2:
            price_trend = ph.iloc[-1] > ph.iloc[0]
            ind_trend   = ih.iloc[-1] < ih.iloc[0]
            pivot_divergence = bool(price_trend and ind_trend)

    # Fallback: slope-based over last 20 candles
    slope_div = slope_divergence(price, indicator, lookback=20)

    return pivot_divergence or slope_div


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


def check_pullback(df: pd.DataFrame) -> dict:
    """
    Hard gate: requires a meaningful swing and an active retrace within the entry window.
    - Swing:   (highest high - lowest low) / lowest low >= PULLBACK_MIN_SWING (15%)
    - Retrace: current price is between 5% and 20% below the swing high
      - < 5%  → too early, still near the top
      - > 20% → too late, move already played out
    """
    highs = df["high"]
    lows  = df["low"]
    close = df["close"].iloc[-1]

    swing_high = highs.max()
    swing_low  = lows.min()

    swing_pct   = (swing_high - swing_low) / swing_low
    retrace_pct = (swing_high - close) / swing_high

    passes = (
        swing_pct   >= PULLBACK_MIN_SWING and
        retrace_pct >= PULLBACK_MIN_RETRACE and
        retrace_pct <= PULLBACK_MAX_RETRACE
    )

    return {
        "passes":      passes,
        "swing_pct":   round(swing_pct * 100, 1),
        "retrace_pct": round(retrace_pct * 100, 1),
        "swing_high":  round(swing_high, 6),
        "swing_low":   round(swing_low, 6),
    }


# ─── SCAN ONE SYMBOL ─────────────────────────────────────────────────────────

async def scan_symbol(client: httpx.AsyncClient, symbol: str) -> dict | None:
    try:
        # Fetch data concurrently — includes 1D candles for RSI(6) gate
        df_1h, df_4h, df_1d, funding = await asyncio.gather(
            get_klines(client, symbol, "1h", 72),
            get_klines(client, symbol, "4h", 20),
            get_klines(client, symbol, "1d", 20),
            get_funding_rate(client, symbol),
        )

        # Hard gate 1 — Daily RSI(6) must be >= 60
        rsi_1d_series = calc_rsi(df_1d["close"], period=RSI_1D_PERIOD)
        rsi_1d_value  = rsi_1d_series.iloc[-1]
        if rsi_1d_value < RSI_1D_THRESHOLD:
            return None

        # Hard gate 2 — pullback filter using last 20 × 4H candles (~80 hours)
        pullback = check_pullback(df_4h.iloc[-20:])
        if not pullback["passes"]:
            return None

        bb     = calc_bollinger(df_4h)
        obv_1h = check_obv_divergence(df_1h)
        obv_4h = check_obv_divergence(df_4h)
        cvd_4h = check_cvd_divergence(df_4h)
        rsi    = check_rsi_signal(df_4h)
        fund = {
            "signal":   funding > FUNDING_THRESH,
            "positive": funding > 0,
            "negative": funding < 0,
            "rate":     funding,
        }

        # Conditional 1H RSI check — only when funding is NEGATIVE and RSI 4H is favourable
        rsi_1h = None
        if fund["negative"] and rsi["signal"]:
            rsi_1h_series = calc_rsi(df_1h["close"])
            rsi_1h = {
                "value":    round(rsi_1h_series.iloc[-1], 1),
                "above_85": rsi_1h_series.iloc[-1] > 85,
            }

        score = (
            (W_BB      if bb["signal"]     else 0) +
            (W_OBV_1H  if obv_1h["signal"] else 0) +
            (W_OBV_4H  if obv_4h["signal"] else 0) +
            (W_CVD_4H  if cvd_4h["signal"] else 0) +
            (W_RSI_4H  if rsi["signal"]    else 0) +
            (W_FUNDING if fund["signal"]   else 0)
        )

        return {
            "symbol":   symbol,
            "score":    score,
            "price":    bb["close"],
            "pullback": pullback,
            "bb":       bb,
            "obv_1h":   obv_1h,
            "obv_4h":   obv_4h,
            "cvd_4h":   cvd_4h,
            "rsi":      rsi,
            "rsi_1h":   rsi_1h,
            "funding":  fund,
        }

    except Exception as e:
        log.warning(f"  {symbol}: skipped — {e}")
        return None


# ─── FORMAT TELEGRAM MESSAGE ─────────────────────────────────────────────────

# ─── FORMAT TELEGRAM MESSAGE ─────────────────────────────────────────────────

TELEGRAM_MAX_LEN = 4096


def format_candidate(r: dict) -> str:
    """Format a single candidate as a compact block."""
    s   = r["score"]
    pct = round(s / MAX_SCORE * 100)
    verdict = (
        "Strong"   if pct >= 88 else
        "Good"     if pct >= 70 else
        "Moderate"
    )

    bb_line    = "✅" if r["bb"]["signal"]     else "❌"
    obv1h_line = "✅" if r["obv_1h"]["signal"] else "❌"
    obv4h_line = "✅" if r["obv_4h"]["signal"] else "❌"
    cvd4h_line = "✅" if r["cvd_4h"]["signal"] else "❌"
    rsi_line   = "✅" if r["rsi"]["signal"]    else "❌"

    if r["funding"]["signal"]:
        fund_line = "✅"
    elif r["funding"]["positive"]:
        fund_line = "➖"
    else:
        fund_line = "❌"

    rsi_1h_line = ""
    if r.get("rsi_1h") is not None:
        v    = r["rsi_1h"]["value"]
        flag = "⚠️" if r["rsi_1h"]["above_85"] else "ℹ️"
        rsi_1h_line = f"\n{flag} RSI 1H: *{v}*"

    return (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*{r['symbol']}* — {s}/{MAX_SCORE} ({pct}%) — _{verdict}_\n"
        f"{bb_line} BB (4H)\n"
        f"{obv1h_line} OBV (1H)\n"
        f"{obv4h_line} OBV (4H)\n"
        f"{cvd4h_line} CVD (4H)\n"
        f"{rsi_line} RSI (4H)\n"
        f"{fund_line} Funding rate"
        f"{rsi_1h_line}"
    )


def build_messages(results: list[dict]) -> list[str]:
    """
    Build one or more Telegram messages, splitting automatically
    if total length exceeds TELEGRAM_MAX_LEN.
    """
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    total          = len(sorted_results)

    messages  = []
    header    = f"🔴 *{total} short candidate{'s' if total > 1 else ''}*\n"
    current   = header
    chunk_num = 1

    for r in sorted_results:
        block = format_candidate(r) + "\n"
        # If adding this block would exceed the limit, flush and start a new message
        if len(current) + len(block) > TELEGRAM_MAX_LEN:
            messages.append(current.rstrip())
            chunk_num += 1
            current = f"🔴 *{total} short candidates (cont.)*\n"
        current += block

    if current.strip():
        messages.append(current.rstrip())

    return messages


def format_no_results(total_symbols: int, duration_secs: float, next_scan: str) -> str:
    return (
        f"🔍 *No short candidates found*\n\n"
        f"📊 {total_symbols} symbols · {duration_secs:.0f}s · next scan {next_scan} UTC"
    )


def format_startup() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    next_scan = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")
    return (
        f"✅ *Short Trade Screener is live* — {now}\n\n"
        f"Scanning all USDT-M perpetuals every 4 hours.\n\n"
        f"*Settings*\n"
        f"• Alert threshold: {MIN_SCORE}/{MAX_SCORE}\n"
        f"• Signals: BB (4H), OBV (1H), OBV (4H), RSI (4H), Funding Rate\n\n"
        f"⏰ First scan starting now. Next at *{next_scan} UTC*"
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
            await asyncio.sleep(0.8)  # slightly longer pause to reduce 418 risk

    duration = time.monotonic() - scan_start
    next_scan = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M")

    log.info(f"Scan complete in {duration:.0f}s. {len(results)} candidate(s) found.")

    if results:
        messages = build_messages(results)
        for msg in messages:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
    else:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=format_no_results(len(symbols), duration, next_scan),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    log.info("Telegram message(s) sent.")


async def health_handler(request):
    return web.Response(text="OK")


async def run_health_server():
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info(f"Health server running on port {port}")
    # Keep running forever alongside the scanner
    while True:
        await asyncio.sleep(3600)


async def main():
    log.info(f"Short Trade Screener started. Scanning every {SCAN_INTERVAL // 3600}h, threshold {MIN_SCORE}/{MAX_SCORE}.")
    bot = Bot(token=TELEGRAM_TOKEN)

    # Send startup confirmation
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=format_startup(),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as e:
        log.error(f"Could not send startup message: {e}")

    async def scan_loop():
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

    # Run health server and scanner concurrently
    await asyncio.gather(
        run_health_server(),
        scan_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
