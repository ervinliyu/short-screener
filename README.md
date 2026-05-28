# Short Trade Screener Bot

Scans all Binance USDT-M perpetuals every 4 hours and sends a Telegram alert
when a symbol scores **12+/17** on the weighted signal matrix.

## Signals & Weights

| Signal | Weight | Logic |
|---|---|---|
| Bollinger Band upper (4H) | 4 | Close ≥ upper band (20 SMA, 2σ) |
| OBV divergence (1H) | 4 | Price HH, OBV LH over last 3 swing highs |
| OBV divergence (4H) | 4 | Same on 4H |
| RSI 4H | 3 | RSI > 70 OR bearish RSI divergence |
| Funding rate | 2 | Last funding rate > 0.01% |

**Max score: 17 — Alert threshold: 12**

---

## Setup (2 steps)

### Step 1 — Create your Telegram bot

1. Open Telegram and message **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** it gives you (looks like `123456:ABCdef...`)
4. Start a chat with your new bot (search its username and press Start)
5. Get your **chat ID** by messaging **@userinfobot** — it replies with your ID

### Step 2 — Run the bot

**Option A — Run locally with Python**

```bash
# Clone / copy the files, then:
pip install -r requirements.txt

export TELEGRAM_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"

python scanner.py
```

**Option B — Run with Docker (recommended for always-on)**

```bash
docker build -t short-screener .

docker run -d \
  --name short-screener \
  --restart unless-stopped \
  -e TELEGRAM_TOKEN="your_bot_token_here" \
  -e TELEGRAM_CHAT_ID="your_chat_id_here" \
  short-screener
```

**Option C — Deploy to a VPS (Railway / Render / Fly.io)**

1. Push the files to a GitHub repo (make sure `.env` is in `.gitignore`)
2. Connect the repo to Railway or Render
3. Set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` as environment variables in the dashboard
4. Deploy — it runs 24/7

---

## What the Telegram message looks like

```
🔴 Short Trade Screener — 2026-05-27 08:00 UTC

Found 2 candidate(s) with score ≥ 12/17

━━━━━━━━━━━━━━━━━━━━
BTCUSDT  |  Score: 15/17 (88%)  |  Strong setup
Price: 68420.5

✅ Bollinger Band upper touch (4H)  [w:4]
✅ OBV divergence (1H)  [w:4]
✅ OBV divergence (4H)  [w:4]
✅ RSI 4H (RSI 73.2 > 70)  [w:3]
❌ Funding rate elevated  [w:2]

👉 Binance Chart
...
⚠️ Screener only — always apply your own analysis before entering.
```

---

## Adjusting thresholds

Edit the top of `scanner.py`:

```python
MIN_SCORE      = 12      # raise to 15 for strong-only alerts
SCAN_INTERVAL  = 4*3600  # change to 3600 for hourly scans
RSI_OB         = 70      # RSI overbought level
FUNDING_THRESH = 0.0001  # 0.01% per 8h funding threshold
```

---

## Notes

- No Binance API key needed — all endpoints used are public
- Scanning ~300+ symbols takes ~2-3 minutes; batched to stay within rate limits
- CVD and Exhaustion are intentionally excluded — apply those manually on the chart
- The bot does not place trades; it only sends notifications
