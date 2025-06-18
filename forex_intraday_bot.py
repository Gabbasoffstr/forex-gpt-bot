import json
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import os

# Загрузка переменных окружения
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
START_CAPITAL = 100
RISK_PER_TRADE = 0.01
LEVERAGE = 10
CHECK_INTERVAL = 300  # 5 минут
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]

last_signals = {pair: None for pair in PAIRS}
last_signal_time = {pair: None for pair in PAIRS}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Анализ сигнала по таймфрейму M5 с индикаторами =====
def get_intraday_signal(pair):
    try:
        df = yf.download(pair, period="2d", interval="5m")
        if df.empty:
            return None, None

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)

        last = df.iloc[-1]
        price = last['Close']
        ema20 = last['EMA_20']
        ema50 = last['EMA_50']
        rsi = last['RSI_14']
        time = df.index[-1]

        if last_signal_time[pair] == time:
            return None, price

        if price > ema20 > ema50 and rsi < 30:
            return "BUY", price
        elif price < ema20 < ema50 and rsi > 70:
            return "SELL", price
        return None, price
    except Exception as e:
        logger.error(f"Ошибка анализа {pair}: {e}")
        return None, None

# ===== Расчет объема =====
def calculate_lot(capital, risk, price, sl_pips, pair):
    pip_val = 0.01 if "JPY" in pair else 0.0001
    risk_amount = capital * risk
    lot = risk_amount / (sl_pips * pip_val * price)
    return round(lot * LEVERAGE, 4)

# ===== Отправка и логирование =====
def save_trade(pair, signal, price, sl, tp, lot, profit=0):
    trade = {
        "pair": pair, "signal": signal, "price": price,
        "sl": sl, "tp": tp, "lot": lot,
        "profit": profit, "timestamp": datetime.utcnow().isoformat()
    }
    try:
        with open("trades.json", "r") as f:
            trades = json.load(f)
    except:
        trades = []
    trades.append(trade)
    with open("trades.json", "w") as f:
        json.dump(trades, f, indent=2)

# ===== Проверка сигналов каждые 5 минут =====
async def check_signals(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    capital_per_pair = START_CAPITAL / len(PAIRS)
    messages = []

    for pair in PAIRS:
        signal, price = get_intraday_signal(pair)
        if not signal or signal == last_signals.get(pair):
            continue

        sl_pips = 30
        tp_pips = 60
        pip_val = 0.01 if "JPY" in pair else 0.0001

        sl = price - sl_pips * pip_val if signal == "BUY" else price + sl_pips * pip_val
        tp = price + tp_pips * pip_val if signal == "BUY" else price - tp_pips * pip_val
        lot = calculate_lot(capital_per_pair, RISK_PER_TRADE, price, sl_pips, pair)

        message = (
            f"\uD83D\uDD14 Сигнал: {pair}\n"
            f"Тип: {signal}\nЦена: {price:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}\n"
            f"Лот: {lot:.4f} (плечо 1:{LEVERAGE})\nРиск: ${capital_per_pair * RISK_PER_TRADE:.2f}"
        )
        messages.append(message)
        save_trade(pair, signal, price, sl, tp, lot)
        last_signals[pair] = signal
        last_signal_time[pair] = pd.Timestamp.utcnow()

    if messages:
        await context.bot.send_message(chat_id=chat_id, text="\n\n".join(messages))

# ===== Telegram команды =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.job_queue.run_repeating(check_signals, interval=CHECK_INTERVAL, first=1, data={"chat_id": chat_id})
    await update.message.reply_text("Бот запущен! Проверка сигналов каждые 5 минут.")

# ===== Запуск приложения =====
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()

if __name__ == "__main__":
    main()
