import json
import logging
import requests
import time
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict
from telethon.sync import TelegramClient
import asyncio
import uuid
from dotenv import load_dotenv
import os

# Загрузка переменных окружения
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL", "ApexBull")

# Проверка переменных окружения
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан в .env")
if not API_ID or not API_HASH:
    logging.warning("API_ID или API_HASH не заданы, парсинг Telegram будет отключен")

# Конфигурация
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
START_CAPITAL = 100
RISK_PER_TRADE = 0.005  # 0.5% риска
LEVERAGE = 10
CHECK_INTERVAL = 300  # 5 минут

# Хранилище сигналов
last_signals = {pair: None for pair in PAIRS}
last_signal_time = {pair: None for pair in PAIRS}
last_signal_id = {pair: None for pair in PAIRS}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Календарь новостей
def get_economic_calendar():
    try:
        response = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        events = response.json()
        return [(datetime.strptime(e["date"], "%Y-%m-%dT%H:%M:%S+00:00"),
                 datetime.strptime(e["date"], "%Y-%m-%dT%H:%M:%S+00:00") + timedelta(minutes=30))
                for e in events if e["impact"] in ["High", "Medium"]]
    except Exception as e:
        logger.warning(f"Ошибка загрузки календаря новостей: {e}")
        return []

# Проверка новостей
def is_news_time():
    now = datetime.utcnow()
    for start, end in get_economic_calendar():
        if start <= now <= end:
            return True
    return False

# Парсинг Telegram
async def parse_telegram_signals(pair):
    if not API_ID or not API_HASH:
        logger.warning("Парсинг Telegram отключен из-за отсутствия API_ID или API_HASH")
        return None
    try:
        async with TelegramClient('session', API_ID, API_HASH) as client:
            async for message in client.iter_messages(TELEGRAM_CHANNEL, limit=10):
                if pair.replace("=X", "") in message.text and ("BUY" in message.text or "SELL" in message.text):
                    return "BUY" if "BUY" in message.text else "SELL"
        return None
    except Exception as e:
        logger.error(f"Ошибка парсинга Telegram: {e}")
        return None

# Мультитаймфреймовый анализ
def get_intraday_signal(pair):
    try:
        # Загрузка данных
        data_5m = yf.download(pair, period="2d", interval="5m")
        data_15m = yf.download(pair, period="3d", interval="15m")
        data_30m = yf.download(pair, period="5d", interval="30m")
        data_1h = yf.download(pair, period="10d", interval="1h")
        if any(df.empty for df in [data_5m, data_15m, data_30m, data_1h]):
            logger.warning(f"Пустые данные для {pair}")
            return None, None, None, None

        # Индикаторы для 5m
        data_5m.ta.ema(length=20, append=True)
        data_5m.ta.ema(length=50, append=True)
        data_5m.ta.rsi(length=14, append=True)
        data_5m.ta.atr(length=14, append=True)

        # Индикаторы для 15m
        data_15m.ta.macd(append=True)

        # Индикаторы для 30m
        data_30m.ta.ema(length=50, append=True)

        # Индикаторы для 1h
        data_1h.ta.ema(length=200, append=True)
        data_1h.ta.adx(length=14, append=True)

        # Последние данные
        last_5m = data_5m.iloc[-1]
        last_15m = data_15m.iloc[-1]
        last_30m = data_30m.iloc[-1]
        last_1h = data_1h.iloc[-1]

        price = last_5m["Close"]
        ema20_5m = last_5m["EMA_20"]
        ema50_5m = last_5m["EMA_50"]
        rsi_5m = last_5m["RSI_14"]
        atr_5m = last_5m["ATRr_14"]
        macd_15m = last_15m["MACD_12_26_9"]
        ema50_30m = last_30m["EMA_50"]
        ema200_1h = last_1h["EMA_200"]
        adx_1h = last_1h["ADX_14"]
        candle_time = data_5m.index[-1]
        avg_atr = data_5m["ATRr_14"].mean()

        now_utc = datetime.utcnow()
        if not (7 <= now_utc.hour <= 17) or is_news_time() or atr_5m > avg_atr * 1.5:
            return None, price, atr_5m, None

        if last_signal_time[pair] == candle_time:
            return None, price, atr_5m, None

        signal_id = str(uuid.uuid4())
        if (price > ema20_5m > ema50_5m and rsi_5m < 35 and
            macd_15m > 0 and price > ema50_30m and price > ema200_1h and adx_1h > 25):
            return "BUY", price, atr_5m, signal_id
        elif (price < ema20_5m < ema50_5m and rsi_5m > 65 and
              macd_15m < 0 and price < ema50_30m and price < ema200_1h and adx_1h > 25):
            return "SELL", price, atr_5m, signal_id
        return None, price, atr_5m, None
    except Exception as e:
        logger.error(f"Ошибка анализа {pair}: {e}")
        return None, None, None, None

# Расчет объема
def calculate_lot(capital, risk, price, sl_distance, pair):
    pip_value = 0.0001 if "JPY" not in pair else 0.01
    risk_amount = capital * risk
    lot = risk_amount / (sl_distance * pip_value * price)
    return round(lot * LEVERAGE, 4)

# Сохранение сделки
def save_trade(pair, signal, price, sl, tp, lot, profit=0):
    trade = {
        "pair": pair, "signal": signal, "price": price,
        "sl": sl, "tp": tp, "lot": lot,
        "profit": profit, "timestamp": datetime.utcnow().isoformat()
    }
    try:
        try:
            with open("trades.json", "r") as f:
                trades = json.load(f)
        except FileNotFoundError:
            trades = []
        trades.append(trade)
        with open("trades.json", "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения сделки: {e}")

# Сохранение открытой сделки
def save_open_trade(pair, signal, price, sl, tp, lot, signal_id):
    trade = {
        "pair": pair, "signal": signal, "price": price,
        "sl": sl, "tp": tp, "lot": lot, "signal_id": signal_id,
        "open_time": datetime.utcnow().isoformat()
    }
    try:
        try:
            with open("open_trades.json", "r") as f:
                trades = json.load(f)
        except FileNotFoundError:
            trades = []
        trades.append(trade)
        with open("open_trades.json", "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения открытой сделки: {e}")

# Проверка закрытия сделок
async def check_trade_closures(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    try:
        with open("open_trades.json", "r") as f:
            open_trades = json.load(f)
    except FileNotFoundError:
        logger.info("Файл open_trades.json не найден")
        return
    except Exception as e:
        logger.error(f"Ошибка чтения open_trades.json: {e}")
        return

    closed = []
    updated_trades = []

    for trade in open_trades:
        data = yf.download(trade["pair"], period="1d", interval="1m")
        if data.empty:
            continue
        current_price = data["Close"].iloc[-1]
        atr = ta.atr(data["High"], data["Low"], data["Close"], length=14).iloc[-1]

        # Трейлинг-стоп
        if trade["signal"] == "BUY" and current_price > trade["price"]:
            trade["sl"] = max(trade["sl"], current_price - atr * 1.0)
        elif trade["signal"] == "SELL" and current_price < trade["price"]:
            trade["sl"] = min(trade["sl"], current_price + atr * 1.0)

        pip_value = 0.0001 if "JPY" not in trade["pair"] else 0.01
        if trade["signal"] == "BUY":
            if current_price >= trade["tp"] or current_price <= trade["sl"]:
                profit = (trade["tp"] - trade["price"]) / pip_value * trade["lot"] if current_price >= trade["tp"] else \
                         (trade["sl"] - trade["price"]) / pip_value * trade["lot"]
                closed.append((trade, profit))
            else:
                updated_trades.append(trade)
        else:
            if current_price <= trade["tp"] or current_price >= trade["sl"]:
                profit = (trade["price"] - trade["tp"]) / pip_value * trade["lot"] if current_price <= trade["tp"] else \
                         (trade["price"] - trade["sl"]) / pip_value * trade["lot"]
                closed.append((trade, profit))
            else:
                updated_trades.append(trade)

    try:
        with open("open_trades.json", "w") as f:
            json.dump(updated_trades, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка записи open_trades.json: {e}")

    for trade, profit in closed:
        save_trade(trade["pair"], trade["signal"], trade["price"], trade["sl"], trade["tp"], trade["lot"], profit)
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"\u2705 Сделка {trade['pair']} закрыта. Прибыль: ${profit:.2f}")
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")

# График статистики
def generate_stats_chart():
    try:
        with open("trades.json", "r") as f:
            trades = json.load(f)
    except FileNotFoundError:
        logger.info("Файл trades.json не найден")
        return None
    except Exception as e:
        logger.error(f"Ошибка чтения trades.json: {e}")
        return None

    labels = [t["timestamp"] for t in trades]
    profits = [t["profit"] for t in trades]
    cumulative_profit = [sum(profits[:i+1]) for i in range(len(profits))]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Статистика торговли</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <canvas id="profitChart" width="800" height="400"></canvas>
        <script>
            const ctx = document.getElementById('profitChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: 'Кумулятивная прибыль ($)',
                        data: {json.dumps(cumulative_profit)},
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.2)',
                        fill: true
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{ title: {{ display: true, text: 'Прибыль ($)' }} }},
                        x: {{ title: {{ display: true, text: 'Дата' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    try:
        with open("stats_chart.html", "w") as f:
            f.write(html_content)
        return "stats_chart.html"
    except Exception as e:
        logger.error(f"Ошибка записи stats_chart.html: {e}")
        return None

# Проверка сигналов
async def check_signals(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    capital_per_pair = START_CAPITAL / len(PAIRS)
    messages = []

    for pair in PAIRS:
        signal, price, atr, signal_id = get_intraday_signal(pair)
        if not signal or signal_id == last_signal_id.get(pair):
            continue

        telegram_signal = await parse_telegram_signals(pair)
        if telegram_signal != signal:
            continue

        sl_distance = round(atr * 1.5, 5)
        tp_distance = round(sl_distance * 2, 5)
        pip_factor = 0.0001 if "JPY" not in pair else 0.01
        sl = price - sl_distance if signal == "BUY" else price + sl_distance
        tp = price + tp_distance if signal == "BUY" else price - tp_distance
        lot = calculate_lot(capital_per_pair, RISK_PER_TRADE, price, sl_distance / pip_factor, pair)

        message = (
            f"\U0001F4B0 Сигнал: {pair}\n"
            f"Тип: {signal} (подтверждено @{TELEGRAM_CHANNEL}, мультитаймфрейм)\nЦена: {price:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}\n"
            f"Лот: {lot:.4f} (плечо 1:{LEVERAGE})\nРиск: ${capital_per_pair * RISK_PER_TRADE:.2f}"
        )
        messages.append(message)
        save_trade(pair, signal, price, sl, tp, lot)
        save_open_trade(pair, signal, price, sl, tp, lot, signal_id)
        last_signals[pair] = signal
        last_signal_time[pair] = pd.Timestamp.utcnow()
        last_signal_id[pair] = signal_id

    if messages:
        try:
            await context.bot.send_message(chat_id=chat_id, text="\n\n".join(messages))
        except Exception as e:
            logger.error(f"Ошибка отправки сигналов: {e}")

# Команды Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job_queue = context.application.job_queue

    if job_queue is None:
        logger.error("Job queue не инициализирован")
        await update.message.reply_text("Ошибка: очередь задач не работает.")
        return

    job_queue.run_repeating(check_signals, interval=CHECK_INTERVAL, first=1, data={"chat_id": chat_id})
    job_queue.run_repeating(check_trade_closures, interval=CHECK_INTERVAL, first=30, data={"chat_id": chat_id})
    await update.message.reply_text("Бот запущен! Проверка сигналов каждые 5 минут.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open("trades.json", "r") as f:
            trades = json.load(f)
        profit = sum(t["profit"] for t in trades)
        wins = len([t for t in trades if t["profit"] > 0])
        total = len(trades)
        winrate = wins / total * 100 if total else 0
        await update.message.reply_text(f"Сделок: {total}\nПрибыль: ${profit:.2f}\nWinrate: {winrate:.1f}%")
    except FileNotFoundError:
        await update.message.reply_text("Нет данных о сделках.")
    except Exception as e:
        logger.error(f"Ошибка чтения статистики: {e}")
        await update.message.reply_text("Ошибка при загрузке статистики.")

async def stats_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chart_file = generate_stats_chart()
    if chart_file:
        try:
            with open(chart_file, "rb") as f:
                await update.message.reply_document(document=f, filename="stats_chart.html",
                                                  caption="График прибыли/убытков")
        except Exception as e:
            logger.error(f"Ошибка отправки графика: {e}")
            await update.message.reply_text("Ошибка при отправке графика.")
    else:
        await update.message.reply_text("Нет данных для графика.")

# Запуск приложения
def main():
    logger.info(f"Запуск бота с токеном: {BOT_TOKEN[:10]}...")
    retry_count = 3
    for attempt in range(retry_count):
        try:
            app = Application.builder().token(BOT_TOKEN).build()
            logger.info("Application успешно инициализирован")
            app.add_handler(CommandHandler("start", start))
            app.add_handler(CommandHandler("stats", stats))
            app.add_handler(CommandHandler("stats_chart", stats_chart))
            app.run_polling(poll_interval=1, timeout=10)
            break
        except Conflict as e:
            logger.warning(f"Конфликт getUpdates (попытка {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(5)  # Ждем перед повторной попыткой
            else:
                logger.error("Не удалось устранить конфликт getUpdates")
                raise
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            raise