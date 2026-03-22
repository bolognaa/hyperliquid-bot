import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HL_PRIVATE_KEY = os.getenv("HL_PRIVATE_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8348882695:AAFoaEfwxTE2NPzNCFgnNKieGU15HwSW1qM")
TELEGRAM_CHAT_ID_FILE = ".chat_id"

# --- Hyperliquid ---
HL_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

# --- Paper Trading ---
STARTING_BALANCE = 50.0
ASSETS = ["BTC", "ETH", "SOL"]
MAX_LEVERAGE = 20
MAX_POSITION_PCT = 0.5
STOP_LOSS_BALANCE_PCT = 0.02   # max loss per trade = 2% of total balance
TAKE_PROFIT_MULTIPLIER = 3.0   # TP distance = 3x SL distance
MAX_DRAWDOWN_PCT = 0.40        # stop if balance drops 40% below start
MIN_BALANCE_ALERT = 25.0       # Telegram alert when below this
MAX_SAME_DIRECTION = 3         # max open positions in same direction

# --- Kraken API ---
KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
KRAKEN_PAIRS = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "SOL": "SOLUSD",
}
OHLC_INTERVAL = 60   # 1h candles
OHLC_CANDLES = 200   # number of candles to fetch

# --- OpenRouter / AI ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AI_MODEL = "minimax/minimax-m2.7"

# --- Dashboard ---
DASHBOARD_PORT = 3456
DASHBOARD_HOST = "0.0.0.0"

# --- Logging ---
LOG_FILE = "logs/bot.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# --- Timing ---
MARKET_DATA_INTERVAL = 60       # seconds per trading cycle
POSITION_MONITOR_INTERVAL = 30  # seconds between SL/TP checks
STATE_SAVE_INTERVAL = 60        # seconds between state persistence
OPTIMIZER_TRADE_THRESHOLD = 10  # trades before each optimization attempt

# --- State ---
STATE_FILE = "state.json"


class TunableParams:
    """Mutable parameters that the Optimizer Agent can adjust."""

    BOUNDS = {
        "RSI_OVERSOLD":              (20,  40),
        "RSI_OVERBOUGHT":            (60,  80),
        "EMA_FAST":                  (5,   15),
        "EMA_SLOW":                  (15,  30),
        "STOP_LOSS_BALANCE_PCT":     (0.01, 0.05),
        "AI_CONFIDENCE_THRESHOLD":   (0.5, 0.85),
        "MAX_POSITION_PCT":          (0.2, 0.5),
        "TAKE_PROFIT_MULTIPLIER":    (2.0, 5.0),
    }

    def __init__(self):
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.EMA_FAST = 9
        self.EMA_SLOW = 21
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        self.BB_PERIOD = 20
        self.BB_STD = 2
        self.ATR_PERIOD = 14
        self.AI_CONFIDENCE_THRESHOLD = 0.6
        self.STOP_LOSS_BALANCE_PCT = STOP_LOSS_BALANCE_PCT
        self.MAX_POSITION_PCT = MAX_POSITION_PCT
        self.TAKE_PROFIT_MULTIPLIER = TAKE_PROFIT_MULTIPLIER

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        if key in self.BOUNDS:
            lo, hi = self.BOUNDS[key]
            value = max(lo, min(hi, value))
        setattr(self, key, value)

    def snapshot(self):
        return {k: self.get(k) for k in self.BOUNDS}


# Singleton used by all agents
TUNABLE = TunableParams()
