import os
import time
import math
import json
import hmac
import hashlib
import threading
from datetime import datetime
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import HistGradientBoostingClassifier
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

STATE_FILE = "apexml_v1_state.json"
GATE_HOST = "https://api.gateio.ws"
GATE_PREFIX = "/api/v4"

TRADING_COINS = ["SUI_USDT", "SOL_USDT", "ARB_USDT"]
MACRO_COIN = "BTC_USDT"
HORIZON_HOURS = 16
RETRAIN_INTERVAL_SEC = 6 * 3600  # Retrains ML models every 6 hours

HTTP_SESSION = requests.Session()
adapter = HTTPAdapter(
    pool_connections=16,
    pool_maxsize=16,
    max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[502, 503, 504])
)
HTTP_SESSION.mount("https://", adapter)

CONTRACT_META_CACHE = {}
LIVE_MODELS = {}  # In-memory trained models: { pair: {"model": clf, "classes": list} }

DEFAULT_STATE = {
    "mode": "PAPER",
    "slot_size_usdt": 10.0,
    "leverage": 2,
    "tp_pct": 4.0,                  # +4.0% price move (+8.0% on margin at 2x)
    "sl_pct": 2.0,                  # -2.0% price move (-4.0% on margin at 2x)
    "confidence_threshold": 65.0,   # Minimum ML probability (%) to enter
    "max_consecutive_losses": 2,    # Circuit breaker trigger
    "futures_fee_pct": 0.12,        # Gate.io Futures round-trip fee buffer
    "paper_balance_usdt": 15.0,
    "paper_starting_usdt": 15.0,
    "live_futures_usdt": 0.0,
    "api_key": os.environ.get("GATE_API_KEY", "").strip(),
    "api_secret": os.environ.get("GATE_API_SECRET", "").strip(),
    "api_connected": False,
    "api_status_msg": "Paper Mode Active — Connect Gate.io Futures Keys When Ready",
    "consecutive_losses": 0,
    "circuit_breaker_until": 0.0,
    "last_trained_ts": 0.0,
    "last_trained_str": "Not Trained Yet",
    "model_reports": {},            # Per-coin walk-forward backtest + live probabilities
    "active_positions": {},         # Currently open trades
    "trade_history": [],            # Closed trades
    "stats": {
        "paper_trades": 0,
        "paper_wins": 0,
        "paper_pnl_usdt": 0.0,
        "live_trades": 0,
        "live_wins": 0,
        "live_pnl_usdt": 0.0
    },
    "logs": [],
    "cycle_count": 0,
    "last_update": "--:--:--",
    "last_worker_heartbeat": 0.0
}

state_lock = threading.Lock()
cycle_lock = threading.Lock()
worker_started = False
STATE = json.loads(json.dumps(DEFAULT_STATE))


def fmt_px(val):
    v = float(val or 0.0)
    if v == 0:
        return "$0.0000"
    if v < 0.001:
        return f"${v:.6f}"
    if v < 1.0:
        return f"${v:.4f}"
    return f"${v:.3f}"


def add_log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    STATE["logs"].insert(0, f"[{ts}] {msg}")
    STATE["logs"] = STATE["logs"][:100]


def load_state():
    global STATE
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                sk = (saved.get("api_key") or "").strip()
                ss = (saved.get("api_secret") or "").strip()
                STATE.update(saved)
                if sk:
                    STATE["api_key"] = sk
                if ss:
                    STATE["api_secret"] = ss
        except Exception as e:
            add_log(f"State load notice: {e}")

    ek = os.environ.get("GATE_API_KEY", "").strip()
    es = os.environ.get("GATE_API_SECRET", "").strip()
    if ek:
        STATE["api_key"] = ek
    if es:
        STATE["api_secret"] = es


def save_state():
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(STATE, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass


# =============================================================================
# GATE.IO FUTURES APIv4 HELPERS
# =============================================================================
def gate_sign(method, url, query_string="", body_string="", custom_key=None, custom_secret=None):
    key = (custom_key if custom_key else STATE.get("api_key", "")).strip()
    secret = (custom_secret if custom_secret else STATE.get("api_secret", "")).strip()
    t = str(int(time.time()))
    hashed_payload = hashlib.sha512(body_string.encode("utf-8")).hexdigest()
    s = f"{method}\n{url}\n{query_string}\n{hashed_payload}\n{t}"
    sign = hmac.new(secret.encode("utf-8"), s.encode("utf-8"), hashlib.sha512).hexdigest()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "KEY": key,
        "Timestamp": t,
        "SIGN": sign
    }


def fetch_futures_candles(contract, interval="1h", limit=900):
    r = HTTP_SESSION.get(
        f"{GATE_HOST}{GATE_PREFIX}/futures/usdt/candlesticks",
        params={"contract": contract, "interval": interval, "limit": limit},
        timeout=8
    )
    r.raise_for_status()
    raw = r.json()
    candles = []
    for c in raw:
        if "c" in c and "t" in c:
            candles.append({
                "t": int(float(c["t"])),
                "o": float(c["o"]),
                "h": float(c["h"]),
                "l": float(c["l"]),
                "c": float(c["c"]),
                "v": float(c.get("v") or 0.0)
            })
    candles.sort(key=lambda x: x["t"])
    return candles


def fetch_futures_tickers(pairs):
    r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/futures/usdt/tickers", timeout=6)
    r.raise_for_status()
    wanted = set(pairs)
    out = {}
    for t in r.json():
        c = t.get("contract")
        if c in wanted:
            last = float(t.get("last") or 0.0)
            if last > 0:
                out[c] = {
                    "last": last,
                    "change_24h": float(t.get("change_percentage") or 0.0)
                }
    return out


def get_contract_meta(contract):
    now = time.time()
    cached = CONTRACT_META_CACHE.get(contract)
    if not cached or (now - cached["ts"] > 3600):
        r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/futures/usdt/contracts/{contract}", timeout=5)
        if r.status_code == 200:
            info = r.json()
            cached = {"quanto_multiplier": float(info.get("quanto_multiplier") or 1.0), "ts": now}
        else:
            cached = {"quanto_multiplier": 1.0, "ts": now}
        CONTRACT_META_CACHE[contract] = cached
    return cached


def fetch_futures_usdt_balance(custom_key=None, custom_secret=None):
    url = f"{GATE_PREFIX}/futures/usdt/accounts"
    headers = gate_sign("GET", url, "", "", custom_key, custom_secret)
    r = HTTP_SESSION.get(f"{GATE_HOST}{url}", headers=headers, timeout=5)
    if r.status_code != 200:
        raise ValueError(f"Gate.io HTTP {r.status_code}: {r.text[:80]}")
    return float(r.json().get("available") or 0.0)


def set_isolated_leverage(contract, leverage):
    try:
        url = f"{GATE_PREFIX}/futures/usdt/positions/{contract}/leverage"
        qs = f"leverage={int(leverage)}"
        headers = gate_sign("POST", url, qs, "")
        HTTP_SESSION.post(f"{GATE_HOST}{url}?{qs}", headers=headers, timeout=4)
    except Exception:
        pass


def place_futures_order(contract, size_contracts, reduce_only=False):
    url = f"{GATE_PREFIX}/futures/usdt/orders"
    body = {"contract": contract, "size": int(size_contracts), "price": "0", "tif": "ioc"}
    if reduce_only:
        body["reduce_only"] = True
    body_str = json.dumps(body)
    headers = gate_sign("POST", url, "", body_str)
    r = HTTP_SESSION.post(f"{GATE_HOST}{url}", headers=headers, data=body_str, timeout=7)
    return r.status_code, r.json()


# =============================================================================
# TECHNICAL INDICATORS & ML FEATURE ENGINEERING (ZERO LEAKAGE)
# =============================================================================
def ema_array(arr, span):
    out = np.zeros_like(arr, dtype=float)
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi_array(closes, period=14):
    out = np.full(len(closes), 50.0, dtype=float)
    if len(closes) <= period:
        return out
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, len(closes)):
        if i > period:
            avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
            avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        if avg_l == 0:
            out[i] = 100.0
        else:
            rs = avg_g / avg_l
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr_array(highs, lows, closes, period=14):
    tr = np.zeros(len(closes), dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    return ema_array(tr, period)


def build_feature_matrix(candles, btc_lookup):
    n = len(candles)
    closes = np.array([c["c"] for c in candles], dtype=float)
    opens = np.array([c["o"] for c in candles], dtype=float)
    highs = np.array([c["h"] for c in candles], dtype=float)
    lows = np.array([c["l"] for c in candles], dtype=float)
    vols = np.array([c["v"] for c in candles], dtype=float)

    ema9 = ema_array(closes, 9)
    ema21 = ema_array(closes, 21)
    ema50 = ema_array(closes, 50)
    rsi14 = rsi_array(closes, 14)
    atr14 = atr_array(highs, lows, closes, 14)

    X = np.zeros((n, 12), dtype=float)
    for i in range(50, n):
        c_now = max(1e-12, closes[i])
        ret_1 = (c_now - closes[i - 1]) / max(1e-12, closes[i - 1])
        ret_3 = (c_now - closes[i - 3]) / max(1e-12, closes[i - 3])
        ret_6 = (c_now - closes[i - 6]) / max(1e-12, closes[i - 6])
        ret_12 = (c_now - closes[i - 12]) / max(1e-12, closes[i - 12])
        ema_sp = (ema9[i] - ema21[i]) / max(1e-12, ema21[i])
        dist_50 = (c_now - ema50[i]) / max(1e-12, ema50[i])
        rsi_n = rsi14[i] / 100.0
        atr_p = atr14[i] / c_now
        avg_v = np.mean(vols[max(0, i - 20):i]) + 1e-9
        vol_r = min(5.0, vols[i] / avg_v)
        rng = max(1e-12, highs[i] - lows[i])
        body = (c_now - opens[i]) / rng

        btc_ctx = btc_lookup.get(candles[i]["t"], (0.0, 0.0))
        X[i] = [
            ret_1, ret_3, ret_6, ret_12,
            ema_sp, dist_50, rsi_n, atr_p,
            vol_r, body, btc_ctx[0], btc_ctx[1]
        ]

    return X, closes, highs, lows


def build_triple_barrier_labels(closes, highs, lows, tp_pct, sl_pct, horizon=HORIZON_HOURS):
    """
    Class 1 = Clean LONG win (+tp_pct hit before -sl_pct)
    Class 2 = Clean SHORT win (-tp_pct hit before +sl_pct)
    Class 0 = HOLD / CHOP (neither hit cleanly)
    """
    n = len(closes)
    y = np.zeros(n, dtype=int)
    tp_mult = tp_pct / 100.0
    sl_mult = sl_pct / 100.0

    for i in range(50, n - horizon):
        ep = closes[i]
        long_tp = ep * (1.0 + tp_mult)
        long_sl = ep * (1.0 - sl_mult)
        short_tp = ep * (1.0 - tp_mult)
        short_sl = ep * (1.0 + sl_mult)

        long_res = 0
        short_res = 0

        for j in range(i + 1, i + 1 + horizon):
            if long_res == 0:
                if lows[j] <= long_sl:
                    long_res = -1
                elif highs[j] >= long_tp:
                    long_res = 1
            if short_res == 0:
                if highs[j] >= short_sl:
                    short_res = -1
                elif lows[j] <= short_tp:
                    short_res = 1
            if long_res != 0 and short_res != 0:
                break

        if long_res == 1 and short_res != 1:
            y[i] = 1
        elif short_res == 1 and long_res != 1:
            y[i] = 2
        else:
            y[i] = 0

    return y


def extract_class_probs(clf, row_2d):
    probs = clf.predict_proba(row_2d)[0]
    p_map = {int(cls): float(prob) for cls, prob in zip(clf.classes_, probs)}
    return p_map.get(0, 0.0), p_map.get(1, 0.0), p_map.get(2, 0.0)


def train_and_evaluate_coin(pair, btc_lookup, tp_pct, sl_pct, leverage, conf_thresh, fee_pct):
    candles = fetch_futures_candles(pair, interval="1h", limit=900)
    if len(candles) < 200:
        raise ValueError(f"Not enough 1h candles for {pair}")

    X, closes, highs, lows = build_feature_matrix(candles, btc_lookup)
    y = build_triple_barrier_labels(closes, highs, lows, tp_pct, sl_pct, HORIZON_HOURS)

    start_idx = 50
    end_labeled = len(candles) - HORIZON_HOURS
    total_samples = end_labeled - start_idx
    split_idx = start_idx + int(total_samples * 0.75)

    X_train, y_train = X[start_idx:split_idx], y[start_idx:split_idx]
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Insufficient class variety for {pair}")

    # 1) Walk-Forward Validation Model (trains ONLY on first 75%)
    val_clf = HistGradientBoostingClassifier(
        max_iter=80,
        max_depth=4,
        learning_rate=0.06,
        l2_regularization=1.2,
        random_state=42
    )
    val_clf.fit(X_train, y_train)

    # Simulate real trades on the unseen last 25% of candles
    bt_trades = 0
    bt_wins = 0
    bt_net_pct = 0.0
    conf_dec = conf_thresh / 100.0
    i = split_idx
    while i < end_labeled:
        _, p_long, p_short = extract_class_probs(val_clf, X[i:i + 1])
        side = None
        if p_long >= conf_dec and p_long > p_short:
            side = "LONG"
        elif p_short >= conf_dec and p_short > p_long:
            side = "SHORT"

        if side:
            bt_trades += 1
            ep = closes[i]
            outcome_pct = 0.0
            exit_j = min(len(candles) - 1, i + HORIZON_HOURS)
            for j in range(i + 1, exit_j + 1):
                if side == "LONG":
                    if lows[j] <= ep * (1.0 - sl_pct / 100.0):
                        outcome_pct = -sl_pct
                        exit_j = j
                        break
                    if highs[j] >= ep * (1.0 + tp_pct / 100.0):
                        outcome_pct = tp_pct
                        exit_j = j
                        break
                else:
                    if highs[j] >= ep * (1.0 + sl_pct / 100.0):
                        outcome_pct = -sl_pct
                        exit_j = j
                        break
                    if lows[j] <= ep * (1.0 - tp_pct / 100.0):
                        outcome_pct = tp_pct
                        exit_j = j
                        break
            else:
                move = ((closes[exit_j] - ep) / ep) * 100.0
                outcome_pct = move if side == "LONG" else -move

            leveraged_net = (outcome_pct * leverage) - (fee_pct * leverage)
            bt_net_pct += leveraged_net
            if leveraged_net > 0:
                bt_wins += 1
            i = exit_j + 1
        else:
            i += 1

    # 2) Final Production Model (retrained on ALL labeled data up to latest closed horizon)
    prod_clf = HistGradientBoostingClassifier(
        max_iter=80,
        max_depth=4,
        learning_rate=0.06,
        l2_regularization=1.2,
        random_state=42
    )
    prod_clf.fit(X[start_idx:end_labeled], y[start_idx:end_labeled])
    LIVE_MODELS[pair] = prod_clf

    # Live prediction on the most recent closed candle (index -2 to avoid repainting)
    p_hold, p_long, p_short = extract_class_probs(prod_clf, X[-2:-1])
    sig = "HOLD"
    if p_long * 100.0 >= conf_thresh and p_long > p_short:
        sig = "LONG"
    elif p_short * 100.0 >= conf_thresh and p_short > p_long:
        sig = "SHORT"

    win_rate = round((bt_wins / bt_trades) * 100.0, 1) if bt_trades > 0 else 0.0
    return {
        "pair": pair,
        "live_price": closes[-1],
        "prob_long": round(p_long * 100.0, 1),
        "prob_short": round(p_short * 100.0, 1),
        "prob_hold": round(p_hold * 100.0, 1),
        "signal": sig,
        "bt_trades": bt_trades,
        "bt_wins": bt_wins,
        "bt_win_rate": win_rate,
        "bt_net_pct": round(bt_net_pct, 2),
        "samples": total_samples
    }


def build_btc_lookup():
    btc_candles = fetch_futures_candles(MACRO_COIN, interval="1h", limit=920)
    closes = np.array([c["c"] for c in btc_candles], dtype=float)
    ema9 = ema_array(closes, 9)
    ema21 = ema_array(closes, 21)
    lookup = {}
    for i in range(21, len(btc_candles)):
        ret_3 = (closes[i] - closes[i - 3]) / max(1e-12, closes[i - 3])
        ema_sp = (ema9[i] - ema21[i]) / max(1e-12, ema21[i])
        lookup[btc_candles[i]["t"]] = (ret_3, ema_sp)
    return lookup


def retrain_all_models():
    with state_lock:
        tp_pct = float(STATE.get("tp_pct", 4.0))
        sl_pct = float(STATE.get("sl_pct", 2.0))
        leverage = int(STATE.get("leverage", 2))
        conf_thresh = float(STATE.get("confidence_threshold", 65.0))
        fee_pct = float(STATE.get("futures_fee_pct", 0.12))

    try:
        btc_lookup = build_btc_lookup()
        reports = {}
        for pair in TRADING_COINS:
            try:
                rep = train_and_evaluate_coin(
                    pair, btc_lookup, tp_pct, sl_pct, leverage, conf_thresh, fee_pct
                )
                reports[pair] = rep
            except Exception as e:
                with state_lock:
                    add_log(f"⚠️ Train notice [{pair}]: {str(e)[:65]}")

        with state_lock:
            STATE["model_reports"] = reports
            STATE["last_trained_ts"] = time.time()
            STATE["last_trained_str"] = datetime.utcnow().strftime("%H:%M:%S UTC")
            add_log(
                f"🧠 Trained Gradient Boosting models on {len(reports)} coins (1h Gate.io Futures + BTC Macro Context)."
            )
            save_state()
    except Exception as e:
        with state_lock:
            add_log(f"⚠️ Retrain error: {str(e)[:70]}")


# =============================================================================
# LIVE & PAPER TRADING ENGINE
# =============================================================================
def run_engine_cycle(force_manual=False):
    if not cycle_lock.acquire(blocking=force_manual):
        return

    try:
        now_ts = time.time()
        with state_lock:
            need_train = (not LIVE_MODELS) or (now_ts - float(STATE.get("last_trained_ts", 0.0)) > RETRAIN_INTERVAL_SEC)
            mode = STATE.get("mode", "PAPER")
            has_keys = bool(STATE.get("api_key") and STATE.get("api_secret"))
            slot_size = float(STATE.get("slot_size_usdt", 10.0))
            leverage = int(STATE.get("leverage", 2))
            tp_pct = float(STATE.get("tp_pct", 4.0))
            sl_pct = float(STATE.get("sl_pct", 2.0))
            conf_thresh = float(STATE.get("confidence_threshold", 65.0))
            fee_pct = float(STATE.get("futures_fee_pct", 0.12))
            max_losses = int(STATE.get("max_consecutive_losses", 2))
            cb_until = float(STATE.get("circuit_breaker_until", 0.0))

        if need_train:
            retrain_all_models()

        if has_keys:
            try:
                fut_usdt = fetch_futures_usdt_balance()
                with state_lock:
                    STATE["api_connected"] = True
                    STATE["live_futures_usdt"] = fut_usdt
                    STATE["api_status_msg"] = f"Gate.io Futures Connected (Available: ${fut_usdt:.2f} USDT)"
            except Exception as e:
                with state_lock:
                    STATE["api_connected"] = False
                    STATE["api_status_msg"] = f"API Notice: {str(e)[:55]}"

        try:
            tk_map = fetch_futures_tickers(TRADING_COINS)
        except Exception:
            return

        # 1) MONITOR ACTIVE POSITIONS FOR TP / SL / 16h TIME-STOP
        for pair in list(STATE["active_positions"].keys()):
            with state_lock:
                pos = STATE["active_positions"].get(pair)
            if not pos or pair not in tk_map:
                continue

            live_px = tk_map[pair]["last"]
            ep = float(pos["entry_price"])
            side = pos["side"]
            margin = float(pos["margin_usdt"])
            lev = int(pos["leverage"])
            tp_px = float(pos["tp_price"])
            sl_px = float(pos["sl_price"])

            raw_move_pct = ((live_px - ep) / ep) * 100.0
            dir_move_pct = raw_move_pct if side == "LONG" else -raw_move_pct
            net_pnl_pct = (dir_move_pct * lev) - (fee_pct * lev)
            net_pnl_usdt = margin * (net_pnl_pct / 100.0)

            with state_lock:
                pos["last_price"] = live_px
                pos["pnl_pct"] = round(net_pnl_pct, 2)
                pos["pnl_usdt"] = round(net_pnl_usdt, 4)

            hit_reason = None
            if side == "LONG":
                if live_px >= tp_px:
                    hit_reason = "TAKE PROFIT"
                elif live_px <= sl_px:
                    hit_reason = "STOP LOSS"
            else:
                if live_px <= tp_px:
                    hit_reason = "TAKE PROFIT"
                elif live_px >= sl_px:
                    hit_reason = "STOP LOSS"

            if not hit_reason and (now_ts - float(pos.get("opened_ts", now_ts))) >= (HORIZON_HOURS * 3600):
                hit_reason = "16H TIME EXIT"

            if hit_reason:
                close_position(pair, live_px, hit_reason)

        # 2) CHECK FOR NEW HIGH-CONFIDENCE ML ENTRIES
        breaker_active = now_ts < cb_until
        if not breaker_active:
            with state_lock:
                reports = STATE.get("model_reports", {})
                active_keys = set(STATE["active_positions"].keys())

            for pair in TRADING_COINS:
                if pair in active_keys or pair not in tk_map or pair not in reports:
                    continue

                rep = reports[pair]
                live_px = tk_map[pair]["last"]
                with state_lock:
                    rep["live_price"] = live_px

                sig = rep.get("signal", "HOLD")
                if sig not in ("LONG", "SHORT"):
                    continue

                conf_val = rep["prob_long"] if sig == "LONG" else rep["prob_short"]
                if conf_val < conf_thresh:
                    continue

                # Open Trade (Paper or Live)
                if mode == "PAPER":
                    with state_lock:
                        avail = float(STATE.get("paper_balance_usdt", 0.0))
                        if avail < slot_size:
                            continue
                        STATE["paper_balance_usdt"] = round(avail - slot_size, 4)
                        tp_px = live_px * (1.0 + tp_pct / 100.0) if sig == "LONG" else live_px * (1.0 - tp_pct / 100.0)
                        sl_px = live_px * (1.0 - sl_pct / 100.0) if sig == "LONG" else live_px * (1.0 + sl_pct / 100.0)
                        STATE["active_positions"][pair] = {
                            "pair": pair,
                            "is_live": False,
                            "side": sig,
                            "leverage": leverage,
                            "margin_usdt": slot_size,
                            "entry_price": live_px,
                            "last_price": live_px,
                            "tp_price": tp_px,
                            "sl_price": sl_px,
                            "confidence": conf_val,
                            "pnl_pct": round(-fee_pct * leverage, 2),
                            "pnl_usdt": round(-slot_size * (fee_pct * leverage / 100.0), 4),
                            "opened_ts": now_ts,
                            "opened_str": datetime.utcnow().strftime("%H:%M:%S")
                        }
                        add_log(
                            f"⚡ [PAPER {sig}][{pair}] {leverage}x ${slot_size:.2f} @ {fmt_px(live_px)} "
                            f"(Conf: {conf_val}% | TP: {fmt_px(tp_px)} | SL: {fmt_px(sl_px)})"
                        )
                elif mode == "LIVE" and has_keys:
                    avail_fut = float(STATE.get("live_futures_usdt", 0.0))
                    if avail_fut < slot_size:
                        continue
                    meta = get_contract_meta(pair)
                    notional = slot_size * leverage
                    c_val = live_px * meta["quanto_multiplier"]
                    num_c = max(1, int(round(notional / max(1e-9, c_val))))
                    signed_c = num_c if sig == "LONG" else -num_c
                    set_isolated_leverage(pair, leverage)
                    code, resp = place_futures_order(pair, signed_c, reduce_only=False)
                    if code in (200, 201):
                        actual_margin = round((num_c * c_val) / leverage, 4)
                        tp_px = live_px * (1.0 + tp_pct / 100.0) if sig == "LONG" else live_px * (1.0 - tp_pct / 100.0)
                        sl_px = live_px * (1.0 - sl_pct / 100.0) if sig == "LONG" else live_px * (1.0 + sl_pct / 100.0)
                        with state_lock:
                            STATE["live_futures_usdt"] = max(0.0, avail_fut - actual_margin)
                            STATE["active_positions"][pair] = {
                                "pair": pair,
                                "is_live": True,
                                "side": sig,
                                "leverage": leverage,
                                "contracts": num_c,
                                "margin_usdt": actual_margin,
                                "entry_price": live_px,
                                "last_price": live_px,
                                "tp_price": tp_px,
                                "sl_price": sl_px,
                                "confidence": conf_val,
                                "pnl_pct": round(-fee_pct * leverage, 2),
                                "pnl_usdt": round(-actual_margin * (fee_pct * leverage / 100.0), 4),
                                "opened_ts": now_ts,
                                "opened_str": datetime.utcnow().strftime("%H:%M:%S")
                            }
                            add_log(
                                f"🚀 [LIVE {sig}][{pair}] {leverage}x {num_c} contracts @ {fmt_px(live_px)} (Conf: {conf_val}%)"
                            )

        with state_lock:
            STATE["cycle_count"] = STATE.get("cycle_count", 0) + 1
            STATE["last_update"] = datetime.utcnow().strftime("%H:%M:%S")
            STATE["last_worker_heartbeat"] = time.time()
            save_state()

    finally:
        cycle_lock.release()


def close_position(pair, exit_px, reason):
    with state_lock:
        pos = STATE["active_positions"].pop(pair, None)
        if not pos:
            return
        fee_pct = float(STATE.get("futures_fee_pct", 0.12))
        max_losses = int(STATE.get("max_consecutive_losses", 2))
        is_live = bool(pos.get("is_live"))

    ep = float(pos["entry_price"])
    side = pos["side"]
    lev = int(pos["leverage"])
    margin = float(pos["margin_usdt"])

    if is_live and STATE.get("api_key"):
        contracts = int(pos.get("contracts", 1))
        close_size = -contracts if side == "LONG" else contracts
        place_futures_order(pair, close_size, reduce_only=True)

    raw_move = ((exit_px - ep) / ep) * 100.0
    dir_move = raw_move if side == "LONG" else -raw_move
    net_pct = (dir_move * lev) - (fee_pct * lev)
    net_usdt = round(margin * (net_pct / 100.0), 4)
    is_win = net_usdt > 0

    with state_lock:
        if not is_live:
            STATE["paper_balance_usdt"] = round(max(0.0, STATE.get("paper_balance_usdt", 0.0) + margin + net_usdt), 4)
            STATE["stats"]["paper_trades"] += 1
            if is_win:
                STATE["stats"]["paper_wins"] += 1
            STATE["stats"]["paper_pnl_usdt"] = round(STATE["stats"]["paper_pnl_usdt"] + net_usdt, 4)
        else:
            STATE["stats"]["live_trades"] += 1
            if is_win:
                STATE["stats"]["live_wins"] += 1
            STATE["stats"]["live_pnl_usdt"] = round(STATE["stats"]["live_pnl_usdt"] + net_usdt, 4)

        if is_win:
            STATE["consecutive_losses"] = 0
        else:
            STATE["consecutive_losses"] = int(STATE.get("consecutive_losses", 0)) + 1
            if STATE["consecutive_losses"] >= max_losses:
                STATE["circuit_breaker_until"] = time.time() + (24 * 3600)
                add_log(f"🛑 CIRCUIT BREAKER TRIGGERED: {max_losses} consecutive losses — paused for 24 hours.")

        STATE["trade_history"].insert(0, {
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "pair": pair,
            "mode": "LIVE" if is_live else "PAPER",
            "side": side,
            "entry": fmt_px(ep),
            "exit": fmt_px(exit_px),
            "reason": reason,
            "pnl_pct": round(net_pct, 2),
            "pnl_usdt": net_usdt
        })
        STATE["trade_history"] = STATE["trade_history"][:50]
        sign = "+" if net_usdt >= 0 else ""
        add_log(f"{'✅' if is_win else '🔻'} [{reason}][{pair} {side}] {fmt_px(ep)} → {fmt_px(exit_px)} | Net: {sign}${net_usdt:.4f} ({sign}{net_pct:.2f}%)")
        save_state()


def background_worker():
    load_state()
    add_log("🧠 ApexML v1.0 PRO Online — 1h Gradient Boosting Engine (Walk-Forward Validated).")
    while True:
        run_engine_cycle(force_manual=False)
        time.sleep(10.0)


def ensure_worker_alive():
    global worker_started
    now = time.time()
    if not worker_started or (now - STATE.get("last_worker_heartbeat", 0) > 35):
        worker_started = True
        STATE["last_worker_heartbeat"] = now
        threading.Thread(target=background_worker, daemon=True).start()


@app.before_request
def check_worker():
    ensure_worker_alive()


# =============================================================================
# PWA MOBILE INSTALLER ENDPOINTS
# =============================================================================
@app.route("/manifest.json")
def pwa_manifest():
    return jsonify({
        "name": "ApexML v1.0 PRO",
        "short_name": "ApexML",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#06070c",
        "theme_color": "#6366f1",
        "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}]
    })


@app.route("/icon.svg")
def pwa_icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
      <rect width="512" height="512" rx="115" fill="#090b14"/>
      <circle cx="256" cy="256" r="205" fill="none" stroke="#6366f1" stroke-width="18"/>
      <path d="M145 335 L256 150 L367 335" fill="none" stroke="#00f5a0" stroke-width="34" stroke-linecap="round" stroke-linejoin="round"/>
      <line x1="190" y1="275" x2="322" y2="275" stroke="#38bdf8" stroke-width="26" stroke-linecap="round"/>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/sw.js")
def pwa_sw():
    return Response("self.addEventListener('fetch', e => {});", mimetype="application/javascript")


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.route("/api/state")
def api_state():
    with state_lock:
        safe = STATE.copy()
        safe["api_key"] = ""
        safe["api_secret"] = ""
        safe["now_ts"] = time.time()
        return jsonify(safe)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    threading.Thread(target=retrain_all_models, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    run_engine_cycle(force_manual=True)
    return jsonify({"ok": True})


@app.route("/api/toggle_mode", methods=["POST"])
def api_toggle_mode():
    with state_lock:
        if STATE.get("mode") == "PAPER" and not (STATE.get("api_key") and STATE.get("api_secret")):
            return jsonify({"ok": False, "error": "Sync your Gate.io Futures API keys before switching to LIVE mode."}), 400
        STATE["mode"] = "LIVE" if STATE.get("mode") == "PAPER" else "PAPER"
        add_log(f"🔀 Switched ApexML Mode to: {STATE['mode']}")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/unlock_breaker", methods=["POST"])
def api_unlock_breaker():
    with state_lock:
        STATE["consecutive_losses"] = 0
        STATE["circuit_breaker_until"] = 0.0
        add_log("🔓 Circuit Breaker manually unlocked.")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    pair = (request.json or {}).get("pair", "")
    with state_lock:
        pos = STATE["active_positions"].get(pair)
    if not pos:
        return jsonify({"ok": False, "error": "Trade not found"}), 400
    close_position(pair, float(pos["last_price"]), "MANUAL CLOSE")
    return jsonify({"ok": True})


@app.route("/api/reset_paper", methods=["POST"])
def api_reset_paper():
    d = request.json or {}
    amt = max(5.0, float(d.get("amount_usdt") or 15.0))
    with state_lock:
        STATE["active_positions"] = {k: v for k, v in STATE["active_positions"].items() if v.get("is_live")}
        STATE["paper_balance_usdt"] = round(amt, 4)
        STATE["paper_starting_usdt"] = round(amt, 4)
        STATE["stats"]["paper_trades"] = 0
        STATE["stats"]["paper_wins"] = 0
        STATE["stats"]["paper_pnl_usdt"] = 0.0
        STATE["consecutive_losses"] = 0
        STATE["circuit_breaker_until"] = 0.0
        STATE["trade_history"] = [t for t in STATE["trade_history"] if t.get("mode") == "LIVE"]
        add_log(f"🧹 Paper Bankroll reset to ${amt:.2f} USDT.")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    d = request.json or {}
    with state_lock:
        if d.get("slot_size_usdt"):
            STATE["slot_size_usdt"] = max(2.0, float(d["slot_size_usdt"]))
        if d.get("leverage"):
            STATE["leverage"] = max(1, min(5, int(float(d["leverage"]))))
        if d.get("tp_pct"):
            STATE["tp_pct"] = max(1.0, float(d["tp_pct"]))
        if d.get("sl_pct"):
            STATE["sl_pct"] = max(0.5, float(d["sl_pct"]))
        if d.get("confidence_threshold"):
            STATE["confidence_threshold"] = max(50.0, min(90.0, float(d["confidence_threshold"])))
        if d.get("max_consecutive_losses"):
            STATE["max_consecutive_losses"] = max(1, int(float(d["max_consecutive_losses"])))
        add_log(
            f"⚙️ Saved Risk Rules: Slot=${STATE['slot_size_usdt']:.2f} | {STATE['leverage']}x | "
            f"TP=+{STATE['tp_pct']}% / SL=-{STATE['sl_pct']}% | Conf≥{STATE['confidence_threshold']}%"
        )
        save_state()
    threading.Thread(target=retrain_all_models, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/sync_keys", methods=["POST"])
def api_sync_keys():
    d = request.json or {}
    rk = (d.get("api_key") or "").strip().replace(" ", "").replace("\n", "")
    rs = (d.get("api_secret") or "").strip().replace(" ", "").replace("\n", "")
    with state_lock:
        if rk:
            STATE["api_key"] = rk
        if rs:
            STATE["api_secret"] = rs
        ak, asec = STATE.get("api_key", ""), STATE.get("api_secret", "")
    if not ak or not asec:
        return jsonify({"ok": False, "error": "Paste both Gate.io Futures API Key and Secret"}), 400
    try:
        u = fetch_futures_usdt_balance(ak, asec)
        with state_lock:
            STATE["api_connected"] = True
            STATE["live_futures_usdt"] = u
            STATE["api_status_msg"] = f"Gate.io Futures Connected (Available: ${u:.2f} USDT)"
            add_log(f"🔗 Gate.io Futures API Verified! Available: ${u:.2f} USDT")
            save_state()
        return jsonify({"ok": True, "usdt_balance": u})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:80]}), 400


# =============================================================================
# CLEAN OBSIDIAN-GLASS UI — APEXML v1.0 PRO
# =============================================================================
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#6366f1" />
  <link rel="manifest" href="/manifest.json" />
  <link rel="icon" type="image/svg+xml" href="/icon.svg" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />
  <title>ApexML v1.0 PRO | AI Futures Terminal</title>
  <style>
    :root {
      --bg: #06070c;
      --card: rgba(15, 18, 30, 0.78);
      --border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --cyan: #38bdf8;
      --green: #00f5a0;
      --red: #ff3b6b;
      --amber: #f59e0b;
      --text: #f1f5f9;
      --muted: #8b95a9;
      --mono: 'JetBrains Mono', ui-monospace, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }
    body {
      background: var(--bg);
      background-image:
        radial-gradient(ellipse 70% 45% at 20% -5%, rgba(99, 102, 241, 0.16), transparent 65%),
        radial-gradient(ellipse 60% 40% at 85% 5%, rgba(0, 245, 160, 0.09), transparent 60%);
      color: var(--text);
      padding: 20px 16px 40px;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }
    .wrap { max-width: 1200px; margin: 0 auto; }

    /* HEADER */
    .topbar {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px;
      background: var(--card); backdrop-filter: blur(20px); border: 1px solid var(--border);
      border-radius: 18px; padding: 18px 22px; margin-bottom: 18px;
    }
    .brand { display: flex; align-items: center; gap: 10px; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.4px; }
    .tag {
      background: linear-gradient(135deg, var(--accent), var(--cyan));
      color: #fff; font-size: 0.66rem; font-weight: 800; padding: 4px 10px; border-radius: 99px; letter-spacing: 0.6px;
    }
    .subtitle { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }

    .btn {
      cursor: pointer; border: none; border-radius: 10px; padding: 9px 14px;
      font-weight: 700; font-size: 0.76rem; transition: all 0.15s ease;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .btn:active { transform: scale(0.97); }
    .btn-primary { background: linear-gradient(135deg, var(--accent), #4f46e5); color: #fff; }
    .btn-green { background: linear-gradient(135deg, var(--green), var(--cyan)); color: #041c15; }
    .btn-amber { background: var(--amber); color: #1a1200; }
    .btn-ghost { background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--text); }
    .btn-danger { background: rgba(255, 59, 107, 0.15); border: 1px solid rgba(255, 59, 107, 0.4); color: #fda4af; }

    /* KPI GRID */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 18px; }
    .card {
      background: var(--card); backdrop-filter: blur(18px); border: 1px solid var(--border);
      border-radius: 16px; padding: 18px 20px;
    }
    .kpi-lbl { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); }
    .kpi-val { font-family: var(--mono); font-size: 1.6rem; font-weight: 700; margin: 8px 0 4px; }
    .kpi-sub { font-size: 0.76rem; color: var(--muted); }

    /* SIGNAL MATRIX */
    .sec-head {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
      font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.7px; color: #cbd5e1;
    }
    .coin-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-bottom: 18px; }
    .coin-card {
      background: rgba(12, 15, 26, 0.88); border: 1px solid var(--border); border-radius: 15px; padding: 16px;
    }
    .coin-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .coin-name { font-size: 1.05rem; font-weight: 800; }
    .pill { font-size: 0.68rem; font-weight: 800; padding: 4px 10px; border-radius: 99px; letter-spacing: 0.4px; }
    .pill-long { background: rgba(0, 245, 160, 0.14); color: var(--green); border: 1px solid rgba(0, 245, 160, 0.4); }
    .pill-short { background: rgba(255, 59, 107, 0.14); color: var(--red); border: 1px solid rgba(255, 59, 107, 0.4); }
    .pill-hold { background: rgba(255, 255, 255, 0.06); color: var(--muted); border: 1px solid var(--border); }

    .prob-bar { display: flex; height: 8px; border-radius: 99px; overflow: hidden; background: rgba(255,255,255,0.06); margin: 8px 0; }
    .prob-l { background: var(--green); }
    .prob-h { background: #334155; }
    .prob-s { background: var(--red); }
    .prob-nums { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.72rem; color: var(--muted); margin-bottom: 12px; }

    .bt-box {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
      background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);
      border-radius: 10px; padding: 10px; text-align: center;
    }
    .bt-k { font-size: 0.64rem; color: var(--muted); text-transform: uppercase; font-weight: 700; }
    .bt-v { font-family: var(--mono); font-size: 0.86rem; font-weight: 700; margin-top: 3px; }

    /* MAIN TWO-COLUMN LAYOUT */
    .split { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; margin-bottom: 18px; }
    @media (max-width: 900px) { .split { grid-template-columns: 1fr; } }

    .trade-row {
      background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px; margin-bottom: 10px;
    }
    .trade-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-family: var(--mono); font-size: 0.75rem; margin: 10px 0; color: var(--muted); }
    .trade-meta strong { color: var(--text); }

    .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
    .form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
    label { display: block; font-size: 0.68rem; color: var(--muted); font-weight: 700; margin-bottom: 5px; }
    input {
      width: 100%; background: rgba(5, 7, 13, 0.85); border: 1px solid var(--border);
      color: var(--text); padding: 9px 11px; border-radius: 9px; font-family: var(--mono); font-size: 0.82rem;
    }
    input:focus { outline: none; border-color: var(--accent); }

    table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
    th, td { padding: 9px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: var(--mono); }
    th { font-family: 'Inter', sans-serif; color: var(--muted); font-size: 0.66rem; text-transform: uppercase; font-weight: 700; }

    .logs {
      background: rgba(4, 6, 11, 0.9); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 14px; height: 180px; overflow-y: auto; font-family: var(--mono);
      font-size: 0.72rem; color: #cbd5e1; line-height: 1.65;
    }
    .c-green { color: var(--green); }
    .c-red { color: var(--red); }
    .c-cyan { color: var(--cyan); }
    .c-amber { color: var(--amber); }
  </style>
</head>
<body>
<div class="wrap">

  <!-- TOP BAR -->
  <div class="topbar">
    <div>
      <div class="brand">🧠 APEXML <span class="tag">v1.0 PRO • 1H FUTURES</span></div>
      <div class="subtitle" id="statusLine">Gradient Boosting • Walk-Forward Validated • 2:1 Risk Engine</div>
    </div>
    <div class="actions">
      <button class="btn btn-green" onclick="installApp()">📲 Install App</button>
      <button class="btn btn-amber" onclick="toggleMode()" id="modeBtn">Mode: PAPER</button>
      <button class="btn btn-primary" onclick="retrainModels()" id="trainBtn">🧠 Retrain Brain</button>
      <button class="btn btn-ghost" onclick="sweepNow()">⚡ Sweep</button>
    </div>
  </div>

  <!-- 4 CLEAN KPI CARDS -->
  <div class="kpi-grid">
    <div class="card">
      <div class="kpi-lbl">Available Balance</div>
      <div class="kpi-val c-cyan" id="balVal">$15.00</div>
      <div class="kpi-sub" id="balSub">Paper Mode • 2x Isolated</div>
    </div>
    <div class="card">
      <div class="kpi-lbl">Verified Net PnL</div>
      <div class="kpi-val c-green" id="pnlVal">+$0.0000</div>
      <div class="kpi-sub">Trades: <strong id="tradeCount" style="color:#fff;">0</strong> | Win Rate: <strong id="liveWr" style="color:#fff;">0%</strong></div>
    </div>
    <div class="card">
      <div class="kpi-lbl">Unseen Backtest Avg</div>
      <div class="kpi-val" id="btAvgVal">--%</div>
      <div class="kpi-sub">Last Trained: <span id="trainTime">--</span></div>
    </div>
    <div class="card">
      <div class="kpi-lbl">Circuit Breaker</div>
      <div class="kpi-val c-green" id="cbVal">ARMED</div>
      <div class="kpi-sub" id="cbSub">0 / 2 consecutive losses</div>
    </div>
  </div>

  <!-- AI SIGNAL & WALK-FORWARD MATRIX -->
  <div class="sec-head">
    <span>📡 Live ML Signal Matrix (1h Candles + BTC Macro Context)</span>
    <span style="color:var(--muted);font-weight:600;">Confidence Gate: ≥<span id="gateLbl">65</span>%</span>
  </div>
  <div class="coin-grid" id="coinGrid"></div>

  <!-- ACTIVE TRADES & SETTINGS -->
  <div class="split">
    <div class="card">
      <div class="sec-head">
        <span>⚡ Active Trades</span>
        <span id="openCount" style="color:var(--cyan);">0 Open</span>
      </div>
      <div id="activeTradesBox"></div>
    </div>

    <div class="card">
      <div class="sec-head"><span>⚙️ Risk & Execution Settings</span></div>
      <div class="form-grid">
        <div><label>Slot ($)</label><input type="number" id="cfgSlot" step="1" value="10" /></div>
        <div><label>Leverage (x)</label><input type="number" id="cfgLev" min="1" max="5" value="2" /></div>
        <div><label>Confidence (%)</label><input type="number" id="cfgConf" step="1" value="65" /></div>
      </div>
      <div class="form-grid">
        <div><label>Take Profit (%)</label><input type="number" id="cfgTp" step="0.5" value="4.0" /></div>
        <div><label>Stop Loss (%)</label><input type="number" id="cfgSl" step="0.5" value="2.0" /></div>
        <div><label>Max Loss Streak</label><input type="number" id="cfgLoss" min="1" max="5" value="2" /></div>
      </div>
      <button class="btn btn-primary" style="width:100%;justify-content:center;margin-bottom:14px;" onclick="saveConfig()">💾 Save Rules & Retrain</button>

      <div class="sec-head" style="margin-top:6px;"><span>💰 Paper Reset & Gate.io Keys</span></div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-bottom:12px;">
        <input type="number" id="paperResetAmt" value="15" step="5" placeholder="Paper USDT" />
        <button class="btn btn-ghost" onclick="resetPaper()">🧹 Reset Paper</button>
      </div>
      <div class="form-grid-2">
        <input type="password" id="apiKey" placeholder="Gate.io Futures API Key" autocomplete="off" />
        <input type="password" id="apiSec" placeholder="Gate.io Futures Secret" autocomplete="off" />
      </div>
      <button class="btn btn-ghost" style="width:100%;justify-content:center;" onclick="syncKeys()" id="syncBtn">🔗 Connect Gate.io Futures Keys</button>
    </div>
  </div>

  <!-- TRADE HISTORY & LOGS -->
  <div class="split">
    <div class="card">
      <div class="sec-head"><span>📜 Closed Trade History</span></div>
      <div style="overflow-x:auto;max-height:200px;">
        <table>
          <thead><tr><th>Time</th><th>Pair</th><th>Side</th><th>Entry → Exit</th><th>Reason</th><th>Net PnL</th></tr></thead>
          <tbody id="histBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="sec-head"><span>🖥️ Engine Telemetry</span><span id="lastUpd" style="font-family:var(--mono);color:var(--muted);">--:--:--</span></div>
      <div class="logs" id="logBox"></div>
    </div>
  </div>

</div>

<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
  }
  let deferredPrompt = null;
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); deferredPrompt = e; });
  async function installApp() {
    if (deferredPrompt) { deferredPrompt.prompt(); deferredPrompt = null; }
    else alert("Tap your browser menu (⋮) and choose 'Add to Home screen' or 'Install app'.");
  }

  function fmtPx(v) {
    v = Number(v || 0);
    if (v === 0) return '$0.0000';
    if (v < 0.001) return '$' + v.toFixed(6);
    if (v < 1) return '$' + v.toFixed(4);
    return '$' + v.toFixed(3);
  }

  async function retrainModels() {
    const btn = document.getElementById('trainBtn');
    btn.innerText = '⏳ Training...';
    await fetch('/api/retrain', { method: 'POST' });
    setTimeout(() => { btn.innerText = '🧠 Retrain Brain'; refresh(); }, 4000);
  }
  async function sweepNow() { await fetch('/api/sweep', { method: 'POST' }); await refresh(); }
  async function toggleMode() {
    const r = await fetch('/api/toggle_mode', { method: 'POST' });
    const d = await r.json();
    if (!d.ok) alert(d.error);
    await refresh();
  }
  async function unlockBreaker() { await fetch('/api/unlock_breaker', { method: 'POST' }); await refresh(); }
  async function closeTrade(pair) {
    await fetch('/api/close_trade', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({pair}) });
    await refresh();
  }
  async function resetPaper() {
    const amt = document.getElementById('paperResetAmt').value || 15;
    await fetch('/api/reset_paper', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({amount_usdt: amt}) });
    await refresh();
  }
  async function saveConfig() {
    await fetch('/api/update_config', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        slot_size_usdt: document.getElementById('cfgSlot').value,
        leverage: document.getElementById('cfgLev').value,
        confidence_threshold: document.getElementById('cfgConf').value,
        tp_pct: document.getElementById('cfgTp').value,
        sl_pct: document.getElementById('cfgSl').value,
        max_consecutive_losses: document.getElementById('cfgLoss').value
      })
    });
    alert('Settings saved! Retraining models with your new TP/SL targets...');
    await refresh();
  }
  async function syncKeys() {
    const btn = document.getElementById('syncBtn');
    btn.innerText = 'Connecting...';
    const r = await fetch('/api/sync_keys', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ api_key: document.getElementById('apiKey').value, api_secret: document.getElementById('apiSec').value })
    });
    const d = await r.json();
    if (d.ok) {
      document.getElementById('apiKey').value = '';
      document.getElementById('apiSec').value = '';
      btn.innerText = '✅ Connected';
    } else {
      alert(d.error || 'Check API keys');
      btn.innerText = '❌ Retry';
    }
    await refresh();
    setTimeout(() => { btn.innerText = '🔗 Connect Gate.io Futures Keys'; }, 2500);
  }

  let loadedCfg = false;
  async function refresh() {
    try {
      const r = await fetch('/api/state');
      const s = await r.json();

      if (!loadedCfg) {
        document.getElementById('cfgSlot').value = s.slot_size_usdt;
        document.getElementById('cfgLev').value = s.leverage;
        document.getElementById('cfgConf').value = s.confidence_threshold;
        document.getElementById('cfgTp').value = s.tp_pct;
        document.getElementById('cfgSl').value = s.sl_pct;
        document.getElementById('cfgLoss').value = s.max_consecutive_losses;
        document.getElementById('paperResetAmt').value = s.paper_starting_usdt;
        loadedCfg = true;
      }

      document.getElementById('statusLine').innerText = s.api_status_msg;
      document.getElementById('lastUpd').innerText = s.last_update;
      document.getElementById('trainTime').innerText = s.last_trained_str;
      document.getElementById('gateLbl').innerText = s.confidence_threshold;

      const isLive = s.mode === 'LIVE';
      const modeBtn = document.getElementById('modeBtn');
      modeBtn.innerText = isLive ? 'Mode: LIVE FUTURES' : 'Mode: PAPER';
      modeBtn.className = isLive ? 'btn btn-green' : 'btn btn-amber';

      const bal = isLive ? Number(s.live_futures_usdt || 0) : Number(s.paper_balance_usdt || 0);
      document.getElementById('balVal').innerText = '$' + bal.toFixed(2);
      document.getElementById('balSub').innerText = `${s.mode} Mode • ${s.leverage}x Isolated • TP +${s.tp_pct}% / SL -${s.sl_pct}%`;

      const trades = isLive ? s.stats.live_trades : s.stats.paper_trades;
      const wins = isLive ? s.stats.live_wins : s.stats.paper_wins;
      const pnl = isLive ? Number(s.stats.live_pnl_usdt || 0) : Number(s.stats.paper_pnl_usdt || 0);
      const wr = trades > 0 ? ((wins / trades) * 100).toFixed(1) : '0.0';

      const pnlEl = document.getElementById('pnlVal');
      pnlEl.innerText = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(4);
      pnlEl.className = 'kpi-val ' + (pnl >= 0 ? 'c-green' : 'c-red');
      document.getElementById('tradeCount').innerText = trades;
      document.getElementById('liveWr').innerText = wr + '%';

      // Circuit Breaker
      const locked = Number(s.now_ts || 0) < Number(s.circuit_breaker_until || 0);
      const cbEl = document.getElementById('cbVal');
      if (locked) {
        cbEl.innerText = 'PAUSED';
        cbEl.className = 'kpi-val c-red';
        document.getElementById('cbSub').innerHTML = `<button class="btn btn-danger" style="padding:3px 9px;font-size:0.68rem;" onclick="unlockBreaker()">🔓 Unlock Breaker</button>`;
      } else {
        cbEl.innerText = 'ACTIVE';
        cbEl.className = 'kpi-val c-green';
        document.getElementById('cbSub').innerText = `${s.consecutive_losses} / ${s.max_consecutive_losses} consecutive losses`;
      }

      // Coin Signal Matrix
      const reps = Object.values(s.model_reports || {});
      if (reps.length > 0) {
        const avgNet = reps.reduce((a, b) => a + Number(b.bt_net_pct || 0), 0) / reps.length;
        const btEl = document.getElementById('btAvgVal');
        btEl.innerText = (avgNet >= 0 ? '+' : '') + avgNet.toFixed(1) + '%';
        btEl.className = 'kpi-val ' + (avgNet >= 0 ? 'c-green' : 'c-red');

        document.getElementById('coinGrid').innerHTML = reps.map(c => {
          const pillCls = c.signal === 'LONG' ? 'pill-long' : (c.signal === 'SHORT' ? 'pill-short' : 'pill-hold');
          const netCls = c.bt_net_pct >= 0 ? 'c-green' : 'c-red';
          return `
            <div class="coin-card">
              <div class="coin-top">
                <div>
                  <div class="coin-name">${c.pair}</div>
                  <div style="font-family:var(--mono);font-size:0.8rem;color:var(--muted);">${fmtPx(c.live_price)}</div>
                </div>
                <span class="pill ${pillCls}">${c.signal}</span>
              </div>
              <div class="prob-bar">
                <div class="prob-l" style="width:${c.prob_long}%"></div>
                <div class="prob-h" style="width:${c.prob_hold}%"></div>
                <div class="prob-s" style="width:${c.prob_short}%"></div>
              </div>
              <div class="prob-nums">
                <span class="c-green">LONG ${c.prob_long}%</span>
                <span>HOLD ${c.prob_hold}%</span>
                <span class="c-red">SHORT ${c.prob_short}%</span>
              </div>
              <div class="bt-box">
                <div><div class="bt-k">Unseen Win %</div><div class="bt-v">${c.bt_win_rate}%</div></div>
                <div><div class="bt-k">Test Trades</div><div class="bt-v">${c.bt_trades}</div></div>
                <div><div class="bt-k">Unseen PnL</div><div class="bt-v ${netCls}">${c.bt_net_pct >= 0 ? '+' : ''}${c.bt_net_pct}%</div></div>
              </div>
            </div>
          `;
        }).join('');
      } else {
        document.getElementById('coinGrid').innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;color:var(--muted);">Training ML models on 900 hours of live Gate.io Futures candles... (takes ~8 seconds)</div>`;
      }

      // Active Trades
      const posList = Object.values(s.active_positions || {});
      document.getElementById('openCount').innerText = posList.length + ' Open';
      if (posList.length === 0) {
        document.getElementById('activeTradesBox').innerHTML = `<div style="text-align:center;color:var(--muted);padding:36px 12px;font-size:0.84rem;">100% in USDT Cash — Waiting for a ≥${s.confidence_threshold}% ML signal on SUI, SOL, or ARB.</div>`;
      } else {
        document.getElementById('activeTradesBox').innerHTML = posList.map(p => {
          const pnlCls = p.pnl_usdt >= 0 ? 'c-green' : 'c-red';
          const pillCls = p.side === 'LONG' ? 'pill-long' : 'pill-short';
          return `
            <div class="trade-row">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <strong>${p.pair}</strong>
                  <span class="pill ${pillCls}">${p.side} ${p.leverage}x</span>
                  <span style="font-size:0.72rem;color:var(--muted);">Conf: ${p.confidence}%</span>
                </div>
                <strong class="${pnlCls}" style="font-family:var(--mono);">${p.pnl_usdt >= 0 ? '+' : ''}$${Number(p.pnl_usdt).toFixed(4)} (${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct}%)</strong>
              </div>
              <div class="trade-meta">
                <div>Entry: <strong>${fmtPx(p.entry_price)}</strong></div>
                <div>Live: <strong>${fmtPx(p.last_price)}</strong></div>
                <div>Margin: <strong>$${Number(p.margin_usdt).toFixed(2)}</strong></div>
                <div>TP: <strong class="c-green">${fmtPx(p.tp_price)}</strong></div>
                <div>SL: <strong class="c-red">${fmtPx(p.sl_price)}</strong></div>
                <div style="text-align:right;"><button class="btn btn-danger" style="padding:4px 10px;font-size:0.7rem;" onclick="closeTrade('${p.pair}')">✖ Close</button></div>
              </div>
            </div>
          `;
        }).join('');
      }

      // Trade History
      const hb = document.getElementById('histBody');
      if (s.trade_history && s.trade_history.length > 0) {
        hb.innerHTML = s.trade_history.map(t => `
          <tr>
            <td>${t.time}</td>
            <td><strong>${t.pair}</strong></td>
            <td class="${t.side === 'LONG' ? 'c-green' : 'c-red'}">${t.side}</td>
            <td>${t.entry} → ${t.exit}</td>
            <td>${t.reason}</td>
            <td class="${t.pnl_usdt >= 0 ? 'c-green' : 'c-red'}"><strong>${t.pnl_usdt >= 0 ? '+' : ''}$${Number(t.pnl_usdt).toFixed(4)}</strong></td>
          </tr>
        `).join('');
      } else {
        hb.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:20px;">No closed trades yet.</td></tr>`;
      }

      document.getElementById('logBox').innerHTML = (s.logs || []).join('<br/>');
    } catch (e) {}
  }

  setInterval(refresh, 4000);
  refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_UI)


if __name__ == "__main__":
    ensure_worker_alive()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
