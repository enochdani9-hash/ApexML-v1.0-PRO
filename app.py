import os
import time
import math
import json
import hmac 
import hashlib 
import threading 
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

STATE_FILE = "indexvault_100_state.json"
GATE_HOST = "https://api.gateio.ws"
GATE_PREFIX = "/api/v4"

STABLECOINS = {
    "USDT", "USDC", "FDUSD", "DAI", "TUSD", "USDD", "BUSD", "PYUSD",
    "USDE", "EUR", "EURT", "USTC", "USDP", "GUSD", "FRAX", "USD1", "RLUSD"
}
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

HTTP_SESSION = requests.Session()
adapter = HTTPAdapter(
    pool_connections=24,
    pool_maxsize=24,
    max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[502, 503, 504])
)
HTTP_SESSION.mount("https://", adapter)

PAIR_META_CACHE = {}
PAIR_META_TS = 0.0

DEFAULT_STATE = {
    "mode": "PAPER",
    "auto_rebalance": True,         # Toggle: True = Auto-Rebalance Compounder, False = Pure Buy & Hold
    "target_coin_count": 100,       # 100 coins in Paper Mode; auto-sized in Live Mode
    "rebalance_trigger_pct": 25.0,  # Trim winner when it pumps +25% above its target weight
    "live_min_order_usdt": 5.0,     # Safety floor per coin in Live Mode (respects Gate.io min_quote_amount)
    "spot_fee_pct": 0.20,           # Gate.io Spot single-side fee
    "api_key": os.environ.get("GATE_API_KEY", "").strip(),
    "api_secret": os.environ.get("GATE_API_SECRET", "").strip(),
    "api_connected": False,
    "api_status_msg": "Paper Mode Active — Deploy $50 or $100 Across Top 100 Coins",
    "live_spot_usdt": 0.0,
    "paper_cash_usdt": 100.0,
    "total_deposited_usdt": 100.0,
    "compound_pool_usdt": 0.0,      # Swept rebalance profits waiting to buy dips/new coins
    "holdings": {},                 # { "BTC_USDT": { qty, avg_price, cost_usdt, last_price, value_usdt, pnl_pct, pnl_usdt, change_24h, rank, rebalances } }
    "equity_history": [],           # [{ ts, date, time, equity, deposited, pnl_usdt, pnl_pct }]
    "daily_open_equity": 100.0,
    "daily_open_date": "",
    "rebalance_ledger": [],
    "stats": {
        "total_rebalances": 0,
        "compounded_profit_usdt": 0.0
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
    if v < 0.0001:
        return f"${v:.8f}"
    if v < 0.01:
        return f"${v:.6f}"
    if v < 1.0:
        return f"${v:.4f}"
    return f"${v:.2f}"


def floor_prec(val, prec):
    factor = 10 ** int(max(0, prec))
    return math.floor(float(val) * factor) / factor


def add_log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    STATE["logs"].insert(0, f"[{ts}] {msg}")
    STATE["logs"] = STATE["logs"][:120]


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
# GATE.IO SPOT APIv4 HELPERS + AUTO-JUNK FILTER
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


def refresh_pair_meta():
    global PAIR_META_CACHE, PAIR_META_TS
    now = time.time()
    if PAIR_META_CACHE and (now - PAIR_META_TS < 3600):
        return PAIR_META_CACHE

    r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/spot/currency_pairs", timeout=10)
    r.raise_for_status()
    meta = {}
    for item in r.json():
        pid = item.get("id", "")
        if not pid.endswith("_USDT"):
            continue
        if item.get("trade_status") != "tradable":
            continue
        if item.get("st_tag"):  # Auto-Junk Filter: blocks delisting/risk-flagged tokens
            continue
        base = item.get("base", "").upper()
        if base in STABLECOINS or base.endswith(LEVERAGED_SUFFIXES):
            continue
        meta[pid] = {
            "base": base,
            "amount_precision": int(item.get("amount_precision") or 2),
            "min_quote_amount": max(1.0, float(item.get("min_quote_amount") or 3.0))
        }
    PAIR_META_CACHE = meta
    PAIR_META_TS = now
    return meta


def fetch_top_index_candidates(limit=120):
    meta = refresh_pair_meta()
    r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/spot/tickers", timeout=8)
    r.raise_for_status()
    ranked = []
    for t in r.json():
        cp = t.get("currency_pair")
        if cp not in meta:
            continue
        last = float(t.get("last") or 0.0)
        qvol = float(t.get("quote_volume") or 0.0)
        if last <= 0 or qvol < 150000:  # Minimum $150k 24h liquidity filter
            continue
        ranked.append({
            "pair": cp,
            "base": meta[cp]["base"],
            "last": last,
            "change_24h": float(t.get("change_percentage") or 0.0),
            "quote_volume": qvol,
            "min_quote_amount": meta[cp]["min_quote_amount"],
            "amount_precision": meta[cp]["amount_precision"]
        })
    ranked.sort(key=lambda x: x["quote_volume"], reverse=True)
    for idx, item in enumerate(ranked[:limit], start=1):
        item["rank"] = idx
    return ranked[:limit]


def fetch_all_spot_tickers():
    r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/spot/tickers", timeout=8)
    r.raise_for_status()
    out = {}
    for t in r.json():
        cp = t.get("currency_pair")
        last = float(t.get("last") or 0.0)
        if cp and last > 0:
            out[cp] = {
                "last": last,
                "change_24h": float(t.get("change_percentage") or 0.0)
            }
    return out


def fetch_spot_accounts(custom_key=None, custom_secret=None):
    url = f"{GATE_PREFIX}/spot/accounts"
    headers = gate_sign("GET", url, "", "", custom_key, custom_secret)
    r = HTTP_SESSION.get(f"{GATE_HOST}{url}", headers=headers, timeout=6)
    if r.status_code != 200:
        raise ValueError(f"Gate.io Spot HTTP {r.status_code}: {r.text[:80]}")
    bals = {}
    for item in r.json():
        cur = item.get("currency", "").upper()
        avail = float(item.get("available") or 0.0)
        if avail > 0:
            bals[cur] = avail
    return bals


def place_spot_market_buy(pair, usdt_amount):
    url = f"{GATE_PREFIX}/spot/orders"
    body = {
        "currency_pair": pair,
        "account": "spot",
        "type": "market",
        "side": "buy",
        "amount": f"{round(float(usdt_amount), 2):.2f}",
        "time_in_force": "ioc"
    }
    body_str = json.dumps(body)
    headers = gate_sign("POST", url, "", body_str)
    r = HTTP_SESSION.post(f"{GATE_HOST}{url}", headers=headers, data=body_str, timeout=7)
    return r.status_code, r.json()


def place_spot_market_sell(pair, coin_qty, prec=2):
    clean_qty = floor_prec(coin_qty, prec)
    if clean_qty <= 0:
        return 400, {"message": "Quantity too small"}
    url = f"{GATE_PREFIX}/spot/orders"
    body = {
        "currency_pair": pair,
        "account": "spot",
        "type": "market",
        "side": "sell",
        "amount": f"{clean_qty:.{prec}f}",
        "time_in_force": "ioc"
    }
    body_str = json.dumps(body)
    headers = gate_sign("POST", url, "", body_str)
    r = HTTP_SESSION.post(f"{GATE_HOST}{url}", headers=headers, data=body_str, timeout=7)
    return r.status_code, r.json()


# =============================================================================
# INDEX DEPLOYMENT & AUTO-REBALANCING COMPOUNDER
# =============================================================================
def deploy_index_basket(deposit_usdt, target_count=100):
    candidates = fetch_top_index_candidates(limit=max(110, target_count + 10))
    if not candidates:
        raise ValueError("Could not fetch Gate.io Top 100 market list.")

    with state_lock:
        mode = STATE.get("mode", "PAPER")
        fee_pct = float(STATE.get("spot_fee_pct", 0.20))
        live_floor = float(STATE.get("live_min_order_usdt", 5.0))
        has_keys = bool(STATE.get("api_key") and STATE.get("api_secret"))

    bought_count = 0
    total_spent = 0.0

    if mode == "PAPER":
        selected = candidates[:target_count]
        per_coin = round(deposit_usdt / max(1, len(selected)), 4)
        with state_lock:
            for c in selected:
                pair = c["pair"]
                px = c["last"]
                net_usdt = per_coin * (1.0 - fee_pct / 100.0)
                qty = net_usdt / px
                existing = STATE["holdings"].get(pair)
                if existing:
                    new_qty = float(existing["qty"]) + qty
                    new_cost = float(existing["cost_usdt"]) + per_coin
                    existing["qty"] = new_qty
                    existing["cost_usdt"] = round(new_cost, 4)
                    existing["avg_price"] = new_cost / max(1e-12, new_qty)
                    existing["last_price"] = px
                    existing["value_usdt"] = round(new_qty * px, 4)
                    existing["change_24h"] = round(c["change_24h"], 2)
                    existing["rank"] = c["rank"]
                else:
                    STATE["holdings"][pair] = {
                        "pair": pair,
                        "base": c["base"],
                        "rank": c["rank"],
                        "qty": qty,
                        "avg_price": px,
                        "cost_usdt": round(per_coin, 4),
                        "last_price": px,
                        "value_usdt": round(net_usdt, 4),
                        "pnl_usdt": round(net_usdt - per_coin, 4),
                        "pnl_pct": round(-fee_pct, 2),
                        "change_24h": round(c["change_24h"], 2),
                        "min_quote_amount": c["min_quote_amount"],
                        "amount_precision": c["amount_precision"],
                        "rebalances": 0
                    }
                bought_count += 1
                total_spent += per_coin

            STATE["paper_cash_usdt"] = round(max(0.0, STATE.get("paper_cash_usdt", 0.0) - total_spent), 4)
            add_log(f"🚀 [PAPER INDEX] Deployed ${total_spent:.2f} across {bought_count} Top Gate.io tokens (${per_coin:.2f}/coin).")
            save_state()
    else:
        if not has_keys:
            raise ValueError("Connect your Gate.io Spot API keys first for Live Mode.")
        avail_usdt = float(fetch_spot_accounts().get("USDT", 0.0))
        budget = min(deposit_usdt, avail_usdt)
        if budget < live_floor:
            raise ValueError(f"Insufficient Gate.io Spot USDT (Available: ${avail_usdt:.2f})")

        # Smart Auto-Sizing: buy as many Top-100 coins as budget allows while obeying Gate.io min_quote_amount
        remaining = budget
        for c in candidates[:target_count]:
            req_amt = max(live_floor, c["min_quote_amount"] * 1.05)
            if remaining < req_amt:
                break
            code, resp = place_spot_market_buy(c["pair"], req_amt)
            if code in (200, 201):
                filled_qty = float(resp.get("filled_amount") or (req_amt / c["last"]))
                avg_px = float(resp.get("avg_deal_price") or c["last"])
                if avg_px <= 0:
                    avg_px = c["last"]
                with state_lock:
                    existing = STATE["holdings"].get(c["pair"])
                    if existing:
                        new_qty = float(existing["qty"]) + filled_qty
                        new_cost = float(existing["cost_usdt"]) + req_amt
                        existing["qty"] = new_qty
                        existing["cost_usdt"] = round(new_cost, 4)
                        existing["avg_price"] = new_cost / max(1e-12, new_qty)
                    else:
                        STATE["holdings"][c["pair"]] = {
                            "pair": c["pair"],
                            "base": c["base"],
                            "rank": c["rank"],
                            "qty": filled_qty,
                            "avg_price": avg_px,
                            "cost_usdt": round(req_amt, 4),
                            "last_price": avg_px,
                            "value_usdt": round(filled_qty * avg_px, 4),
                            "pnl_usdt": 0.0,
                            "pnl_pct": 0.0,
                            "change_24h": round(c["change_24h"], 2),
                            "min_quote_amount": c["min_quote_amount"],
                            "amount_precision": c["amount_precision"],
                            "rebalances": 0
                        }
                remaining -= req_amt
                total_spent += req_amt
                bought_count += 1
                time.sleep(0.12)

        with state_lock:
            STATE["live_spot_usdt"] = round(max(0.0, avail_usdt - total_spent), 4)
            add_log(f"🚀 [LIVE INDEX] Smart Auto-Sized ${total_spent:.2f} into {bought_count} Top Gate.io coins!")
            save_state()

    return bought_count, round(total_spent, 2)


def run_index_cycle(force_manual=False):
    if not cycle_lock.acquire(blocking=force_manual):
        return

    try:
        with state_lock:
            mode = STATE.get("mode", "PAPER")
            has_keys = bool(STATE.get("api_key") and STATE.get("api_secret"))
            auto_reb = bool(STATE.get("auto_rebalance", True))
            reb_trigger = float(STATE.get("rebalance_trigger_pct", 25.0))
            fee_pct = float(STATE.get("spot_fee_pct", 0.20))
            live_floor = float(STATE.get("live_min_order_usdt", 5.0))

        if has_keys:
            try:
                spot_usdt = float(fetch_spot_accounts().get("USDT", 0.0))
                with state_lock:
                    STATE["api_connected"] = True
                    STATE["live_spot_usdt"] = spot_usdt
                    STATE["api_status_msg"] = f"Gate.io Spot Connected (Available: ${spot_usdt:.2f} USDT)"
            except Exception as e:
                with state_lock:
                    STATE["api_connected"] = False
                    STATE["api_status_msg"] = f"API Notice: {str(e)[:55]}"

        try:
            tk_map = fetch_all_spot_tickers()
        except Exception:
            return

        total_holdings_val = 0.0
        with state_lock:
            holdings = STATE.get("holdings", {})
            for pair, h in holdings.items():
                tk = tk_map.get(pair)
                if tk:
                    px = tk["last"]
                    h["last_price"] = px
                    h["change_24h"] = round(tk["change_24h"], 2)
                val = float(h["qty"]) * float(h["last_price"])
                cost = max(0.0001, float(h["cost_usdt"]))
                pnl_u = val - cost
                pnl_p = (pnl_u / cost) * 100.0
                h["value_usdt"] = round(val, 4)
                h["pnl_usdt"] = round(pnl_u, 4)
                h["pnl_pct"] = round(pnl_p, 2)
                total_holdings_val += val

            # AUTO-REBALANCING COMPOUNDER
            if auto_reb and len(holdings) >= 2 and total_holdings_val > 0:
                target_weight_val = total_holdings_val / len(holdings)
                # Find any winner that surged >= reb_trigger % above target weight
                winners = [
                    h for h in holdings.values()
                    if h["value_usdt"] >= target_weight_val * (1.0 + reb_trigger / 100.0)
                ]
                laggards = sorted(holdings.values(), key=lambda x: x["pnl_pct"])

                for w in winners[:3]:
                    trim_usdt = round(w["value_usdt"] - target_weight_val, 4)
                    if trim_usdt < 0.05:
                        continue
                    # In LIVE mode, respect Gate.io minimum order size
                    if mode == "LIVE" and trim_usdt < max(live_floor, float(w.get("min_quote_amount", 3.0))):
                        continue

                    trim_qty = trim_usdt / max(1e-12, float(w["last_price"]))
                    if mode == "LIVE" and has_keys:
                        code, _ = place_spot_market_sell(w["pair"], trim_qty, int(w.get("amount_precision", 2)))
                        if code not in (200, 201):
                            continue

                    net_trimmed = round(trim_usdt * (1.0 - fee_pct / 100.0), 4)
                    w["qty"] = max(0.0, float(w["qty"]) - trim_qty)
                    w["value_usdt"] = round(w["qty"] * float(w["last_price"]), 4)
                    w["rebalances"] = int(w.get("rebalances", 0)) + 1

                    # Reinvest into the deepest laggard
                    target_lag = laggards[0] if laggards and laggards[0]["pair"] != w["pair"] else None
                    reinvest_note = "Swept to Compound Pool"
                    if target_lag and (mode == "PAPER" or net_trimmed >= max(live_floor, float(target_lag.get("min_quote_amount", 3.0)))):
                        ok_buy = True
                        if mode == "LIVE" and has_keys:
                            c_b, _ = place_spot_market_buy(target_lag["pair"], net_trimmed)
                            ok_buy = c_b in (200, 201)
                        if ok_buy:
                            buy_net = net_trimmed * (1.0 - fee_pct / 100.0)
                            add_qty = buy_net / max(1e-12, float(target_lag["last_price"]))
                            target_lag["qty"] = float(target_lag["qty"]) + add_qty
                            target_lag["cost_usdt"] = round(float(target_lag["cost_usdt"]) + net_trimmed, 4)
                            target_lag["value_usdt"] = round(target_lag["qty"] * float(target_lag["last_price"]), 4)
                            reinvest_note = f"Bought Dip -> {target_lag['base']}"
                        else:
                            STATE["compound_pool_usdt"] = round(STATE.get("compound_pool_usdt", 0.0) + net_trimmed, 4)
                    else:
                        STATE["compound_pool_usdt"] = round(STATE.get("compound_pool_usdt", 0.0) + net_trimmed, 4)

                    STATE["stats"]["total_rebalances"] += 1
                    STATE["stats"]["compounded_profit_usdt"] = round(
                        STATE["stats"]["compounded_profit_usdt"] + net_trimmed, 4
                    )
                    STATE["rebalance_ledger"].insert(0, {
                        "time": datetime.utcnow().strftime("%H:%M:%S"),
                        "winner": w["pair"],
                        "trimmed": f"${net_trimmed:.2f}",
                        "action": reinvest_note
                    })
                    STATE["rebalance_ledger"] = STATE["rebalance_ledger"][:40]
                    add_log(f"♻️ [COMPOUND] Trimmed +${net_trimmed:.2f} from {w['pair']} (+{w['pnl_pct']}%) → {reinvest_note}")

            # DAILY SNAPSHOT & EQUITY CURVE RECORDING
            cash_val = float(STATE.get("live_spot_usdt", 0.0)) if mode == "LIVE" else float(STATE.get("paper_cash_usdt", 0.0))
            pool_val = float(STATE.get("compound_pool_usdt", 0.0))
            total_equity = round(total_holdings_val + cash_val + pool_val, 4)
            deposited = max(0.01, float(STATE.get("total_deposited_usdt", 100.0)))
            total_pnl = round(total_equity - deposited, 4)
            total_pnl_pct = round((total_pnl / deposited) * 100.0, 2)

            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            if STATE.get("daily_open_date") != today_str:
                STATE["daily_open_date"] = today_str
                STATE["daily_open_equity"] = total_equity

            now_ts = time.time()
            hist = STATE.get("equity_history", [])
            if not hist or (now_ts - float(hist[-1].get("ts", 0.0)) >= 900) or hist[-1].get("date") != today_str:
                hist.append({
                    "ts": now_ts,
                    "date": today_str,
                    "time": datetime.utcnow().strftime("%H:%M"),
                    "equity": total_equity,
                    "deposited": deposited,
                    "pnl_usdt": total_pnl,
                    "pnl_pct": total_pnl_pct
                })
                STATE["equity_history"] = hist[-96:]  # Keeps rolling equity curve points

            STATE["cycle_count"] = STATE.get("cycle_count", 0) + 1
            STATE["last_update"] = datetime.utcnow().strftime("%H:%M:%S")
            STATE["last_worker_heartbeat"] = now_ts
            save_state()
    finally:
        cycle_lock.release()


def background_worker():
    load_state()
    add_log("🏛️ IndexVault 100 PRO Online — Crypto S&P 500 Auto-Compounding Engine Ready.")
    while True:
        run_index_cycle(force_manual=False)
        time.sleep(15.0)


def ensure_worker_alive():
    global worker_started
    now = time.time()
    if not worker_started or (now - STATE.get("last_worker_heartbeat", 0) > 45):
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
        "name": "IndexVault 100 PRO",
        "short_name": "IndexVault",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#04060a",
        "theme_color": "#f59e0b",
        "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}]
    })


@app.route("/icon.svg")
def pwa_icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
      <rect width="512" height="512" rx="115" fill="#090d16"/>
      <circle cx="256" cy="256" r="205" fill="none" stroke="#f59e0b" stroke-width="18"/>
      <path d="M135 335 L210 265 L275 295 L375 170" fill="none" stroke="#00f5a0" stroke-width="34" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="375" cy="170" r="22" fill="#f59e0b"/>
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
        return jsonify(safe)


@app.route("/api/deploy_index", methods=["POST"])
def api_deploy_index():
    d = request.json or {}
    amt = max(10.0, float(d.get("amount_usdt") or 100.0))
    cnt = max(5, min(120, int(float(d.get("coin_count") or 100))))
    with state_lock:
        if STATE.get("mode") == "PAPER":
            # If starting fresh or adding cash, ensure paper cash covers deployment
            if len(STATE.get("holdings", {})) == 0:
                STATE["paper_cash_usdt"] = amt
                STATE["total_deposited_usdt"] = amt
                STATE["daily_open_equity"] = amt
                STATE["equity_history"] = []
            else:
                STATE["paper_cash_usdt"] = round(STATE.get("paper_cash_usdt", 0.0) + amt, 4)
                STATE["total_deposited_usdt"] = round(STATE.get("total_deposited_usdt", 0.0) + amt, 4)
        else:
            STATE["total_deposited_usdt"] = round(STATE.get("total_deposited_usdt", 0.0) + amt, 4)
        STATE["target_coin_count"] = cnt
        save_state()

    try:
        bought, spent = deploy_index_basket(amt, cnt)
        run_index_cycle(force_manual=True)
        return jsonify({"ok": True, "bought": bought, "spent": spent})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/toggle_mode", methods=["POST"])
def api_toggle_mode():
    with state_lock:
        if STATE.get("mode") == "PAPER" and not (STATE.get("api_key") and STATE.get("api_secret")):
            return jsonify({"ok": False, "error": "Connect your Gate.io Spot API keys before switching to LIVE mode."}), 400
        STATE["mode"] = "LIVE" if STATE.get("mode") == "PAPER" else "PAPER"
        add_log(f"🔀 Switched IndexVault Mode to: {STATE['mode']}")
        save_state()
    run_index_cycle(force_manual=True)
    return jsonify({"ok": True})


@app.route("/api/toggle_rebalance", methods=["POST"])
def api_toggle_rebalance():
    with state_lock:
        STATE["auto_rebalance"] = not bool(STATE.get("auto_rebalance", True))
        status = "ON (Auto-Compounding)" if STATE["auto_rebalance"] else "OFF (Pure Buy & Hold)"
        add_log(f"♻️ Auto-Rebalance switched {status}.")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    d = request.json or {}
    with state_lock:
        if d.get("rebalance_trigger_pct"):
            STATE["rebalance_trigger_pct"] = max(5.0, min(100.0, float(d["rebalance_trigger_pct"])))
        if d.get("live_min_order_usdt"):
            STATE["live_min_order_usdt"] = max(3.0, float(d["live_min_order_usdt"]))
        add_log(f"⚙️ Saved Rules: Rebalance Trigger=+{STATE['rebalance_trigger_pct']}% | Live Min Order=${STATE['live_min_order_usdt']:.2f}")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/reset_paper", methods=["POST"])
def api_reset_paper():
    d = request.json or {}
    amt = max(10.0, float(d.get("amount_usdt") or 100.0))
    with state_lock:
        STATE["holdings"] = {}
        STATE["paper_cash_usdt"] = amt
        STATE["total_deposited_usdt"] = amt
        STATE["compound_pool_usdt"] = 0.0
        STATE["daily_open_equity"] = amt
        STATE["equity_history"] = []
        STATE["rebalance_ledger"] = []
        STATE["stats"] = {"total_rebalances": 0, "compounded_profit_usdt": 0.0}
        add_log(f"🧹 IndexVault Paper Portfolio reset to ${amt:.2f} USDT.")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    run_index_cycle(force_manual=True)
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
        return jsonify({"ok": False, "error": "Paste both Gate.io Spot API Key and Secret"}), 400
    try:
        u = float(fetch_spot_accounts(ak, asec).get("USDT", 0.0))
        with state_lock:
            STATE["api_connected"] = True
            STATE["live_spot_usdt"] = u
            STATE["api_status_msg"] = f"Gate.io Spot Connected (Available: ${u:.2f} USDT)"
            add_log(f"🔗 Gate.io Spot API Verified! Available: ${u:.2f} USDT")
            save_state()
        return jsonify({"ok": True, "usdt_balance": u})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:80]}), 400


# =============================================================================
# PEAK INSTITUTIONAL OBSIDIAN-GLASS UI — INDEXVAULT 100 PRO
# =============================================================================
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#f59e0b" />
  <link rel="manifest" href="/manifest.json" />
  <link rel="icon" type="image/svg+xml" href="/icon.svg" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet" />
  <title>IndexVault 100 PRO | Crypto S&P 500 Compounder</title>
  <style>
    :root {
      --bg: #04060a;
      --card: rgba(13, 17, 28, 0.84);
      --border: rgba(255, 255, 255, 0.08);
      --gold: #f59e0b;
      --gold-soft: rgba(245, 158, 11, 0.14);
      --green: #00f5a0;
      --green-soft: rgba(0, 245, 160, 0.12);
      --cyan: #38bdf8;
      --red: #ff3b6b;
      --text: #f1f5f9;
      --muted: #8b95a9;
      --mono: 'JetBrains Mono', ui-monospace, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }
    body {
      background-color: var(--bg);
      background-image:
        radial-gradient(ellipse 75% 45% at 15% -10%, rgba(245, 158, 11, 0.16), transparent 65%),
        radial-gradient(ellipse 65% 40% at 88% 0%, rgba(0, 245, 160, 0.11), transparent 60%),
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
      color: var(--text);
      padding: 20px 16px 44px;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    .wrap { max-width: 1260px; margin: 0 auto; }

    .topbar {
      background: linear-gradient(135deg, rgba(24, 20, 14, 0.90), rgba(10, 13, 22, 0.95));
      backdrop-filter: blur(22px);
      border: 1px solid rgba(245, 158, 11, 0.38);
      border-radius: 20px;
      padding: 20px 24px;
      margin-bottom: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .topbar-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px; }
    .brand { display: flex; align-items: center; gap: 10px; font-size: 1.5rem; font-weight: 900; letter-spacing: -0.5px; }
    .tag {
      background: linear-gradient(135deg, var(--gold), #fbbf24);
      color: #1a1000; font-size: 0.66rem; font-weight: 900; padding: 4px 11px; border-radius: 99px;
      letter-spacing: 0.6px; text-transform: uppercase; box-shadow: 0 0 18px rgba(245, 158, 11, 0.38);
    }
    .subtitle { color: var(--muted); font-size: 0.79rem; margin-top: 5px; display: flex; align-items: center; gap: 6px; }
    .pulse-dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 10px var(--green);
      animation: pulse 1.8s infinite ease-in-out;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .btn {
      cursor: pointer; border: none; border-radius: 11px; padding: 9px 14px;
      font-weight: 800; font-size: 0.75rem; letter-spacing: 0.2px;
      transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 6px;
    }
    .btn:hover { filter: brightness(1.08); transform: translateY(-1px); }
    .btn:active { transform: scale(0.97); }
    .btn-gold { background: linear-gradient(135deg, var(--gold), #fbbf24); color: #1a1000; box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3); }
    .btn-green { background: linear-gradient(135deg, var(--green), var(--cyan)); color: #041c15; box-shadow: 0 4px 16px rgba(0, 245, 160, 0.25); }
    .btn-ghost { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); color: var(--text); }
    .btn-ghost:hover { border-color: var(--gold); background: var(--gold-soft); }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 18px; }
    .card {
      background: var(--card); backdrop-filter: blur(18px); border: 1px solid var(--border);
      border-radius: 16px; padding: 18px 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .kpi-card { position: relative; overflow: hidden; }
    .kpi-card::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    .kpi-card.glow-gold::before { background: linear-gradient(90deg, transparent, var(--gold), transparent); }
    .kpi-card.glow-green::before { background: linear-gradient(90deg, transparent, var(--green), transparent); }
    .kpi-lbl { font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); }
    .kpi-val { font-family: var(--mono); font-size: 1.65rem; font-weight: 800; margin: 8px 0 4px; letter-spacing: -0.5px; }
    .kpi-sub { font-size: 0.75rem; color: var(--muted); }

    .split { display: grid; grid-template-columns: 1.35fr 1fr; gap: 16px; margin-bottom: 18px; }
    @media (max-width: 960px) { .split { grid-template-columns: 1fr; } }

    .sec-head {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
      margin-bottom: 14px; font-size: 0.76rem; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.8px; color: #cbd5e1;
    }

    /* CHART BOX */
    .chart-wrap {
      background: rgba(4, 6, 12, 0.72); border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px; padding: 14px; height: 210px; position: relative;
    }
    svg.eq-svg { width: 100%; height: 100%; overflow: visible; }

    /* MOVERS */
    .movers-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
    @media (max-width: 600px) { .movers-grid { grid-template-columns: 1fr; } }
    .mover-box {
      background: rgba(255,255,255,0.025); border: 1px solid var(--border);
      border-radius: 12px; padding: 12px;
    }
    .mover-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.76rem; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .mover-row:last-child { border-bottom: none; }

    .form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    label { display: block; font-size: 0.68rem; color: var(--muted); font-weight: 700; margin-bottom: 5px; }
    input {
      width: 100%; background: rgba(5, 7, 13, 0.85); border: 1px solid rgba(255,255,255,0.1);
      color: var(--text); padding: 9px 11px; border-radius: 9px; font-family: var(--mono); font-size: 0.82rem;
    }
    input:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.16); }

    table { width: 100%; border-collapse: collapse; font-size: 0.77rem; }
    th, td { padding: 9px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: var(--mono); }
    th { font-family: 'Inter', sans-serif; color: var(--muted); font-size: 0.65rem; text-transform: uppercase; font-weight: 800; position: sticky; top: 0; background: #0c101b; }

    .logs {
      background: rgba(4, 6, 11, 0.92); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 14px; height: 185px; overflow-y: auto; font-family: var(--mono);
      font-size: 0.72rem; color: #cbd5e1; line-height: 1.65;
    }
    .c-green { color: var(--green); }
    .c-red { color: var(--red); }
    .c-gold { color: var(--gold); }
    .c-cyan { color: var(--cyan); }
  </style>
</head>
<body>
<div class="wrap">

  <!-- TOPBAR -->
  <div class="topbar">
    <div class="topbar-row">
      <div>
        <div class="brand">🏛️ INDEXVAULT 100 <span class="tag">PRO • CRYPTO S&P 500</span></div>
        <div class="subtitle"><span class="pulse-dot"></span><span id="statusLine">Auto-Rebalancing 100-Coin Index • Auto-Junk Filtered</span></div>
      </div>
      <div class="actions">
        <button class="btn btn-green" onclick="installApp()">📲 Install App</button>
        <button class="btn btn-gold" onclick="toggleMode()" id="modeBtn">Mode: PAPER</button>
        <button class="btn btn-ghost" onclick="toggleRebalance()" id="rebBtn">♻️ Auto-Rebalance: ON</button>
        <button class="btn btn-ghost" onclick="sweepNow()">⚡ Refresh</button>
      </div>
    </div>
  </div>

  <!-- 4 KPI CARDS -->
  <div class="kpi-grid">
    <div class="card kpi-card glow-gold">
      <div class="kpi-lbl">Total Index Net Worth</div>
      <div class="kpi-val c-gold" id="netWorthVal">$100.00</div>
      <div class="kpi-sub">Deposited: <strong id="depVal" style="color:#fff;">$100.00</strong> • Coins: <strong id="coinCnt" style="color:#fff;">0</strong></div>
    </div>
    <div class="card kpi-card glow-green">
      <div class="kpi-lbl">All-Time Growth / Profit</div>
      <div class="kpi-val c-green" id="allPnlVal">+$0.00</div>
      <div class="kpi-sub" id="allPnlPct">+0.00% Total Return</div>
    </div>
    <div class="card kpi-card">
      <div class="kpi-lbl">Today's Daily Growth</div>
      <div class="kpi-val c-cyan" id="dailyVal">+$0.00</div>
      <div class="kpi-sub" id="dailyPct">+0.00% Since Daily Open</div>
    </div>
    <div class="card kpi-card">
      <div class="kpi-lbl">Compounded Rebalances</div>
      <div class="kpi-val c-green" id="rebTotalVal">$0.00</div>
      <div class="kpi-sub">Trims: <strong id="rebCnt" style="color:#fff;">0</strong> • Pool: <strong id="poolVal" style="color:#fff;">$0.00</strong></div>
    </div>
  </div>

  <!-- EQUITY CURVE & DEPLOY COMMAND DECK -->
  <div class="split">
    <div class="card">
      <div class="sec-head">
        <span>📈 Daily Money Growth & Equity Curve</span>
        <span id="lastUpd" style="font-family:var(--mono);color:var(--muted);">--:--:--</span>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <div style="text-align:center;color:var(--muted);padding-top:75px;font-size:0.82rem;">Deploy your basket on the right to start tracking your live Daily Equity Curve.</div>
      </div>
      <div class="movers-grid">
        <div class="mover-box">
          <div class="sec-head" style="margin-bottom:8px;font-size:0.7rem;color:var(--green);">🚀 Top 5 Index Gainers</div>
          <div id="topGainers"><div style="color:var(--muted);font-size:0.75rem;">--</div></div>
        </div>
        <div class="mover-box">
          <div class="sec-head" style="margin-bottom:8px;font-size:0.7rem;color:var(--red);">🔻 Top 5 Dip Targets</div>
          <div id="topLaggards"><div style="color:var(--muted);font-size:0.75rem;">--</div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="sec-head"><span>🚀 Launch / Add Cash to Index Basket</span></div>
      <div class="form-grid-2">
        <div><label>Deposit Amount (USDT)</label><input type="number" id="depInput" value="100" step="10" /></div>
        <div><label>Target Coins (Max 100)</label><input type="number" id="cntInput" value="100" min="5" max="120" /></div>
      </div>
      <button class="btn btn-gold" style="width:100%;justify-content:center;padding:12px;margin-bottom:14px;" onclick="deployIndex()" id="deployBtn">🏛️ Buy Top 100 Index Basket Now</button>

      <div class="sec-head"><span>⚙️ Compounding & Live Rules</span></div>
      <div class="form-grid-2">
        <div><label>Rebalance Trim (+%)</label><input type="number" id="cfgTrig" value="25" step="5" /></div>
        <div><label>Live Min / Coin ($)</label><input type="number" id="cfgMin" value="5" step="1" /></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-bottom:14px;">
        <button class="btn btn-ghost" style="justify-content:center;" onclick="saveConfig()">💾 Save Rules</button>
        <button class="btn btn-ghost" onclick="resetPaper()">🧹 Reset Paper</button>
      </div>

      <div class="sec-head"><span>🔑 Gate.io Spot API Keys (Optional Live)</span></div>
      <div class="form-grid-2">
        <input type="password" id="apiKey" placeholder="Gate.io Spot API Key" autocomplete="off" />
        <input type="password" id="apiSec" placeholder="Gate.io Spot Secret" autocomplete="off" />
      </div>
      <button class="btn btn-ghost" style="width:100%;justify-content:center;" onclick="syncKeys()" id="syncBtn">🔗 Connect Gate.io Spot Keys</button>
    </div>
  </div>

  <!-- 100-COIN HOLDINGS TABLE -->
  <div class="card" style="margin-bottom:18px;">
    <div class="sec-head">
      <span>💎 Live Index Basket Holdings</span>
      <input type="text" id="searchBox" placeholder="Filter coin (e.g. BTC, SOL, SUI)..." style="width:220px;padding:6px 10px;font-size:0.75rem;" oninput="renderHoldings()" />
    </div>
    <div style="max-height:340px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Token</th>
            <th>Live Price</th>
            <th>24h %</th>
            <th>Cost Basis</th>
            <th>Current Value</th>
            <th>Net PnL</th>
            <th>Trims</th>
          </tr>
        </thead>
        <tbody id="holdingsBody"></tbody>
      </table>
    </div>
  </div>

  <!-- REBALANCE LEDGER & TELEMETRY -->
  <div class="split">
    <div class="card">
      <div class="sec-head"><span>♻️ Auto-Rebalance Compounding Ledger</span></div>
      <div style="max-height:185px;overflow-y:auto;">
        <table>
          <thead><tr><th>Time</th><th>Winner Trimmed</th><th>Amount</th><th>Reinvested Into</th></tr></thead>
          <tbody id="rebBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="sec-head"><span>🖥️ IndexVault Telemetry</span></div>
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
    if (v < 0.0001) return '$' + v.toFixed(8);
    if (v < 0.01) return '$' + v.toFixed(6);
    if (v < 1) return '$' + v.toFixed(4);
    return '$' + v.toFixed(2);
  }

  async function deployIndex() {
    const btn = document.getElementById('deployBtn');
    const amt = document.getElementById('depInput').value || 100;
    const cnt = document.getElementById('cntInput').value || 100;
    btn.innerText = '⏳ Buying Basket...';
    const r = await fetch('/api/deploy_index', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ amount_usdt: amt, coin_count: cnt })
    });
    const d = await r.json();
    if (d.ok) alert(`Deployed $${d.spent} across ${d.bought} Top Gate.io tokens!`);
    else alert('Notice: ' + (d.error || 'Failed to deploy'));
    btn.innerText = '🏛️ Buy Top 100 Index Basket Now';
    await refresh();
  }

  async function toggleMode() {
    const r = await fetch('/api/toggle_mode', { method: 'POST' });
    const d = await r.json();
    if (!d.ok) alert(d.error);
    await refresh();
  }
  async function toggleRebalance() { await fetch('/api/toggle_rebalance', { method: 'POST' }); await refresh(); }
  async function sweepNow() { await fetch('/api/sweep', { method: 'POST' }); await refresh(); }
  async function saveConfig() {
    await fetch('/api/update_config', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        rebalance_trigger_pct: document.getElementById('cfgTrig').value,
        live_min_order_usdt: document.getElementById('cfgMin').value
      })
    });
    await refresh();
  }
  async function resetPaper() {
    const amt = document.getElementById('depInput').value || 100;
    await fetch('/api/reset_paper', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ amount_usdt: amt })
    });
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
    setTimeout(() => { btn.innerText = '🔗 Connect Gate.io Spot Keys'; }, 2500);
  }

  function renderChart(hist) {
    const wrap = document.getElementById('chartWrap');
    if (!hist || hist.length === 0) return;
    const pts = hist.length === 1 ? [hist[0], hist[0]] : hist;
    const vals = pts.map(p => Number(p.equity));
    const minV = Math.min(...vals) * 0.998;
    const maxV = Math.max(...vals) * 1.002;
    const span = Math.max(0.01, maxV - minV);
    const w = 600, h = 175;
    const coords = pts.map((p, idx) => {
      const x = (idx / (pts.length - 1)) * w;
      const y = h - ((Number(p.equity) - minV) / span) * (h - 20) - 10;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const linePts = coords.join(' ');
    const areaPts = `0,${h} ${linePts} ${w},${h}`;
    const lastEq = vals[vals.length - 1];
    const firstEq = vals[0];
    const strokeCol = lastEq >= firstEq ? '#00f5a0' : '#ff3b6b';

    wrap.innerHTML = `
      <svg class="eq-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${strokeCol}" stop-opacity="0.32" />
            <stop offset="100%" stop-color="${strokeCol}" stop-opacity="0.0" />
          </linearGradient>
        </defs>
        <polygon points="${areaPts}" fill="url(#eqGrad)" />
        <polyline points="${linePts}" fill="none" stroke="${strokeCol}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    `;
  }

  let latestHoldings = [];
  function renderHoldings() {
    const q = (document.getElementById('searchBox').value || '').trim().toUpperCase();
    const list = q ? latestHoldings.filter(h => h.pair.includes(q)) : latestHoldings;
    const body = document.getElementById('holdingsBody');
    if (list.length === 0) {
      body.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:24px;">No coins in basket yet — click "🏛️ Buy Top 100 Index Basket Now" above.</td></tr>`;
      return;
    }
    body.innerHTML = list.map(h => `
      <tr>
        <td style="color:var(--muted);">#${h.rank || '-'}</td>
        <td><strong>${h.pair}</strong></td>
        <td>${fmtPx(h.last_price)}</td>
        <td class="${h.change_24h >= 0 ? 'c-green' : 'c-red'}">${h.change_24h >= 0 ? '+' : ''}${Number(h.change_24h).toFixed(2)}%</td>
        <td>$${Number(h.cost_usdt).toFixed(2)}</td>
        <td><strong>$${Number(h.value_usdt).toFixed(2)}</strong></td>
        <td class="${h.pnl_usdt >= 0 ? 'c-green' : 'c-red'}"><strong>${h.pnl_usdt >= 0 ? '+' : ''}$${Number(h.pnl_usdt).toFixed(2)} (${h.pnl_pct >= 0 ? '+' : ''}${h.pnl_pct}%)</strong></td>
        <td class="c-gold">${h.rebalances || 0}</td>
      </tr>
    `).join('');
  }

  let cfgLoaded = false;
  async function refresh() {
    try {
      const r = await fetch('/api/state');
      const s = await r.json();

      if (!cfgLoaded) {
        document.getElementById('cfgTrig').value = s.rebalance_trigger_pct || 25;
        document.getElementById('cfgMin').value = s.live_min_order_usdt || 5;
        cfgLoaded = true;
      }

      document.getElementById('statusLine').innerText = s.api_status_msg;
      document.getElementById('lastUpd').innerText = s.last_update;

      const isLive = s.mode === 'LIVE';
      const modeBtn = document.getElementById('modeBtn');
      modeBtn.innerText = isLive ? 'Mode: LIVE SPOT' : 'Mode: PAPER';
      modeBtn.className = isLive ? 'btn btn-green' : 'btn btn-gold';

      document.getElementById('rebBtn').innerText = s.auto_rebalance ? '♻️ Auto-Rebalance: ON' : '⏸️ Pure Buy & Hold';

      latestHoldings = Object.values(s.holdings || {}).sort((a, b) => (a.rank || 999) - (b.rank || 999));
      const holdVal = latestHoldings.reduce((acc, h) => acc + Number(h.value_usdt || 0), 0);
      const cashVal = isLive ? Number(s.live_spot_usdt || 0) : Number(s.paper_cash_usdt || 0);
      const poolVal = Number(s.compound_pool_usdt || 0);
      const netWorth = holdVal + cashVal + poolVal;
      const dep = Math.max(0.01, Number(s.total_deposited_usdt || 100));
      const allPnl = netWorth - dep;
      const allPct = (allPnl / dep) * 100;

      const dailyOpen = Math.max(0.01, Number(s.daily_open_equity || netWorth));
      const dailyDiff = netWorth - dailyOpen;
      const dailyPct = (dailyDiff / dailyOpen) * 100;

      document.getElementById('netWorthVal').innerText = '$' + netWorth.toFixed(2);
      document.getElementById('depVal').innerText = '$' + dep.toFixed(2);
      document.getElementById('coinCnt').innerText = latestHoldings.length;

      const allEl = document.getElementById('allPnlVal');
      allEl.innerText = (allPnl >= 0 ? '+$' : '-$') + Math.abs(allPnl).toFixed(2);
      allEl.className = 'kpi-val ' + (allPnl >= 0 ? 'c-green' : 'c-red');
      document.getElementById('allPnlPct').innerText = `${allPct >= 0 ? '+' : ''}${allPct.toFixed(2)}% Total Return`;

      const dEl = document.getElementById('dailyVal');
      dEl.innerText = (dailyDiff >= 0 ? '+$' : '-$') + Math.abs(dailyDiff).toFixed(2);
      dEl.className = 'kpi-val ' + (dailyDiff >= 0 ? 'c-green' : 'c-red');
      document.getElementById('dailyPct').innerText = `${dailyPct >= 0 ? '+' : ''}${dailyPct.toFixed(2)}% Since Daily Open`;

      document.getElementById('rebTotalVal').innerText = '$' + Number(s.stats.compounded_profit_usdt || 0).toFixed(2);
      document.getElementById('rebCnt').innerText = s.stats.total_rebalances || 0;
      document.getElementById('poolVal').innerText = '$' + poolVal.toFixed(2);

      renderChart(s.equity_history || []);
      renderHoldings();

      if (latestHoldings.length > 0) {
        const byPnl = [...latestHoldings].sort((a, b) => b.pnl_pct - a.pnl_pct);
        document.getElementById('topGainers').innerHTML = byPnl.slice(0, 5).map(h => `
          <div class="mover-row"><span>${h.base}</span><strong class="${h.pnl_pct >= 0 ? 'c-green' : 'c-red'}">${h.pnl_pct >= 0 ? '+' : ''}${h.pnl_pct}%</strong></div>
        `).join('');
        document.getElementById('topLaggards').innerHTML = byPnl.slice(-5).reverse().map(h => `
          <div class="mover-row"><span>${h.base}</span><strong class="${h.pnl_pct >= 0 ? 'c-green' : 'c-red'}">${h.pnl_pct >= 0 ? '+' : ''}${h.pnl_pct}%</strong></div>
        `).join('');
      }

      const rb = document.getElementById('rebBody');
      if (s.rebalance_ledger && s.rebalance_ledger.length > 0) {
        rb.innerHTML = s.rebalance_ledger.map(r => `
          <tr><td>${r.time}</td><td><strong>${r.winner}</strong></td><td class="c-gold">${r.trimmed}</td><td class="c-green">${r.action}</td></tr>
        `).join('');
      } else {
        rb.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--muted);padding:18px;">Waiting for a coin to surge +${s.rebalance_trigger_pct}% above target weight...</td></tr>`;
      }

      document.getElementById('logBox').innerHTML = (s.logs || []).join('<br/>');
    } catch (e) {}
  }

  setInterval(refresh, 5000);
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
