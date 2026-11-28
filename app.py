import os
import time
import uuid
import math
import json
import hmac
import base64
import hashlib
import threading
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

STATE_FILE = "indexvault_500_v3_state.json"

KUCOIN_HOST = "https://api.kucoin.com"
GATE_HOST = "https://api.gateio.ws"
GATE_PREFIX = "/api/v4"

STABLECOINS = {
    "USDT", "USDC", "FDUSD", "DAI", "TUSD", "USDD", "BUSD", "PYUSD",
    "USDE", "EUR", "EURT", "USTC", "USDP", "GUSD", "FRAX", "USD1", "RLUSD", "USDG"
}
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "2L", "2S", "UP", "DOWN", "BULL", "BEAR")

HTTP_SESSION = requests.Session()
adapter = HTTPAdapter(
    pool_connections=32,
    pool_maxsize=32,
    max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[502, 503, 504])
)
HTTP_SESSION.mount("https://", adapter)

META_CACHE = {"KUCOIN": {"data": {}, "ts": 0.0}, "GATEIO": {"data": {}, "ts": 0.0}}

JOB_STATUS = {
    "running": False,
    "title": "",
    "done": 0,
    "total": 0,
    "error": ""
}

DEFAULT_STATE = {
    "exchange": "KUCOIN",           # "KUCOIN" or "GATEIO"
    "mode": "PAPER",                # "PAPER" (100% Live-Parity Sim) or "LIVE"
    "auto_rebalance": True,
    "target_coin_count": 500,
    "rebalance_trigger_pct": 25.0,
    "kucoin_api_key": os.environ.get("KUCOIN_API_KEY", "").strip(),
    "kucoin_api_secret": os.environ.get("KUCOIN_API_SECRET", "").strip(),
    "kucoin_passphrase": os.environ.get("KUCOIN_API_PASSPHRASE", "").strip(),
    "gate_api_key": os.environ.get("GATE_API_KEY", "").strip(),
    "gate_api_secret": os.environ.get("GATE_API_SECRET", "").strip(),
    "api_connected": False,
    "api_status_msg": "KuCoin Live-Parity Paper Mode — Real Bid/Ask, Exact Class A/B/C Fees & Min Order Sizes",
    "live_spot_usdt": 0.0,
    "paper_cash_usdt": 100.0,
    "harvested_usdt_vault": 0.0,
    "total_deposited_usdt": 100.0,
    "compound_pool_usdt": 0.0,
    "holdings": {},
    "equity_history": [],
    "daily_open_equity": 100.0,
    "daily_open_date": "",
    "rebalance_ledger": [],
    "stats": {
        "total_rebalances": 0,
        "compounded_profit_usdt": 0.0,
        "total_harvests": 0,
        "total_fees_paid_usdt": 0.0
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


def floor_to_step(qty, step_str):
    try:
        d_qty = Decimal(str(qty))
        d_step = Decimal(str(step_str))
        if d_step <= 0:
            return float(d_qty), str(d_qty)
        floored = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        normalized = floored.normalize()
        # Format without scientific notation
        s_out = format(normalized, "f")
        if "." in s_out:
            s_out = s_out.rstrip("0").rstrip(".")
        return float(floored), (s_out if s_out else "0")
    except Exception:
        f_val = math.floor(float(qty) * 10000) / 10000.0
        return f_val, f"{f_val:.4f}"


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
                STATE.update(saved)
                STATE.setdefault("exchange", "KUCOIN")
                STATE.setdefault("harvested_usdt_vault", 0.0)
                STATE.setdefault("stats", {}).setdefault("total_fees_paid_usdt", 0.0)
                STATE["stats"].setdefault("total_harvests", 0)
        except Exception as e:
            add_log(f"State load notice: {e}")

    for env_k, st_k in [
        ("KUCOIN_API_KEY", "kucoin_api_key"),
        ("KUCOIN_API_SECRET", "kucoin_api_secret"),
        ("KUCOIN_API_PASSPHRASE", "kucoin_passphrase"),
        ("GATE_API_KEY", "gate_api_key"),
        ("GATE_API_SECRET", "gate_api_secret"),
    ]:
        val = os.environ.get(env_k, "").strip()
        if val:
            STATE[st_k] = val


def save_state():
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(STATE, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass


# =============================================================================
# EXCHANGE AUTHENTICATION (KUCOIN API v2 + GATE.IO API v4)
# =============================================================================
def kucoin_headers(method, endpoint, body_str="", key=None, secret=None, passphrase=None):
    k = (key if key is not None else STATE.get("kucoin_api_key", "")).strip()
    sec = (secret if secret is not None else STATE.get("kucoin_api_secret", "")).strip()
    pw = (passphrase if passphrase is not None else STATE.get("kucoin_passphrase", "")).strip()

    ts = str(int(time.time() * 1000))
    prehash = ts + method.upper() + endpoint + body_str
    sig = base64.b64encode(
        hmac.new(sec.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    enc_pw = base64.b64encode(
        hmac.new(sec.encode("utf-8"), pw.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

    return {
        "Content-Type": "application/json",
        "KC-API-KEY": k,
        "KC-API-SIGN": sig,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": enc_pw,
        "KC-API-KEY-VERSION": "2"
    }


def gate_headers(method, url, query_string="", body_str="", key=None, secret=None):
    k = (key if key is not None else STATE.get("gate_api_key", "")).strip()
    sec = (secret if secret is not None else STATE.get("gate_api_secret", "")).strip()
    t = str(int(time.time()))
    hashed_payload = hashlib.sha512(body_str.encode("utf-8")).hexdigest()
    s = f"{method}\n{url}\n{query_string}\n{hashed_payload}\n{t}"
    sign = hmac.new(sec.encode("utf-8"), s.encode("utf-8"), hashlib.sha512).hexdigest()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "KEY": k,
        "Timestamp": t,
        "SIGN": sign
    }


def has_active_keys(exch):
    if exch == "KUCOIN":
        return bool(STATE.get("kucoin_api_key") and STATE.get("kucoin_api_secret") and STATE.get("kucoin_passphrase"))
    return bool(STATE.get("gate_api_key") and STATE.get("gate_api_secret"))


# =============================================================================
# UNIFIED EXCHANGE MARKET DATA & LIVE-PARITY METADATA
# =============================================================================
def refresh_exchange_meta(exch):
    now = time.time()
    cached = META_CACHE[exch]
    if cached["data"] and (now - cached["ts"] < 3600):
        return cached["data"]

    meta = {}
    if exch == "KUCOIN":
        r = HTTP_SESSION.get(f"{KUCOIN_HOST}/api/v2/symbols", timeout=12)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("data", []):
            if item.get("quoteCurrency") != "USDT":
                continue
            if not item.get("enableTrading"):
                continue
            if item.get("st"):  # Blocks Special Treatment / delisting risk tokens
                continue
            base = (item.get("baseCurrency") or "").upper()
            if not base or base in STABLECOINS or base.endswith(LEVERAGED_SUFFIXES):
                continue
            sym = item.get("symbol", "")
            fee_cat = int(item.get("feeCategory") or 1)
            # KuCoin VIP 0 real spot fee tiers: Class A=0.10%, Class B=0.20%, Class C=0.30%
            fee_pct = 0.10 if fee_cat == 1 else (0.20 if fee_cat == 2 else 0.30)
            fee_label = "A 0.1%" if fee_cat == 1 else ("B 0.2%" if fee_cat == 2 else "C 0.3%")
            min_funds = float(item.get("minFunds") or 0.1)
            base_min = float(item.get("baseMinSize") or 0.0)
            base_inc = str(item.get("baseIncrement") or "0.0001")
            meta[sym] = {
                "pair": sym,
                "base": base,
                "min_funds": max(0.10, min_funds),
                "base_min_size": base_min,
                "step_str": base_inc,
                "fee_pct": fee_pct,
                "fee_label": fee_label
            }
    else:
        r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/spot/currency_pairs", timeout=12)
        r.raise_for_status()
        for item in r.json():
            pid = item.get("id", "")
            if not pid.endswith("_USDT"):
                continue
            if item.get("trade_status") != "tradable":
                continue
            if item.get("st_tag"):
                continue
            base = (item.get("base") or "").upper()
            if not base or base in STABLECOINS or base.endswith(LEVERAGED_SUFFIXES):
                continue
            prec = int(item.get("amount_precision") or 2)
            step_str = "1" if prec <= 0 else ("0." + "0" * (prec - 1) + "1")
            # Gate.io Spot API enforces 10 USDT minimum order floor
            min_q = max(10.0, float(item.get("min_quote_amount") or 10.0))
            meta[pid] = {
                "pair": pid,
                "base": base,
                "min_funds": min_q,
                "base_min_size": float(item.get("min_base_amount") or 0.0),
                "step_str": step_str,
                "fee_pct": 0.20,
                "fee_label": "0.2%"
            }

    META_CACHE[exch] = {"data": meta, "ts": now}
    return meta


def fetch_exchange_tickers(exch):
    out = {}
    if exch == "KUCOIN":
        r = HTTP_SESSION.get(f"{KUCOIN_HOST}/api/v1/market/allTickers", timeout=10)
        r.raise_for_status()
        for t in r.json().get("data", {}).get("ticker", []):
            sym = t.get("symbol", "")
            if not sym.endswith("-USDT"):
                continue
            last = float(t.get("last") or 0.0)
            bid = float(t.get("buy") or last)
            ask = float(t.get("sell") or last)
            if last <= 0:
                continue
            out[sym] = {
                "last": last,
                "bid": bid if bid > 0 else last,
                "ask": ask if ask > 0 else last,
                "change_24h": float(t.get("changeRate") or 0.0) * 100.0,
                "quote_volume": float(t.get("volValue") or 0.0)
            }
    else:
        r = HTTP_SESSION.get(f"{GATE_HOST}{GATE_PREFIX}/spot/tickers", timeout=10)
        r.raise_for_status()
        for t in r.json():
            cp = t.get("currency_pair", "")
            if not cp.endswith("_USDT"):
                continue
            last = float(t.get("last") or 0.0)
            bid = float(t.get("highest_bid") or last)
            ask = float(t.get("lowest_ask") or last)
            if last <= 0:
                continue
            out[cp] = {
                "last": last,
                "bid": bid if bid > 0 else last,
                "ask": ask if ask > 0 else last,
                "change_24h": float(t.get("change_percentage") or 0.0),
                "quote_volume": float(t.get("quote_volume") or 0.0)
            }
    return out


def fetch_top_candidates(exch, limit=620):
    meta = refresh_exchange_meta(exch)
    tk_map = fetch_exchange_tickers(exch)
    ranked = []
    for pair, m in meta.items():
        tk = tk_map.get(pair)
        if not tk or tk["quote_volume"] < 25000:
            continue
        ask_px = tk["ask"]
        bid_px = tk["bid"]
        # Exact live minimum USDT needed to pass exchange validation (minFunds & baseMinSize * ask)
        real_min = max(m["min_funds"], m["base_min_size"] * ask_px * 1.03)
        if exch == "KUCOIN":
            real_min = max(real_min, 0.12)  # Safety margin over KuCoin's 0.1 USDT floor
        else:
            real_min = max(real_min, 10.10)  # Safety margin over Gate.io's 10 USDT floor

        ranked.append({
            "pair": pair,
            "base": m["base"],
            "last": tk["last"],
            "bid": bid_px,
            "ask": ask_px,
            "change_24h": round(tk["change_24h"], 2),
            "quote_volume": tk["quote_volume"],
            "real_min_usdt": round(real_min, 4),
            "step_str": m["step_str"],
            "fee_pct": m["fee_pct"],
            "fee_label": m["fee_label"]
        })
    ranked.sort(key=lambda x: x["quote_volume"], reverse=True)
    for idx, item in enumerate(ranked[:limit], start=1):
        item["rank"] = idx
    return ranked[:limit]


def fetch_live_usdt_balance(exch, custom_creds=None):
    if exch == "KUCOIN":
        ep = "/api/v1/accounts?currency=USDT&type=trade"
        if custom_creds:
            hdrs = kucoin_headers("GET", ep, "", custom_creds[0], custom_creds[1], custom_creds[2])
        else:
            hdrs = kucoin_headers("GET", ep, "")
        r = HTTP_SESSION.get(f"{KUCOIN_HOST}{ep}", headers=hdrs, timeout=7)
        data = r.json()
        if r.status_code != 200 or str(data.get("code")) != "200000":
            raise ValueError(f"KuCoin Error: {data.get('msg') or r.text[:70]}")
        total = 0.0
        for acc in data.get("data", []):
            if acc.get("type") == "trade" and acc.get("currency") == "USDT":
                total += float(acc.get("available") or 0.0)
        return round(total, 4)
    else:
        url = f"{GATE_PREFIX}/spot/accounts"
        if custom_creds:
            hdrs = gate_headers("GET", url, "", "", custom_creds[0], custom_creds[1])
        else:
            hdrs = gate_headers("GET", url, "", "")
        r = HTTP_SESSION.get(f"{GATE_HOST}{url}", headers=hdrs, timeout=7)
        if r.status_code != 200:
            raise ValueError(f"Gate.io HTTP {r.status_code}: {r.text[:70]}")
        for acc in r.json():
            if acc.get("currency", "").upper() == "USDT":
                return round(float(acc.get("available") or 0.0), 4)
        return 0.0


def place_live_market_buy(exch, pair, usdt_amount):
    if exch == "KUCOIN":
        ep = "/api/v1/orders"
        body = {
            "clientOid": uuid.uuid4().hex,
            "side": "buy",
            "symbol": pair,
            "type": "market",
            "funds": f"{round(float(usdt_amount), 2):.2f}"
        }
        body_str = json.dumps(body, separators=(",", ":"))
        hdrs = kucoin_headers("POST", ep, body_str)
        r = HTTP_SESSION.post(f"{KUCOIN_HOST}{ep}", headers=hdrs, data=body_str, timeout=8)
        d = r.json()
        ok = (r.status_code == 200 and str(d.get("code")) == "200000")
        return ok, d
    else:
        url = f"{GATE_PREFIX}/spot/orders"
        body = {
            "currency_pair": pair,
            "account": "spot",
            "type": "market",
            "side": "buy",
            "amount": f"{round(float(usdt_amount), 2):.2f}",
            "time_in_force": "ioc"
        }
        body_str = json.dumps(body, separators=(",", ":"))
        hdrs = gate_headers("POST", url, "", body_str)
        r = HTTP_SESSION.post(f"{GATE_HOST}{url}", headers=hdrs, data=body_str, timeout=8)
        return r.status_code in (200, 201), r.json()


def place_live_market_sell(exch, pair, coin_qty, step_str):
    clean_f, clean_str = floor_to_step(coin_qty, step_str)
    if clean_f <= 0:
        return False, {"msg": "Quantity below step size"}
    if exch == "KUCOIN":
        ep = "/api/v1/orders"
        body = {
            "clientOid": uuid.uuid4().hex,
            "side": "sell",
            "symbol": pair,
            "type": "market",
            "size": clean_str
        }
        body_str = json.dumps(body, separators=(",", ":"))
        hdrs = kucoin_headers("POST", ep, body_str)
        r = HTTP_SESSION.post(f"{KUCOIN_HOST}{ep}", headers=hdrs, data=body_str, timeout=8)
        d = r.json()
        ok = (r.status_code == 200 and str(d.get("code")) == "200000")
        return ok, d
    else:
        url = f"{GATE_PREFIX}/spot/orders"
        body = {
            "currency_pair": pair,
            "account": "spot",
            "type": "market",
            "side": "sell",
            "amount": clean_str,
            "time_in_force": "ioc"
        }
        body_str = json.dumps(body, separators=(",", ":"))
        hdrs = gate_headers("POST", url, "", body_str)
        r = HTTP_SESSION.post(f"{GATE_HOST}{url}", headers=hdrs, data=body_str, timeout=8)
        return r.status_code in (200, 201), r.json()


# =============================================================================
# LIVE-PARITY ALLOCATION PLANNER (EXACT REAL EXCHANGE LIMITS IN PAPER & LIVE)
# =============================================================================
def plan_realistic_allocation(candidates, budget_usdt, target_count):
    """
    Plans the EXACT coins and USDT amounts that pass the real exchange's
    minFunds & baseMinSize rules. If budget cannot afford target_count coins,
    it automatically scales down to the maximum real number of top coins affordable.
    """
    pool = candidates[:target_count]
    if not pool or budget_usdt <= 0:
        return []

    # Step 1: Trim pool until sum of real minimums fits inside budget
    affordable = []
    running_min = 0.0
    for c in pool:
        if running_min + c["real_min_usdt"] <= budget_usdt:
            affordable.append(c)
            running_min += c["real_min_usdt"]
        else:
            # Skip coins with unusually high minFunds (e.g. high-price coin) if smaller ones fit,
            # or stop once remaining budget is below $0.15
            if (budget_usdt - running_min) < 0.15:
                break

    if not affordable:
        return []

    # Step 2: Equal-weight split while guaranteeing every coin gets >= its real_min_usdt
    equal_share = budget_usdt / len(affordable)
    allocs = []
    remaining = budget_usdt
    for idx, c in enumerate(affordable):
        left_cnt = len(affordable) - idx
        # Reserve enough for remaining coins' minimums
        min_needed_after = sum(x["real_min_usdt"] for x in affordable[idx + 1:])
        max_for_this = max(0.0, remaining - min_needed_after)
        amt = max(c["real_min_usdt"], min(equal_share, max_for_this))
        amt = round(amt, 2)
        if amt < c["real_min_usdt"] or amt > remaining + 0.01:
            continue
        allocs.append((c, amt))
        remaining = round(max(0.0, remaining - amt), 4)

    return allocs


def deploy_index_basket(deposit_usdt, target_count=500, replace_existing=False):
    with state_lock:
        exch = STATE.get("exchange", "KUCOIN")
        mode = STATE.get("mode", "PAPER")

    candidates = fetch_top_candidates(exch, limit=max(620, target_count + 40))
    if not candidates:
        raise ValueError(f"Could not fetch {exch} live market list.")

    if mode == "LIVE":
        if not has_active_keys(exch):
            raise ValueError(f"Connect your {exch} Spot API keys first for Live Mode.")
        avail_usdt = fetch_live_usdt_balance(exch)
        budget = min(deposit_usdt, avail_usdt)
    else:
        budget = deposit_usdt

    plan = plan_realistic_allocation(candidates, budget, target_count)
    if not plan:
        min_ex = "10.10" if exch == "GATEIO" else "0.20"
        raise ValueError(f"Deposit ${budget:.2f} is below {exch}'s real minimum order size (${min_ex} USDT).")

    JOB_STATUS["total"] = len(plan)
    JOB_STATUS["done"] = 0

    bought_count = 0
    total_spent = 0.0
    total_fees = 0.0

    if replace_existing:
        with state_lock:
            STATE["holdings"] = {}

    for idx, (c, alloc_usdt) in enumerate(plan, start=1):
        pair = c["pair"]
        ask_px = c["ask"]
        bid_px = c["bid"]
        fee_pct = c["fee_pct"]

        if mode == "PAPER":
            fee_usdt = round(alloc_usdt * (fee_pct / 100.0), 6)
            net_usdt = alloc_usdt - fee_usdt
            raw_qty = net_usdt / ask_px
            filled_qty, _ = floor_to_step(raw_qty, c["step_str"])
            if filled_qty <= 0:
                continue
            actual_spent = round((filled_qty * ask_px) + fee_usdt, 6)
            fill_px = ask_px
        else:
            ok, resp = place_live_market_buy(exch, pair, alloc_usdt)
            if not ok:
                time.sleep(0.08)
                continue
            fee_usdt = round(alloc_usdt * (fee_pct / 100.0), 6)
            net_usdt = alloc_usdt - fee_usdt
            raw_qty = net_usdt / ask_px
            filled_qty, _ = floor_to_step(raw_qty, c["step_str"])
            actual_spent = alloc_usdt
            fill_px = ask_px
            time.sleep(0.09)

        with state_lock:
            existing = STATE["holdings"].get(pair)
            if existing:
                new_qty = float(existing["qty"]) + filled_qty
                new_cost = float(existing["cost_usdt"]) + actual_spent
                existing["qty"] = new_qty
                existing["cost_usdt"] = round(new_cost, 6)
                existing["avg_price"] = new_cost / max(1e-12, new_qty)
                existing["last_price"] = bid_px
                existing["value_usdt"] = round(new_qty * bid_px, 6)
                existing["change_24h"] = c["change_24h"]
                existing["rank"] = c["rank"]
            else:
                val_now = filled_qty * bid_px
                STATE["holdings"][pair] = {
                    "pair": pair,
                    "base": c["base"],
                    "rank": c["rank"],
                    "qty": filled_qty,
                    "avg_price": fill_px,
                    "cost_usdt": round(actual_spent, 6),
                    "last_price": bid_px,
                    "value_usdt": round(val_now, 6),
                    "pnl_usdt": round(val_now - actual_spent, 6),
                    "pnl_pct": round(((val_now - actual_spent) / max(1e-9, actual_spent)) * 100.0, 2),
                    "change_24h": c["change_24h"],
                    "real_min_usdt": c["real_min_usdt"],
                    "step_str": c["step_str"],
                    "fee_pct": fee_pct,
                    "fee_label": c["fee_label"],
                    "rebalances": 0
                }

        bought_count += 1
        total_spent += actual_spent
        total_fees += fee_usdt
        JOB_STATUS["done"] = idx

    with state_lock:
        if mode == "PAPER" and not replace_existing:
            STATE["paper_cash_usdt"] = round(max(0.0, STATE.get("paper_cash_usdt", 0.0) - total_spent), 4)
        elif mode == "LIVE":
            STATE["live_spot_usdt"] = round(max(0.0, STATE.get("live_spot_usdt", 0.0) - total_spent), 4)
        STATE["stats"]["total_fees_paid_usdt"] = round(STATE["stats"].get("total_fees_paid_usdt", 0.0) + total_fees, 4)
        avg_per = total_spent / max(1, bought_count)
        add_log(
            f"🚀 [{mode} • {exch}] Bought {bought_count} coins (${total_spent:.2f} total, ~${avg_per:.2f}/coin | "
            f"Real Fees Paid: ${total_fees:.4f})"
        )
        save_state()

    return bought_count, round(total_spent, 2)


def bg_deploy_task(amt, cnt):
    try:
        exch = STATE.get("exchange", "KUCOIN")
        JOB_STATUS.update({"running": True, "title": f"{exch}: Deploying ${amt:.2f} (Target {cnt} Coins)", "done": 0, "total": cnt, "error": ""})
        deploy_index_basket(amt, cnt, replace_existing=False)
        run_index_cycle(force_manual=True)
    except Exception as e:
        JOB_STATUS["error"] = str(e)
        with state_lock:
            add_log(f"⚠️ Deploy error: {str(e)[:90]}")
    finally:
        JOB_STATUS["running"] = False


def bg_harvest_task(sell_pct, profit_only):
    try:
        JOB_STATUS.update({"running": True, "title": "Harvesting to USDT Vault & Rebuying Basket", "done": 0, "total": 100, "error": ""})
        with state_lock:
            exch = STATE.get("exchange", "KUCOIN")
            mode = STATE.get("mode", "PAPER")
            holdings = STATE.get("holdings", {})
            if not holdings:
                raise ValueError("No active coins in your Index Basket to harvest.")
            total_val = sum(float(h.get("value_usdt", 0.0)) for h in holdings.values())
            deposited = float(STATE.get("total_deposited_usdt", 100.0))
            target_cnt = int(STATE.get("target_coin_count", 500))
            to_sell = list(holdings.values())

        if total_val <= 1.0:
            raise ValueError("Basket value is too small to harvest.")

        if profit_only:
            profit_avail = round(total_val - deposited, 4)
            if profit_avail <= 0.10:
                raise ValueError(f"Basket (${total_val:.2f}) has no net profit above your ${deposited:.2f} deposit yet.")
            harvest_gross = profit_avail
            actual_pct = round((harvest_gross / total_val) * 100.0, 2)
        else:
            actual_pct = max(1.0, min(95.0, float(sell_pct or 10.0)))
            harvest_gross = round(total_val * (actual_pct / 100.0), 4)

        reinvest_usdt = round(total_val - harvest_gross, 4)
        min_needed = 10.10 if exch == "GATEIO" else 1.00
        if reinvest_usdt < min_needed:
            raise ValueError(f"Remaining reinvest balance would be under ${min_needed:.2f}. Lower your sell percentage.")

        # Calculate exact real exit fees across all coins being sold
        exit_fees = 0.0
        JOB_STATUS["title"] = f"Selling {len(to_sell)} Coins on {exch}"
        JOB_STATUS["total"] = len(to_sell)
        for idx, h in enumerate(to_sell, start=1):
            c_fee = float(h.get("value_usdt", 0.0)) * (float(h.get("fee_pct", 0.10)) / 100.0)
            exit_fees += c_fee
            if mode == "LIVE":
                place_live_market_sell(exch, h["pair"], float(h["qty"]), h.get("step_str", "0.0001"))
                time.sleep(0.08)
            JOB_STATUS["done"] = idx

        harvest_net = round(max(0.0, harvest_gross - (exit_fees * (actual_pct / 100.0))), 4)
        reinvest_after_fee = round(max(0.0, reinvest_usdt - (exit_fees * (1.0 - actual_pct / 100.0))), 4)

        JOB_STATUS["title"] = f"Rebuying Top {target_cnt} Coins on {exch} (${reinvest_after_fee:.2f})"
        bought_cnt, spent = deploy_index_basket(reinvest_after_fee, target_count=target_cnt, replace_existing=True)

        with state_lock:
            STATE["harvested_usdt_vault"] = round(STATE.get("harvested_usdt_vault", 0.0) + harvest_net, 4)
            STATE["stats"]["total_harvests"] = int(STATE["stats"].get("total_harvests", 0)) + 1
            STATE["stats"]["total_fees_paid_usdt"] = round(STATE["stats"].get("total_fees_paid_usdt", 0.0) + exit_fees, 4)
            STATE["rebalance_ledger"].insert(0, {
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "winner": f"HARVEST ({actual_pct:.1f}%)",
                "trimmed": f"+${harvest_net:.2f} USDT",
                "action": f"Locked in Vault & Re-Bought {bought_cnt} Coins (${spent:.2f})"
            })
            STATE["rebalance_ledger"] = STATE["rebalance_ledger"][:40]
            add_log(
                f"💰🔒 [HARVEST & REBUY] Sold {actual_pct:.1f}% (+${harvest_net:.2f} locked in Vault | Exit Fees: ${exit_fees:.4f}) → "
                f"Re-bought {bought_cnt} {exch} coins!"
            )
            save_state()
        run_index_cycle(force_manual=True)
    except Exception as e:
        JOB_STATUS["error"] = str(e)
        with state_lock:
            add_log(f"⚠️ Harvest error: {str(e)[:90]}")
    finally:
        JOB_STATUS["running"] = False


# =============================================================================
# LIVE 5-SECOND TICKER, AUTO-REBALANCE & REAL-TIME EQUITY CURVE
# =============================================================================
def run_index_cycle(force_manual=False):
    if JOB_STATUS.get("running"):
        return
    if not cycle_lock.acquire(blocking=force_manual):
        return

    try:
        with state_lock:
            exch = STATE.get("exchange", "KUCOIN")
            mode = STATE.get("mode", "PAPER")
            auto_reb = bool(STATE.get("auto_rebalance", True))
            reb_trigger = float(STATE.get("rebalance_trigger_pct", 25.0))
            keys_ok = has_active_keys(exch)

        if keys_ok:
            try:
                spot_usdt = fetch_live_usdt_balance(exch)
                with state_lock:
                    STATE["api_connected"] = True
                    STATE["live_spot_usdt"] = spot_usdt
                    STATE["api_status_msg"] = f"{exch} Connected (Trading Account Available: ${spot_usdt:.2f} USDT)"
            except Exception as e:
                with state_lock:
                    STATE["api_connected"] = False
                    STATE["api_status_msg"] = f"{exch} API Notice: {str(e)[:60]}"
        else:
            with state_lock:
                STATE["api_connected"] = False
                min_txt = "~$0.10–$1/coin, Class A/B/C 0.1–0.3% fees" if exch == "KUCOIN" else "$10/coin min, 0.2% fee"
                STATE["api_status_msg"] = f"{exch} Live-Parity Paper Mode ({min_txt})"

        try:
            tk_map = fetch_exchange_tickers(exch)
        except Exception:
            return

        total_holdings_val = 0.0
        with state_lock:
            holdings = STATE.get("holdings", {})
            for pair, h in holdings.items():
                tk = tk_map.get(pair)
                if tk:
                    # Value open holdings at live Best Bid (true liquidation value!)
                    bid_px = tk["bid"]
                    h["last_price"] = bid_px
                    h["ask_price"] = tk["ask"]
                    h["change_24h"] = round(tk["change_24h"], 2)
                val = float(h["qty"]) * float(h["last_price"])
                cost = max(0.000001, float(h["cost_usdt"]))
                pnl_u = val - cost
                pnl_p = (pnl_u / cost) * 100.0
                h["value_usdt"] = round(val, 6)
                h["pnl_usdt"] = round(pnl_u, 6)
                h["pnl_pct"] = round(pnl_p, 2)
                total_holdings_val += val

            # AUTO-REBALANCING COMPOUNDER (ENFORCES REAL EXCHANGE MINS & FEES IN BOTH MODES!)
            if auto_reb and len(holdings) >= 2 and total_holdings_val > 0:
                target_weight_val = total_holdings_val / len(holdings)
                winners = [
                    h for h in holdings.values()
                    if h["value_usdt"] >= target_weight_val * (1.0 + reb_trigger / 100.0)
                ]
                laggards = sorted(holdings.values(), key=lambda x: x["pnl_pct"])

                for w in winners[:3]:
                    trim_usdt = round(w["value_usdt"] - target_weight_val, 6)
                    w_min = float(w.get("real_min_usdt", 1.0))
                    # Live-Parity Rule: only trim if the slice meets the real exchange minimum!
                    if trim_usdt < w_min:
                        continue

                    raw_trim_qty = trim_usdt / max(1e-12, float(w["last_price"]))
                    trim_qty, _ = floor_to_step(raw_trim_qty, w.get("step_str", "0.0001"))
                    if trim_qty <= 0 or trim_qty > float(w["qty"]):
                        continue

                    if mode == "LIVE" and keys_ok:
                        ok_s, _ = place_live_market_sell(exch, w["pair"], trim_qty, w.get("step_str", "0.0001"))
                        if not ok_s:
                            continue

                    actual_trim_gross = trim_qty * float(w["last_price"])
                    sell_fee = round(actual_trim_gross * (float(w.get("fee_pct", 0.10)) / 100.0), 6)
                    net_trimmed = round(actual_trim_gross - sell_fee, 6)

                    w["qty"] = max(0.0, float(w["qty"]) - trim_qty)
                    w["value_usdt"] = round(w["qty"] * float(w["last_price"]), 6)
                    w["rebalances"] = int(w.get("rebalances", 0)) + 1
                    STATE["stats"]["total_fees_paid_usdt"] = round(STATE["stats"].get("total_fees_paid_usdt", 0.0) + sell_fee, 4)

                    # Reinvest into deepest laggard if slice meets laggard's real minFunds
                    target_lag = laggards[0] if laggards and laggards[0]["pair"] != w["pair"] else None
                    reinvest_note = "Swept to Compound Pool"
                    if target_lag and net_trimmed >= float(target_lag.get("real_min_usdt", 1.0)):
                        ok_buy = True
                        if mode == "LIVE" and keys_ok:
                            ok_buy, _ = place_live_market_buy(exch, target_lag["pair"], net_trimmed)
                        if ok_buy:
                            buy_fee = round(net_trimmed * (float(target_lag.get("fee_pct", 0.10)) / 100.0), 6)
                            buy_net = net_trimmed - buy_fee
                            ask_lag = float(target_lag.get("ask_price") or target_lag["last_price"])
                            add_qty, _ = floor_to_step(buy_net / max(1e-12, ask_lag), target_lag.get("step_str", "0.0001"))
                            target_lag["qty"] = float(target_lag["qty"]) + add_qty
                            target_lag["cost_usdt"] = round(float(target_lag["cost_usdt"]) + net_trimmed, 6)
                            target_lag["value_usdt"] = round(target_lag["qty"] * float(target_lag["last_price"]), 6)
                            STATE["stats"]["total_fees_paid_usdt"] = round(STATE["stats"].get("total_fees_paid_usdt", 0.0) + buy_fee, 4)
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

            # LIVE 5s EQUITY CURVE
            cash_val = float(STATE.get("live_spot_usdt", 0.0)) if mode == "LIVE" else float(STATE.get("paper_cash_usdt", 0.0))
            pool_val = float(STATE.get("compound_pool_usdt", 0.0))
            vault_val = float(STATE.get("harvested_usdt_vault", 0.0))
            total_equity = round(total_holdings_val + cash_val + pool_val + vault_val, 4)
            deposited = max(0.01, float(STATE.get("total_deposited_usdt", 100.0)))
            total_pnl = round(total_equity - deposited, 4)
            total_pnl_pct = round((total_pnl / deposited) * 100.0, 2)

            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            if STATE.get("daily_open_date") != today_str:
                STATE["daily_open_date"] = today_str
                STATE["daily_open_equity"] = total_equity

            now_ts = time.time()
            hist = STATE.get("equity_history", [])
            last_eq = float(hist[-1]["equity"]) if hist else -1.0
            if not hist or abs(total_equity - last_eq) >= 0.0005 or (now_ts - float(hist[-1].get("ts", 0.0)) >= 10.0):
                hist.append({
                    "ts": now_ts,
                    "date": today_str,
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "equity": total_equity,
                    "deposited": deposited,
                    "pnl_usdt": total_pnl,
                    "pnl_pct": total_pnl_pct
                })
                STATE["equity_history"] = hist[-180:]

            STATE["cycle_count"] = STATE.get("cycle_count", 0) + 1
            STATE["last_update"] = datetime.utcnow().strftime("%H:%M:%S")
            STATE["last_worker_heartbeat"] = now_ts
            save_state()
    finally:
        cycle_lock.release()


def background_worker():
    load_state()
    add_log("🏛️ IndexVault 500 v3.0 Online — KuCoin + Gate.io Live-Parity Simulation Active.")
    while True:
        run_index_cycle(force_manual=False)
        time.sleep(5.0)


def ensure_worker_alive():
    global worker_started
    now = time.time()
    if not worker_started or (now - STATE.get("last_worker_heartbeat", 0) > 25):
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
        "name": "IndexVault 500 PRO",
        "short_name": "IndexVault",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#04060a",
        "theme_color": "#00d09c",
        "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}]
    })


@app.route("/icon.svg")
def pwa_icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
      <rect width="512" height="512" rx="115" fill="#071014"/>
      <circle cx="256" cy="256" r="205" fill="none" stroke="#00d09c" stroke-width="18"/>
      <path d="M135 335 L210 265 L275 295 L375 170" fill="none" stroke="#f59e0b" stroke-width="34" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="375" cy="170" r="22" fill="#00d09c"/>
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
        for sec_k in ("kucoin_api_key", "kucoin_api_secret", "kucoin_passphrase", "gate_api_key", "gate_api_secret"):
            safe[sec_k] = ""
        safe["job"] = JOB_STATUS.copy()
        return jsonify(safe)


@app.route("/api/preview_capacity", methods=["POST"])
def api_preview_capacity():
    d = request.json or {}
    amt = max(1.0, float(d.get("amount_usdt") or 100.0))
    cnt = max(5, min(600, int(float(d.get("coin_count") or 500))))
    exch = STATE.get("exchange", "KUCOIN")
    try:
        cands = fetch_top_candidates(exch, limit=max(620, cnt + 20))
        plan = plan_realistic_allocation(cands, amt, cnt)
        est_fees = sum(a * (c["fee_pct"] / 100.0) for c, a in plan)
        avg_cost = (sum(a for _, a in plan) / len(plan)) if plan else 0.0
        return jsonify({
            "ok": True,
            "exchange": exch,
            "buyable_coins": len(plan),
            "avg_per_coin": round(avg_cost, 2),
            "est_fees_usdt": round(est_fees, 4),
            "total_available": len(cands)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:80]}), 400


@app.route("/api/switch_exchange", methods=["POST"])
def api_switch_exchange():
    if JOB_STATUS.get("running"):
        return jsonify({"ok": False, "error": "Wait for the current background job to finish."}), 400
    d = request.json or {}
    new_ex = (d.get("exchange") or "KUCOIN").upper()
    if new_ex not in ("KUCOIN", "GATEIO"):
        return jsonify({"ok": False, "error": "Invalid exchange"}), 400
    with state_lock:
        if STATE.get("exchange") != new_ex:
            STATE["exchange"] = new_ex
            STATE["mode"] = "PAPER"
            STATE["holdings"] = {}
            STATE["paper_cash_usdt"] = 100.0
            STATE["harvested_usdt_vault"] = 0.0
            STATE["total_deposited_usdt"] = 100.0
            STATE["compound_pool_usdt"] = 0.0
            STATE["daily_open_equity"] = 100.0
            STATE["equity_history"] = []
            STATE["rebalance_ledger"] = []
            STATE["stats"] = {"total_rebalances": 0, "compounded_profit_usdt": 0.0, "total_harvests": 0, "total_fees_paid_usdt": 0.0}
            add_log(f"🔀 Switched Exchange to {new_ex} (Live-Parity Paper Mode reset to $100.00).")
            save_state()
    run_index_cycle(force_manual=True)
    return jsonify({"ok": True})


@app.route("/api/deploy_index", methods=["POST"])
def api_deploy_index():
    if JOB_STATUS.get("running"):
        return jsonify({"ok": False, "error": "A background job is already running. Please wait."}), 400
    d = request.json or {}
    amt = max(1.0, float(d.get("amount_usdt") or 100.0))
    cnt = max(5, min(600, int(float(d.get("coin_count") or 500))))
    with state_lock:
        if STATE.get("mode") == "PAPER":
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

    threading.Thread(target=bg_deploy_task, args=(amt, cnt), daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.route("/api/harvest_and_rebuy", methods=["POST"])
def api_harvest_and_rebuy():
    if JOB_STATUS.get("running"):
        return jsonify({"ok": False, "error": "A background job is already running. Please wait."}), 400
    d = request.json or {}
    profit_only = bool(d.get("profit_only", False))
    sell_pct = d.get("sell_pct", 10.0)
    threading.Thread(target=bg_harvest_task, args=(sell_pct, profit_only), daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.route("/api/toggle_mode", methods=["POST"])
def api_toggle_mode():
    with state_lock:
        exch = STATE.get("exchange", "KUCOIN")
        if STATE.get("mode") == "PAPER" and not has_active_keys(exch):
            return jsonify({"ok": False, "error": f"Connect your {exch} Spot API keys before switching to LIVE mode."}), 400
        STATE["mode"] = "LIVE" if STATE.get("mode") == "PAPER" else "PAPER"
        add_log(f"🔀 Switched {exch} Mode to: {STATE['mode']}")
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
        add_log(f"⚙️ Saved Rebalance Trigger: +{STATE['rebalance_trigger_pct']}%")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/reset_paper", methods=["POST"])
def api_reset_paper():
    d = request.json or {}
    amt = max(1.0, float(d.get("amount_usdt") or 100.0))
    with state_lock:
        STATE["holdings"] = {}
        STATE["paper_cash_usdt"] = amt
        STATE["harvested_usdt_vault"] = 0.0
        STATE["total_deposited_usdt"] = amt
        STATE["compound_pool_usdt"] = 0.0
        STATE["daily_open_equity"] = amt
        STATE["equity_history"] = []
        STATE["rebalance_ledger"] = []
        STATE["stats"] = {"total_rebalances": 0, "compounded_profit_usdt": 0.0, "total_harvests": 0, "total_fees_paid_usdt": 0.0}
        add_log(f"🧹 Paper Portfolio reset to ${amt:.2f} USDT.")
        save_state()
    return jsonify({"ok": True})


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    run_index_cycle(force_manual=True)
    return jsonify({"ok": True})


@app.route("/api/sync_keys", methods=["POST"])
def api_sync_keys():
    d = request.json or {}
    exch = STATE.get("exchange", "KUCOIN")
    rk = (d.get("api_key") or "").strip().replace(" ", "").replace("\n", "")
    rs = (d.get("api_secret") or "").strip().replace(" ", "").replace("\n", "")
    rp = (d.get("api_passphrase") or "").strip().replace("\n", "")

    if exch == "KUCOIN":
        if not rk or not rs or not rp:
            return jsonify({"ok": False, "error": "Enter KuCoin API Key, Secret, and Passphrase"}), 400
        try:
            u = fetch_live_usdt_balance("KUCOIN", (rk, rs, rp))
            with state_lock:
                STATE["kucoin_api_key"] = rk
                STATE["kucoin_api_secret"] = rs
                STATE["kucoin_passphrase"] = rp
                STATE["api_connected"] = True
                STATE["live_spot_usdt"] = u
                STATE["api_status_msg"] = f"KuCoin Connected (Trading Account: ${u:.2f} USDT)"
                add_log(f"🔗 KuCoin API Verified! Trading Account Available: ${u:.2f} USDT")
                save_state()
            return jsonify({"ok": True, "usdt_balance": u})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)[:85]}), 400
    else:
        if not rk or not rs:
            return jsonify({"ok": False, "error": "Enter Gate.io API Key and Secret"}), 400
        try:
            u = fetch_live_usdt_balance("GATEIO", (rk, rs))
            with state_lock:
                STATE["gate_api_key"] = rk
                STATE["gate_api_secret"] = rs
                STATE["api_connected"] = True
                STATE["live_spot_usdt"] = u
                STATE["api_status_msg"] = f"Gate.io Connected (Spot Available: ${u:.2f} USDT)"
                add_log(f"🔗 Gate.io API Verified! Spot Available: ${u:.2f} USDT")
                save_state()
            return jsonify({"ok": True, "usdt_balance": u})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)[:85]}), 400


# =============================================================================
# PEAK INSTITUTIONAL OBSIDIAN-GLASS UI — INDEXVAULT 500 PRO v3.0
# =============================================================================
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#00d09c" />
  <link rel="manifest" href="/manifest.json" />
  <link rel="icon" type="image/svg+xml" href="/icon.svg" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet" />
  <title>IndexVault 500 PRO | KuCoin & Gate.io</title>
  <style>
    :root {
      --bg: #04060a;
      --card: rgba(12, 17, 26, 0.86);
      --border: rgba(255, 255, 255, 0.08);
      --ku: #00d09c;
      --ku-soft: rgba(0, 208, 156, 0.14);
      --gold: #f59e0b;
      --gold-soft: rgba(245, 158, 11, 0.14);
      --green: #00f5a0;
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
        radial-gradient(ellipse 75% 45% at 15% -10%, rgba(0, 208, 156, 0.16), transparent 65%),
        radial-gradient(ellipse 65% 40% at 88% 0%, rgba(245, 158, 11, 0.12), transparent 60%),
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
    ::-webkit-scrollbar-thumb:hover { background: var(--ku); }

    .wrap { max-width: 1280px; margin: 0 auto; }

    .topbar {
      background: linear-gradient(135deg, rgba(10, 26, 24, 0.92), rgba(10, 13, 22, 0.96));
      backdrop-filter: blur(22px);
      border: 1px solid rgba(0, 208, 156, 0.38);
      border-radius: 20px;
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .topbar-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px; }
    .brand { display: flex; align-items: center; gap: 10px; font-size: 1.52rem; font-weight: 900; letter-spacing: -0.5px; }
    .tag {
      background: linear-gradient(135deg, var(--ku), var(--cyan));
      color: #021c16; font-size: 0.66rem; font-weight: 900; padding: 4px 11px; border-radius: 99px;
      letter-spacing: 0.6px; text-transform: uppercase; box-shadow: 0 0 18px rgba(0, 208, 156, 0.35);
    }
    .subtitle { color: var(--muted); font-size: 0.78rem; margin-top: 5px; display: flex; align-items: center; gap: 6px; }
    .pulse-dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 10px var(--green);
      animation: pulse 1.6s infinite ease-in-out;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.8); } }

    /* EXCHANGE SWITCHER PILL */
    .exch-switch {
      display: inline-flex; background: rgba(4, 7, 12, 0.85); border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px; padding: 3px; gap: 3px;
    }
    .exch-btn {
      border: none; background: transparent; color: var(--muted); font-weight: 800;
      font-size: 0.72rem; padding: 6px 12px; border-radius: 9px; cursor: pointer; transition: all 0.15s;
    }
    .exch-btn.active-ku { background: linear-gradient(135deg, var(--ku), #00a87e); color: #021c16; }
    .exch-btn.active-gate { background: linear-gradient(135deg, var(--gold), #fbbf24); color: #1a1000; }

    .job-banner {
      display: none; background: linear-gradient(135deg, rgba(0, 208, 156, 0.14), rgba(245, 158, 11, 0.10));
      border: 1px solid rgba(0, 208, 156, 0.45); border-radius: 14px; padding: 14px 18px; margin-bottom: 16px;
    }
    .job-banner.active { display: block; }
    .job-track { height: 8px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; margin-top: 8px; }
    .job-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--ku), var(--gold)); transition: width 0.3s ease; }

    .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .btn {
      cursor: pointer; border: none; border-radius: 11px; padding: 9px 14px;
      font-weight: 800; font-size: 0.74rem; letter-spacing: 0.2px;
      transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 6px;
    }
    .btn:hover { filter: brightness(1.08); transform: translateY(-1px); }
    .btn:active { transform: scale(0.97); }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
    .btn-ku { background: linear-gradient(135deg, var(--ku), var(--cyan)); color: #021c16; box-shadow: 0 4px 16px rgba(0, 208, 156, 0.28); }
    .btn-gold { background: linear-gradient(135deg, var(--gold), #fbbf24); color: #1a1000; box-shadow: 0 4px 16px rgba(245, 158, 11, 0.28); }
    .btn-ghost { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); color: var(--text); }
    .btn-ghost:hover { border-color: var(--ku); background: var(--ku-soft); }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 18px; }
    .card {
      background: var(--card); backdrop-filter: blur(18px); border: 1px solid var(--border);
      border-radius: 16px; padding: 18px 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .kpi-card { position: relative; overflow: hidden; }
    .kpi-card::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    .kpi-card.glow-ku::before { background: linear-gradient(90deg, transparent, var(--ku), transparent); }
    .kpi-card.glow-gold::before { background: linear-gradient(90deg, transparent, var(--gold), transparent); }
    .kpi-card.glow-cyan::before { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
    .kpi-lbl { font-size: 0.67rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); }
    .kpi-val { font-family: var(--mono); font-size: 1.62rem; font-weight: 800; margin: 8px 0 4px; letter-spacing: -0.5px; }
    .kpi-sub { font-size: 0.74rem; color: var(--muted); }

    .split { display: grid; grid-template-columns: 1.32fr 1fr; gap: 16px; margin-bottom: 18px; }
    @media (max-width: 960px) { .split { grid-template-columns: 1fr; } }

    .sec-head {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
      margin-bottom: 14px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.8px; color: #cbd5e1;
    }

    .chart-wrap {
      background: rgba(4, 6, 12, 0.78); border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px; padding: 14px 16px; height: 235px; position: relative;
    }
    svg.eq-svg { width: 100%; height: 100%; overflow: visible; }
    .chart-badge {
      position: absolute; top: 10px; left: 14px; font-family: var(--mono); font-size: 0.72rem;
      background: rgba(13, 17, 28, 0.88); border: 1px solid rgba(255,255,255,0.1);
      padding: 4px 10px; border-radius: 99px; color: var(--muted);
    }

    .harvest-box {
      background: linear-gradient(135deg, rgba(0, 208, 156, 0.08), rgba(56, 189, 248, 0.05));
      border: 1px solid rgba(0, 208, 156, 0.30); border-radius: 14px; padding: 14px; margin-bottom: 14px;
    }
    .cap-banner {
      background: rgba(245, 158, 11, 0.09); border: 1px solid rgba(245, 158, 11, 0.32);
      border-radius: 10px; padding: 9px 12px; font-family: var(--mono); font-size: 0.73rem;
      color: #fde68a; margin-bottom: 10px; line-height: 1.5;
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .chip {
      background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
      color: #cbd5e1; font-size: 0.7rem; font-weight: 800; padding: 5px 11px; border-radius: 99px; cursor: pointer;
    }
    .chip:hover { border-color: var(--ku); color: #fff; background: var(--ku-soft); }
    .preview-strip {
      font-family: var(--mono); font-size: 0.72rem; color: var(--muted);
      background: rgba(4, 6, 12, 0.65); border-radius: 9px; padding: 8px 10px; margin-bottom: 10px;
      display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px;
    }

    .movers-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
    @media (max-width: 600px) { .movers-grid { grid-template-columns: 1fr; } }
    .mover-box {
      background: rgba(255,255,255,0.025); border: 1px solid var(--border);
      border-radius: 12px; padding: 12px;
    }
    .mover-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.76rem; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .mover-row:last-child { border-bottom: none; }

    .form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .form-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    @media (max-width: 560px) { .form-grid-3 { grid-template-columns: 1fr; } }
    label { display: block; font-size: 0.67rem; color: var(--muted); font-weight: 700; margin-bottom: 5px; }
    input {
      width: 100%; background: rgba(5, 7, 13, 0.85); border: 1px solid rgba(255,255,255,0.1);
      color: var(--text); padding: 9px 11px; border-radius: 9px; font-family: var(--mono); font-size: 0.81rem;
    }
    input:focus { outline: none; border-color: var(--ku); box-shadow: 0 0 0 3px rgba(0, 208, 156, 0.16); }

    table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
    th, td { padding: 9px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: var(--mono); }
    th { font-family: 'Inter', sans-serif; color: var(--muted); font-size: 0.64rem; text-transform: uppercase; font-weight: 800; position: sticky; top: 0; background: #0c101b; }
    .fee-pill { font-size: 0.64rem; padding: 2px 7px; border-radius: 99px; background: rgba(255,255,255,0.06); color: #cbd5e1; }

    .pager {
      display: flex; justify-content: space-between; align-items: center; margin-top: 12px;
      padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.75rem; color: var(--muted);
    }

    .logs {
      background: rgba(4, 6, 11, 0.92); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 14px; height: 185px; overflow-y: auto; font-family: var(--mono);
      font-size: 0.72rem; color: #cbd5e1; line-height: 1.65;
    }
    .c-green { color: var(--green); }
    .c-red { color: var(--red); }
    .c-gold { color: var(--gold); }
    .c-cyan { color: var(--cyan); }
    .c-ku { color: var(--ku); }
  </style>
</head>
<body>
<div class="wrap">

  <!-- TOPBAR -->
  <div class="topbar">
    <div class="topbar-row">
      <div>
        <div class="brand">🏛️ INDEXVAULT 500 <span class="tag">v3.0 • LIVE-PARITY</span></div>
        <div class="subtitle"><span class="pulse-dot"></span><span id="statusLine">KuCoin Live-Parity Engine • Real Bid/Ask, Exact Fees & Real Min Order Sizes</span></div>
      </div>
      <div class="actions">
        <div class="exch-switch">
          <button class="exch-btn active-ku" id="exKuBtn" onclick="switchExchange('KUCOIN')">KuCoin</button>
          <button class="exch-btn" id="exGateBtn" onclick="switchExchange('GATEIO')">Gate.io</button>
        </div>
        <button class="btn btn-ku" onclick="installApp()">📲 Install App</button>
        <button class="btn btn-gold" onclick="toggleMode()" id="modeBtn">Mode: PAPER</button>
        <button class="btn btn-ghost" onclick="toggleRebalance()" id="rebBtn">♻️ Auto-Rebalance: ON</button>
        <button class="btn btn-ghost" onclick="sweepNow()">⚡ Refresh</button>
      </div>
    </div>
  </div>

  <!-- LIVE BACKGROUND JOB PROGRESS BAR -->
  <div class="job-banner" id="jobBanner">
    <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.8rem;font-weight:800;">
      <span class="c-ku" id="jobTitle">⏳ Running Background Task...</span>
      <span style="font-family:var(--mono);" id="jobPct">0 / 500 (0%)</span>
    </div>
    <div class="job-track"><div class="job-fill" id="jobFill"></div></div>
  </div>

  <!-- 4 KPI CARDS -->
  <div class="kpi-grid">
    <div class="card kpi-card glow-ku">
      <div class="kpi-lbl">Total Index Net Worth</div>
      <div class="kpi-val c-ku" id="netWorthVal">$100.00</div>
      <div class="kpi-sub">Basket: <strong id="basketVal" style="color:#fff;">$0.00</strong> • Coins: <strong id="coinCnt" style="color:#fff;">0</strong></div>
    </div>
    <div class="card kpi-card glow-gold">
      <div class="kpi-lbl">All-Time Net Growth</div>
      <div class="kpi-val c-green" id="allPnlVal">+$0.00</div>
      <div class="kpi-sub" id="allPnlPct">+0.00% After Real Fees</div>
    </div>
    <div class="card kpi-card glow-cyan">
      <div class="kpi-lbl">Today's Daily Growth</div>
      <div class="kpi-val c-cyan" id="dailyVal">+$0.00</div>
      <div class="kpi-sub" id="dailyPct">+0.00% Since Daily Open</div>
    </div>
    <div class="card kpi-card glow-ku">
      <div class="kpi-lbl">🔒 Vault & Real Fees Paid</div>
      <div class="kpi-val c-green" id="vaultVal">$0.00</div>
      <div class="kpi-sub">Fees Paid: <strong class="c-red" id="feesVal">$0.0000</strong> • Rebal: <strong id="rebCnt" style="color:#fff;">0</strong></div>
    </div>
  </div>

  <!-- LIVE EQUITY CURVE & HARVEST / DEPLOY DECK -->
  <div class="split">
    <div class="card">
      <div class="sec-head">
        <span>📈 Live Moving Equity Curve (Valued at Real Best Bid)</span>
        <span id="lastUpd" style="font-family:var(--mono);color:var(--green);">--:--:--</span>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <div style="text-align:center;color:var(--muted);padding-top:85px;font-size:0.82rem;">Deploy your basket on the right to start tracking your live tick-by-tick Equity Curve.</div>
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
      <!-- HARVEST % TO USDT & REBUY ALL COINS -->
      <div class="harvest-box">
        <div class="sec-head" style="margin-bottom:8px;color:var(--ku);"><span>💰 Harvest % to USDT & Rebuy Full Basket</span></div>
        <div class="chip-row">
          <span class="chip" onclick="setHarvestPct(5)">5%</span>
          <span class="chip" onclick="setHarvestPct(10)">10%</span>
          <span class="chip" onclick="setHarvestPct(25)">25%</span>
          <span class="chip" onclick="setHarvestPct(50)">50%</span>
          <span class="chip" style="border-color:var(--gold);color:var(--gold);" onclick="harvestProfitOnly()">✨ Profit Only</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-bottom:8px;">
          <input type="number" id="harvPctInput" value="10" min="1" max="95" step="1" oninput="updateHarvestPreview()" placeholder="Sell %" />
          <button class="btn btn-ku" onclick="runHarvest()" id="harvBtn">🔒 Sell % & Rebuy</button>
        </div>
        <div class="preview-strip">
          <span>Lock to USDT: <strong class="c-green" id="prevLock">$0.00</strong></span>
          <span>Rebuy Basket: <strong class="c-gold" id="prevRebuy">$0.00</strong></span>
        </div>
      </div>

      <!-- LAUNCH / ADD CASH WITH REALISTIC CAPACITY CHECK -->
      <div class="sec-head"><span>🚀 Deploy Basket (Real Exchange Limits Enforced)</span></div>
      <div class="form-grid-2">
        <div><label>Deposit Amount (USDT)</label><input type="number" id="depInput" value="100" step="10" oninput="scheduleCapPreview()" /></div>
        <div><label>Target Coins (Max 600)</label><input type="number" id="cntInput" value="500" min="5" max="600" oninput="scheduleCapPreview()" /></div>
      </div>
      <div class="cap-banner" id="capBanner">Checking real exchange capacity & exact fees...</div>
      <button class="btn btn-gold" style="width:100%;justify-content:center;padding:11px;margin-bottom:14px;" onclick="deployIndex()" id="deployBtn">🏛️ Buy Realistic Index Basket Now</button>

      <!-- RULES & EXCHANGE API KEYS -->
      <div class="sec-head"><span>⚙️ Rebalance Trigger & <span id="apiExLbl">KuCoin</span> API Keys</span></div>
      <div style="display:grid;grid-template-columns:1fr auto auto;gap:8px;margin-bottom:12px;">
        <input type="number" id="cfgTrig" value="25" step="5" title="Rebalance Trim Trigger %" />
        <button class="btn btn-ghost" onclick="saveConfig()">💾 Save %</button>
        <button class="btn btn-ghost" onclick="resetPaper()">🧹 Reset</button>
      </div>
      <div id="apiFieldsWrap" class="form-grid-3">
        <input type="password" id="apiKey" placeholder="KuCoin API Key" autocomplete="off" />
        <input type="password" id="apiSec" placeholder="KuCoin Secret" autocomplete="off" />
        <input type="password" id="apiPass" placeholder="API Passphrase" autocomplete="off" />
      </div>
      <button class="btn btn-ghost" style="width:100%;justify-content:center;" onclick="syncKeys()" id="syncBtn">🔗 Connect KuCoin Spot Keys</button>
    </div>
  </div>

  <!-- PAGINATED 500-COIN HOLDINGS TABLE -->
  <div class="card" style="margin-bottom:18px;">
    <div class="sec-head">
      <span>💎 Live Index Basket Holdings (50 / Page)</span>
      <input type="text" id="searchBox" placeholder="Search coin (e.g. BTC, SOL, SUI)..." style="width:240px;padding:6px 10px;font-size:0.75rem;" oninput="currentPage=1; renderHoldings();" />
    </div>
    <div style="max-height:380px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Token</th>
            <th>Fee Tier</th>
            <th>Bid Price</th>
            <th>24h %</th>
            <th>Cost Basis</th>
            <th>Bid Value</th>
            <th>Net PnL</th>
            <th>Trims</th>
          </tr>
        </thead>
        <tbody id="holdingsBody"></tbody>
      </table>
    </div>
    <div class="pager">
      <button class="btn btn-ghost" style="padding:6px 12px;" id="prevBtn" onclick="changePage(-1)">◀ Prev</button>
      <span id="pageInfo" style="font-family:var(--mono);">Page 1 of 1 (0 coins)</span>
      <button class="btn btn-ghost" style="padding:6px 12px;" id="nextBtn" onclick="changePage(1)">Next ▶</button>
    </div>
  </div>

  <!-- HARVEST / REBALANCE LEDGER & TELEMETRY -->
  <div class="split">
    <div class="card">
      <div class="sec-head"><span>♻️ Harvest & Auto-Rebalance Ledger</span></div>
      <div style="max-height:185px;overflow-y:auto;">
        <table>
          <thead><tr><th>Time</th><th>Event</th><th>Amount</th><th>Result</th></tr></thead>
          <tbody id="rebBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="sec-head"><span>🖥️ IndexVault Live-Parity Telemetry</span></div>
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

  let currentExch = 'KUCOIN';
  let currentBasketVal = 0;
  let capTimer = null;

  function scheduleCapPreview() {
    clearTimeout(capTimer);
    capTimer = setTimeout(fetchCapPreview, 350);
  }

  async function fetchCapPreview() {
    const amt = Number(document.getElementById('depInput').value || 100);
    const cnt = Number(document.getElementById('cntInput').value || 500);
    try {
      const r = await fetch('/api/preview_capacity', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ amount_usdt: amt, coin_count: cnt })
      });
      const d = await r.json();
      if (d.ok) {
        document.getElementById('capBanner').innerHTML =
          `📊 <strong>${d.exchange} Live-Parity Check:</strong> $${amt.toFixed(2)} buys <strong>${d.buyable_coins} real coins</strong> (~$${d.avg_per_coin.toFixed(2)}/coin) • Est. Real Exchange Fee: <strong>$${d.est_fees_usdt.toFixed(4)}</strong>`;
      }
    } catch (e) {}
  }

  async function switchExchange(ex) {
    if (ex === currentExch) return;
    if (latestHoldings.length > 0 && !confirm(`Switch to ${ex}? This will reset your Paper basket to $100 on ${ex} so prices & fees stay accurate.`)) {
      return;
    }
    const r = await fetch('/api/switch_exchange', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ exchange: ex })
    });
    const d = await r.json();
    if (!d.ok) alert(d.error || 'Could not switch exchange');
    currentPage = 1;
    await refresh();
    fetchCapPreview();
  }

  function setHarvestPct(p) {
    document.getElementById('harvPctInput').value = p;
    updateHarvestPreview();
  }
  function updateHarvestPreview() {
    const pct = Math.max(0, Math.min(95, Number(document.getElementById('harvPctInput').value || 0)));
    const lockAmt = currentBasketVal * (pct / 100);
    const rebuyAmt = Math.max(0, currentBasketVal - lockAmt);
    document.getElementById('prevLock').innerText = '$' + lockAmt.toFixed(2);
    document.getElementById('prevRebuy').innerText = '$' + rebuyAmt.toFixed(2);
  }

  async function runHarvest() {
    const pct = Number(document.getElementById('harvPctInput').value || 10);
    const r = await fetch('/api/harvest_and_rebuy', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ sell_pct: pct, profit_only: false })
    });
    const d = await r.json();
    if (!d.ok) alert('Notice: ' + (d.error || 'Could not start harvest'));
    await refresh();
  }

  async function harvestProfitOnly() {
    const r = await fetch('/api/harvest_and_rebuy', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ profit_only: true })
    });
    const d = await r.json();
    if (!d.ok) alert('Notice: ' + (d.error || 'Could not start harvest'));
    await refresh();
  }

  async function deployIndex() {
    const amt = document.getElementById('depInput').value || 100;
    const cnt = document.getElementById('cntInput').value || 500;
    const r = await fetch('/api/deploy_index', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ amount_usdt: amt, coin_count: cnt })
    });
    const d = await r.json();
    if (!d.ok) alert('Notice: ' + (d.error || 'Failed to start deployment'));
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
      body: JSON.stringify({ rebalance_trigger_pct: document.getElementById('cfgTrig').value })
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
    currentPage = 1;
    await refresh();
  }
  async function syncKeys() {
    const btn = document.getElementById('syncBtn');
    btn.innerText = 'Connecting...';
    const r = await fetch('/api/sync_keys', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        api_key: document.getElementById('apiKey').value,
        api_secret: document.getElementById('apiSec').value,
        api_passphrase: document.getElementById('apiPass') ? document.getElementById('apiPass').value : ''
      })
    });
    const d = await r.json();
    if (d.ok) {
      document.getElementById('apiKey').value = '';
      document.getElementById('apiSec').value = '';
      if (document.getElementById('apiPass')) document.getElementById('apiPass').value = '';
      btn.innerText = '✅ Connected';
    } else {
      alert(d.error || 'Check API credentials');
      btn.innerText = '❌ Retry';
    }
    await refresh();
    setTimeout(() => { btn.innerText = `🔗 Connect ${currentExch === 'KUCOIN' ? 'KuCoin' : 'Gate.io'} Spot Keys`; }, 2500);
  }

  function renderChart(hist) {
    const wrap = document.getElementById('chartWrap');
    if (!hist || hist.length === 0) return;
    const pts = hist.length === 1 ? [hist[0], hist[0]] : hist;
    const vals = pts.map(p => Number(p.equity));
    const rawMin = Math.min(...vals);
    const rawMax = Math.max(...vals);
    const pad = Math.max(0.05, (rawMax - rawMin) * 0.15);
    const minV = rawMin - pad;
    const maxV = rawMax + pad;
    const span = Math.max(0.01, maxV - minV);

    const w = 620, h = 190;
    const coords = pts.map((p, idx) => {
      const x = (idx / (pts.length - 1)) * (w - 14) + 4;
      const y = h - ((Number(p.equity) - minV) / span) * (h - 28) - 14;
      return { x, y };
    });
    const linePts = coords.map(c => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' ');
    const lastPt = coords[coords.length - 1];
    const areaPts = `4,${h} ${linePts} ${lastPt.x.toFixed(1)},${h}`;
    const lastEq = vals[vals.length - 1];
    const firstEq = vals[0];
    const strokeCol = lastEq >= firstEq ? '#00f5a0' : '#ff3b6b';

    wrap.innerHTML = `
      <div class="chart-badge">LIVE: <strong style="color:${strokeCol}">$${lastEq.toFixed(4)}</strong> • High $${rawMax.toFixed(2)} / Low $${rawMin.toFixed(2)}</div>
      <svg class="eq-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${strokeCol}" stop-opacity="0.34" />
            <stop offset="100%" stop-color="${strokeCol}" stop-opacity="0.0" />
          </linearGradient>
        </defs>
        <line x1="0" y1="${(h*0.25).toFixed(0)}" x2="${w}" y2="${(h*0.25).toFixed(0)}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4" />
        <line x1="0" y1="${(h*0.50).toFixed(0)}" x2="${w}" y2="${(h*0.50).toFixed(0)}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4" />
        <line x1="0" y1="${(h*0.75).toFixed(0)}" x2="${w}" y2="${(h*0.75).toFixed(0)}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4 4" />
        <polygon points="${areaPts}" fill="url(#eqGrad)" />
        <polyline points="${linePts}" fill="none" stroke="${strokeCol}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />
        <circle cx="${lastPt.x.toFixed(1)}" cy="${lastPt.y.toFixed(1)}" r="5" fill="${strokeCol}" />
        <circle cx="${lastPt.x.toFixed(1)}" cy="${lastPt.y.toFixed(1)}" r="10" fill="${strokeCol}" opacity="0.28">
          <animate attributeName="r" values="5;13;5" dur="1.6s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.45;0.05;0.45" dur="1.6s" repeatCount="indefinite" />
        </circle>
      </svg>
    `;
  }

  let latestHoldings = [];
  let currentPage = 1;
  const PAGE_SIZE = 50;

  function changePage(delta) {
    currentPage += delta;
    renderHoldings();
  }

  function renderHoldings() {
    const q = (document.getElementById('searchBox').value || '').trim().toUpperCase();
    const list = q ? latestHoldings.filter(h => h.pair.includes(q)) : latestHoldings;
    const totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const start = (currentPage - 1) * PAGE_SIZE;
    const pageItems = list.slice(start, start + PAGE_SIZE);

    document.getElementById('pageInfo').innerText = `Page ${currentPage} of ${totalPages} (${list.length} coins)`;
    document.getElementById('prevBtn').disabled = currentPage <= 1;
    document.getElementById('nextBtn').disabled = currentPage >= totalPages;

    const body = document.getElementById('holdingsBody');
    if (pageItems.length === 0) {
      body.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:24px;">No coins in basket yet — click "🏛️ Buy Realistic Index Basket Now" above.</td></tr>`;
      return;
    }
    body.innerHTML = pageItems.map(h => `
      <tr>
        <td style="color:var(--muted);">#${h.rank || '-'}</td>
        <td><strong>${h.pair}</strong></td>
        <td><span class="fee-pill">${h.fee_label || '0.1%'}</span></td>
        <td>${fmtPx(h.last_price)}</td>
        <td class="${h.change_24h >= 0 ? 'c-green' : 'c-red'}">${h.change_24h >= 0 ? '+' : ''}${Number(h.change_24h).toFixed(2)}%</td>
        <td>$${Number(h.cost_usdt).toFixed(3)}</td>
        <td><strong>$${Number(h.value_usdt).toFixed(3)}</strong></td>
        <td class="${h.pnl_usdt >= 0 ? 'c-green' : 'c-red'}"><strong>${h.pnl_usdt >= 0 ? '+' : ''}$${Number(h.pnl_usdt).toFixed(3)} (${h.pnl_pct >= 0 ? '+' : ''}${h.pnl_pct}%)</strong></td>
        <td class="c-gold">${h.rebalances || 0}</td>
      </tr>
    `).join('');
  }

  let cfgLoaded = false;
  let lastRenderedExch = '';
  async function refresh() {
    try {
      const r = await fetch('/api/state');
      const s = await r.json();

      currentExch = s.exchange || 'KUCOIN';
      if (lastRenderedExch !== currentExch) {
        lastRenderedExch = currentExch;
        const isKu = currentExch === 'KUCOIN';
        document.getElementById('exKuBtn').className = 'exch-btn ' + (isKu ? 'active-ku' : '');
        document.getElementById('exGateBtn').className = 'exch-btn ' + (!isKu ? 'active-gate' : '');
        document.getElementById('apiExLbl').innerText = isKu ? 'KuCoin' : 'Gate.io';
        document.getElementById('syncBtn').innerText = isKu ? '🔗 Connect KuCoin Spot Keys' : '🔗 Connect Gate.io Spot Keys';
        const wrap = document.getElementById('apiFieldsWrap');
        if (isKu) {
          wrap.className = 'form-grid-3';
          wrap.innerHTML = `
            <input type="password" id="apiKey" placeholder="KuCoin API Key" autocomplete="off" />
            <input type="password" id="apiSec" placeholder="KuCoin Secret" autocomplete="off" />
            <input type="password" id="apiPass" placeholder="API Passphrase" autocomplete="off" />
          `;
        } else {
          wrap.className = 'form-grid-2';
          wrap.innerHTML = `
            <input type="password" id="apiKey" placeholder="Gate.io API Key" autocomplete="off" />
            <input type="password" id="apiSec" placeholder="Gate.io Secret" autocomplete="off" />
          `;
        }
        fetchCapPreview();
      }

      if (!cfgLoaded) {
        document.getElementById('cfgTrig').value = s.rebalance_trigger_pct || 25;
        document.getElementById('cntInput').value = s.target_coin_count || 500;
        cfgLoaded = true;
      }

      const job = s.job || {};
      const banner = document.getElementById('jobBanner');
      if (job.running) {
        banner.classList.add('active');
        const tot = Math.max(1, Number(job.total || 1));
        const done = Number(job.done || 0);
        const pct = Math.min(100, Math.round((done / tot) * 100));
        document.getElementById('jobTitle').innerText = '⏳ ' + (job.title || 'Processing...');
        document.getElementById('jobPct').innerText = `${done} / ${tot} (${pct}%)`;
        document.getElementById('jobFill').style.width = pct + '%';
        document.getElementById('deployBtn').disabled = true;
        document.getElementById('harvBtn').disabled = true;
      } else {
        banner.classList.remove('active');
        document.getElementById('deployBtn').disabled = false;
        document.getElementById('harvBtn').disabled = false;
      }

      document.getElementById('statusLine').innerText = s.api_status_msg;
      document.getElementById('lastUpd').innerText = 'LIVE ' + s.last_update;

      const isLive = s.mode === 'LIVE';
      const modeBtn = document.getElementById('modeBtn');
      modeBtn.innerText = isLive ? 'Mode: LIVE SPOT' : 'Mode: PAPER';
      modeBtn.className = isLive ? 'btn btn-ku' : 'btn btn-gold';

      document.getElementById('rebBtn').innerText = s.auto_rebalance ? '♻️ Auto-Rebalance: ON' : '⏸️ Pure Buy & Hold';

      latestHoldings = Object.values(s.holdings || {}).sort((a, b) => (a.rank || 999) - (b.rank || 999));
      const holdVal = latestHoldings.reduce((acc, h) => acc + Number(h.value_usdt || 0), 0);
      currentBasketVal = holdVal;
      updateHarvestPreview();

      const cashVal = isLive ? Number(s.live_spot_usdt || 0) : Number(s.paper_cash_usdt || 0);
      const poolVal = Number(s.compound_pool_usdt || 0);
      const vaultVal = Number(s.harvested_usdt_vault || 0);
      const netWorth = holdVal + cashVal + poolVal + vaultVal;
      const dep = Math.max(0.01, Number(s.total_deposited_usdt || 100));
      const allPnl = netWorth - dep;
      const allPct = (allPnl / dep) * 100;

      const dailyOpen = Math.max(0.01, Number(s.daily_open_equity || netWorth));
      const dailyDiff = netWorth - dailyOpen;
      const dailyPct = (dailyDiff / dailyOpen) * 100;

      document.getElementById('netWorthVal').innerText = '$' + netWorth.toFixed(2);
      document.getElementById('basketVal').innerText = '$' + holdVal.toFixed(2);
      document.getElementById('coinCnt').innerText = latestHoldings.length;

      const allEl = document.getElementById('allPnlVal');
      allEl.innerText = (allPnl >= 0 ? '+$' : '-$') + Math.abs(allPnl).toFixed(2);
      allEl.className = 'kpi-val ' + (allPnl >= 0 ? 'c-green' : 'c-red');
      document.getElementById('allPnlPct').innerText = `${allPct >= 0 ? '+' : ''}${allPct.toFixed(2)}% After Real Fees (Dep: $${dep.toFixed(0)})`;

      const dEl = document.getElementById('dailyVal');
      dEl.innerText = (dailyDiff >= 0 ? '+$' : '-$') + Math.abs(dailyDiff).toFixed(2);
      dEl.className = 'kpi-val ' + (dailyDiff >= 0 ? 'c-green' : 'c-red');
      document.getElementById('dailyPct').innerText = `${dailyPct >= 0 ? '+' : ''}${dailyPct.toFixed(2)}% Since Daily Open`;

      document.getElementById('vaultVal').innerText = '$' + vaultVal.toFixed(2);
      document.getElementById('feesVal').innerText = '$' + Number(s.stats.total_fees_paid_usdt || 0).toFixed(4);
      document.getElementById('rebCnt').innerText = s.stats.total_rebalances || 0;

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
        rb.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--muted);padding:18px;">No harvests or rebalances yet.</td></tr>`;
      }

      document.getElementById('logBox').innerHTML = (s.logs || []).join('<br/>');
    } catch (e) {}
  }

  setInterval(refresh, 2500);
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
