"""
MS Trend Matrix [BigBeluga] — Streamlit Live Signal & Accuracy Tracker
=======================================================================
Monitors BTCUSD and ETHUSD perpetuals on Delta Exchange India.
Auto-refreshes every 5 seconds. For every direction flip, opens a simulated
trade (entry = spot at flip moment) and tracks whether target or stop hits first.

TRADE RULES
-----------
                    BTCUSD          ETHUSD
  Target (points):  ±30             ±3
  Stop   (points):  ±40             ±10

A trade closes as:
  WIN       — target hit first
  LOSS      — stop hit first
  REVERSED  — new opposite signal fired before TP/SL (closed at current spot)

RUN LOCALLY
-----------
  pip install streamlit pandas numpy requests
  streamlit run ms_trend_streamlit.py
"""

import time
import os
import csv
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone, timedelta


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
SYMBOLS         = ["BTCUSD", "ETHUSD"]
RESOLUTION      = "5m"
REFRESH_SECONDS = 5
LOOKBACK_BARS   = 3000   # enough history for indicator state to converge to TradingView's value

MS_LEN           = 10
ATR_LENGTH       = 14
ATR_MULT         = 4.0
TARGET_STEP_MULT = 2.0

# Per-symbol simulated trade rules (in price points)
TRADE_RULES = {
    "BTCUSD": {"target": 30, "stop": 40},
    "ETHUSD": {"target": 3,  "stop": 10},
}

BASE_URL = "https://api.india.delta.exchange"
RES_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400,
}

IST = timezone(timedelta(hours=5, minutes=30))
TRADES_CSV = "ms_trend_trades.csv"

# Detect if we're on Streamlit Community Cloud (has STREAMLIT_SERVER_PORT etc.)
ON_CLOUD = os.environ.get("HOSTNAME", "").startswith("streamlit") or \
           os.environ.get("STREAMLIT_RUNTIME_CREDENTIALS_FILE") is not None or \
           os.path.exists("/mount/src")  # cloud mount path


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def to_ist_str(dt):
    if dt is None:
        return "—"
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def fetch_candles(symbol, resolution, lookback_bars):
    sec = RES_SECONDS[resolution]
    end = int(time.time())
    start = end - sec * lookback_bars
    try:
        r = requests.get(
            f"{BASE_URL}/v2/history/candles",
            params={"symbol": symbol, "resolution": resolution,
                    "start": start, "end": end},
            timeout=15,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error reaching Delta India: {e}")

    if r.status_code == 403:
        raise RuntimeError(
            "403 Forbidden — Delta India is blocking this server's IP. "
            "Streamlit Cloud servers are in the US; Delta India often blocks non-IN IPs. "
            "Run the app from a server in India, or use a different host."
        )
    r.raise_for_status()
    data = r.json().get("result", [])
    if not data:
        raise RuntimeError(f"Empty candles for {symbol}")
    df = pd.DataFrame(data)[["time", "open", "high", "low", "close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def fetch_spot_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/v2/tickers/{symbol}", timeout=10)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error reaching Delta India: {e}")
    if r.status_code == 403:
        raise RuntimeError(
            "403 Forbidden — Delta India is blocking this server's IP."
        )
    r.raise_for_status()
    res = r.json().get("result", {})
    for k in ("mark_price", "spot_price", "close", "last_price"):
        if res.get(k):
            return float(res[k])
    raise RuntimeError(f"No price field for {symbol}")


# ═════════════════════════════════════════════════════════════════════════════
#  MS TREND MATRIX
# ═════════════════════════════════════════════════════════════════════════════
def ms_trend_matrix(df, ms_len=MS_LEN, atr_length=ATR_LENGTH,
                    atr_mult=ATR_MULT, target_step_mult=TARGET_STEP_MULT):
    df = df.copy()
    high = df["high"].to_numpy()
    low  = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low  - prev_close)])
    atr = pd.Series(tr).ewm(alpha=1.0 / atr_length, adjust=False).mean().to_numpy()

    ph_at = np.full(n, np.nan)
    pl_at = np.full(n, np.nan)
    last_ph = last_pl = np.nan
    for i in range(ms_len, n - ms_len):
        win_h = high[i - ms_len: i + ms_len + 1]
        win_l = low [i - ms_len: i + ms_len + 1]
        confirm = i + ms_len
        if high[i] == win_h.max() and (win_h == high[i]).sum() == 1:
            last_ph = high[i]
        if low[i]  == win_l.min() and (win_l == low[i]).sum() == 1:
            last_pl = low[i]
        ph_at[confirm] = last_ph
        pl_at[confirm] = last_pl
    ph_at = pd.Series(ph_at).ffill().to_numpy()
    pl_at = pd.Series(pl_at).ffill().to_numpy()

    direction = np.zeros(n, dtype=int)
    cur_dir = 0
    for i in range(1, n):
        if (not np.isnan(ph_at[i]) and close[i] > ph_at[i]
                and close[i-1] <= ph_at[i] and cur_dir != 1):
            cur_dir = 1
        elif (not np.isnan(pl_at[i]) and close[i] < pl_at[i]
                and close[i-1] >= pl_at[i] and cur_dir != -1):
            cur_dir = -1
        direction[i] = cur_dir

    df["direction"] = direction
    return df


def analyze(symbol):
    df = fetch_candles(symbol, RESOLUTION, LOOKBACK_BARS)
    sec = RES_SECONDS[RESOLUTION]
    df["closes_at"] = df["time"] + pd.Timedelta(seconds=sec)
    df = df[df["closes_at"] <= datetime.now(timezone.utc)].drop(columns=["closes_at"]).reset_index(drop=True)

    out = ms_trend_matrix(df)
    last = out.iloc[-1]
    spot = fetch_spot_price(symbol)

    # Find historical flips for diagnostic display
    diffs = out["direction"].diff()
    flip_rows = out[diffs != 0].iloc[1:]   # skip very first row (NaN diff)
    flip_history = []
    for _, r in flip_rows.tail(10).iterrows():
        flip_history.append({
            "time":      r["time"],
            "direction": int(r["direction"]),
            "close":     float(r["close"]),
        })

    return {
        "symbol":        symbol,
        "spot":          spot,
        "bar_time":      last["time"],
        "bar_close":     float(last["close"]),
        "direction":     int(last["direction"]),
        "bars_loaded":   len(df),
        "flip_history":  flip_history,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TRADE STATE MACHINE
# ═════════════════════════════════════════════════════════════════════════════
DIR_NAME  = {1: "LONG", -1: "SHORT", 0: "FLAT"}
DIR_EMOJI = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ FLAT"}


def open_trade(symbol, direction, spot, ts):
    """Build a new trade dict at the moment of a flip."""
    rules = TRADE_RULES[symbol]
    if direction == 1:      # LONG
        target = spot + rules["target"]
        stop   = spot - rules["stop"]
    elif direction == -1:   # SHORT
        target = spot - rules["target"]
        stop   = spot + rules["stop"]
    else:
        return None
    return {
        "symbol":     symbol,
        "direction":  direction,
        "entry_time": ts,
        "entry":      spot,
        "target":     target,
        "stop":       stop,
        "status":     "OPEN",
        "exit_time":  None,
        "exit_price": None,
        "pnl_points": None,
    }


def update_open_trade(trade, current_spot, now_ts):
    """Check if TP or SL has been hit. Returns the (possibly updated) trade."""
    if trade["status"] != "OPEN":
        return trade

    d = trade["direction"]
    # LONG: profit when price ≥ target, loss when price ≤ stop
    # SHORT: profit when price ≤ target, loss when price ≥ stop
    if d == 1:
        if current_spot >= trade["target"]:
            trade["status"] = "WIN"
            trade["exit_price"] = trade["target"]
        elif current_spot <= trade["stop"]:
            trade["status"] = "LOSS"
            trade["exit_price"] = trade["stop"]
    elif d == -1:
        if current_spot <= trade["target"]:
            trade["status"] = "WIN"
            trade["exit_price"] = trade["target"]
        elif current_spot >= trade["stop"]:
            trade["status"] = "LOSS"
            trade["exit_price"] = trade["stop"]

    if trade["status"] in ("WIN", "LOSS"):
        trade["exit_time"]  = now_ts
        trade["pnl_points"] = (trade["exit_price"] - trade["entry"]) * d
    return trade


def close_trade_reversed(trade, current_spot, now_ts):
    """Close an open trade because an opposite signal fired."""
    d = trade["direction"]
    trade["status"]     = "REVERSED"
    trade["exit_time"]  = now_ts
    trade["exit_price"] = current_spot
    trade["pnl_points"] = (current_spot - trade["entry"]) * d
    return trade


def append_trade_csv(trade):
    fieldnames = ["symbol", "direction", "entry_time", "entry", "target", "stop",
                  "status", "exit_time", "exit_price", "pnl_points"]
    new_file = not os.path.exists(TRADES_CSV)
    row = {
        "symbol":     trade["symbol"],
        "direction":  DIR_NAME[trade["direction"]],
        "entry_time": to_ist_str(trade["entry_time"]),
        "entry":      round(trade["entry"], 2),
        "target":     round(trade["target"], 2),
        "stop":       round(trade["stop"], 2),
        "status":     trade["status"],
        "exit_time":  to_ist_str(trade["exit_time"]) if trade["exit_time"] else "",
        "exit_price": round(trade["exit_price"], 2) if trade["exit_price"] is not None else "",
        "pnl_points": round(trade["pnl_points"], 2) if trade["pnl_points"] is not None else "",
    }
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MS Trend Matrix Bot", page_icon="📈", layout="wide")

# Auto-refresh — works correctly on Streamlit Cloud (no worker-blocking)
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="auto_refresh")

# Initialize session state
if "trades"           not in st.session_state: st.session_state.trades = []
if "prev_direction"   not in st.session_state: st.session_state.prev_direction = {s: None for s in SYMBOLS}
if "open_trade"       not in st.session_state: st.session_state.open_trade = {s: None for s in SYMBOLS}
if "baseline_set"     not in st.session_state: st.session_state.baseline_set = False
if "last_refresh"     not in st.session_state: st.session_state.last_refresh = None
if "errors"           not in st.session_state: st.session_state.errors = {}
if "current_state"    not in st.session_state: st.session_state.current_state = {}

# Header
st.title("📈 MS Trend Matrix [BigBeluga] — Live Signal Tracker")
st.caption(f"Delta Exchange India  •  {RESOLUTION} bars  •  auto-refresh every {REFRESH_SECONDS}s")

# Cloud-specific warnings
if ON_CLOUD:
    st.warning(
        "☁️  **Running on Streamlit Cloud.** Two important caveats:\n\n"
        "1. **Filesystem is ephemeral** — the trade CSV is wiped on every app restart "
        "(deploys, idle timeouts, errors). Use the **Download CSV** button to save your history regularly.\n"
        "2. **Geo-block risk** — Delta India may block US-based Streamlit Cloud IPs with 403 errors. "
        "If you see persistent 403s, run the app on a server in India instead."
    )

# Sidebar — controls & rules
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Refresh interval:** {REFRESH_SECONDS}s")
    st.markdown(f"**Timeframe:** {RESOLUTION}")
    st.markdown(f"**Symbols:** {', '.join(SYMBOLS)}")

    st.divider()
    st.subheader("📐 Trade rules (points)")
    rules_df = pd.DataFrame([
        {"Symbol": s, "Target (+)": TRADE_RULES[s]["target"], "Stop (−)": TRADE_RULES[s]["stop"]}
        for s in SYMBOLS
    ])
    st.dataframe(rules_df, hide_index=True, use_container_width=True)

    st.divider()
    # CSV download — works regardless of filesystem persistence
    if st.session_state.trades:
        # Build CSV in memory from session state
        csv_rows = []
        for t in st.session_state.trades:
            csv_rows.append({
                "symbol":     t["symbol"],
                "direction":  DIR_NAME[t["direction"]],
                "entry_time": to_ist_str(t["entry_time"]),
                "entry":      round(t["entry"], 2),
                "target":     round(t["target"], 2),
                "stop":       round(t["stop"], 2),
                "status":     t["status"],
                "exit_time":  to_ist_str(t["exit_time"]) if t["exit_time"] else "",
                "exit_price": round(t["exit_price"], 2) if t["exit_price"] is not None else "",
                "pnl_points": round(t["pnl_points"], 2) if t["pnl_points"] is not None else "",
            })
        csv_buf = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download trade log (CSV)",
            data=csv_buf,
            file_name=f"ms_trend_trades_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("💾 Download trade log (CSV)", disabled=True, use_container_width=True,
                  help="No trades yet to download")

    st.divider()
    if st.button("🗑️ Reset session (clear all trades)", use_container_width=True):
        st.session_state.trades = []
        st.session_state.prev_direction = {s: None for s in SYMBOLS}
        st.session_state.open_trade = {s: None for s in SYMBOLS}
        st.session_state.baseline_set = False
        if os.path.exists(TRADES_CSV):
            os.remove(TRADES_CSV)
        st.success("Reset done.")
        st.rerun()


# ─── Poll Delta India ────────────────────────────────────────────────────────
state = {}
errors = {}
now_ist = datetime.now(IST)

for sym in SYMBOLS:
    try:
        state[sym] = analyze(sym)
    except Exception as e:
        errors[sym] = str(e)

st.session_state.current_state = state
st.session_state.errors = errors
st.session_state.last_refresh = now_ist


# ─── Update trades (TP/SL hits + flips) ─────────────────────────────────────
successes = 0
for sym in SYMBOLS:
    if sym in errors:
        continue
    successes += 1
    s = state[sym]
    new_dir = s["direction"]
    old_dir = st.session_state.prev_direction[sym]
    spot    = s["spot"]

    # Step 1 — update any currently-open trade for this symbol with current spot
    open_t = st.session_state.open_trade[sym]
    if open_t is not None:
        update_open_trade(open_t, spot, now_ist)
        if open_t["status"] in ("WIN", "LOSS"):
            append_trade_csv(open_t)
            st.session_state.open_trade[sym] = None  # closed

    # Step 2 — handle direction flip (only after baseline is established)
    if old_dir is not None and new_dir != old_dir and new_dir in (1, -1):
        # Close any still-open trade as REVERSED
        open_t = st.session_state.open_trade[sym]
        if open_t is not None and open_t["status"] == "OPEN":
            close_trade_reversed(open_t, spot, now_ist)
            append_trade_csv(open_t)
            st.session_state.open_trade[sym] = None

        # Open a fresh trade
        new_trade = open_trade(sym, new_dir, spot, now_ist)
        if new_trade is not None:
            st.session_state.trades.append(new_trade)
            st.session_state.open_trade[sym] = new_trade

    st.session_state.prev_direction[sym] = new_dir

if not st.session_state.baseline_set and successes == len(SYMBOLS):
    st.session_state.baseline_set = True


# ─── Top: Live spot prices + direction ───────────────────────────────────────
cols = st.columns(len(SYMBOLS))
for i, sym in enumerate(SYMBOLS):
    with cols[i]:
        if sym in errors:
            st.error(f"**{sym}**\n\n❌ {errors[sym]}")
            continue
        s = state[sym]
        st.metric(
            label=f"{sym}  •  spot",
            value=f"${s['spot']:,.2f}",
            delta=f"{DIR_EMOJI[s['direction']]}",
        )
        st.caption(f"Last bar close: ${s['bar_close']:,.2f}  ·  {to_ist_str(s['bar_time'])}")


# ─── Diagnostic: recent flips per symbol (cross-check against TradingView) ───
with st.expander("🔍 Diagnostics — recent flips in loaded history (compare with TradingView)"):
    st.caption(
        f"Loaded **{LOOKBACK_BARS}** bars per symbol. The last 10 direction "
        f"changes the indicator detected are shown below — they should match "
        f"the ChoCh ↑ / ChoCh ↓ markers on your TradingView chart. "
        f"If they don't, increase LOOKBACK_BARS in the code."
    )
    diag_cols = st.columns(len(SYMBOLS))
    for i, sym in enumerate(SYMBOLS):
        with diag_cols[i]:
            st.markdown(f"**{sym}**")
            if sym in errors:
                st.warning(errors[sym])
                continue
            s = state[sym]
            st.caption(f"Bars actually loaded (closed only): {s['bars_loaded']}")
            if not s["flip_history"]:
                st.info("No flips found in loaded history — try increasing LOOKBACK_BARS.")
            else:
                hist_rows = []
                for h in s["flip_history"]:
                    hist_rows.append({
                        "Time (IST)": to_ist_str(h["time"]),
                        "Flipped to": DIR_EMOJI[h["direction"]],
                        "Close":      f"${h['close']:,.2f}",
                    })
                st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)


# ─── Status bar ──────────────────────────────────────────────────────────────
st.markdown(f"🕒 **Last refresh:** {to_ist_str(now_ist)}  •  "
            f"**Total signals tracked:** {len(st.session_state.trades)}")

if not st.session_state.baseline_set:
    st.info("⏳ Capturing baseline directions — flips will be tracked from the next change.")


# ─── Accuracy stats ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🎯 Accuracy")

trades = st.session_state.trades
closed = [t for t in trades if t["status"] in ("WIN", "LOSS")]
wins   = [t for t in closed if t["status"] == "WIN"]
losses = [t for t in closed if t["status"] == "LOSS"]
opens  = [t for t in trades if t["status"] == "OPEN"]
revd   = [t for t in trades if t["status"] == "REVERSED"]

c1, c2, c3, c4, c5 = st.columns(5)
win_rate = (len(wins) / len(closed) * 100) if closed else 0
c1.metric("Win rate (TP vs SL)", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
c2.metric("Wins",     len(wins))
c3.metric("Losses",   len(losses))
c4.metric("Open",     len(opens))
c5.metric("Reversed", len(revd))

# Per-symbol breakdown
st.markdown("**Per-symbol breakdown**")
breakdown = []
for sym in SYMBOLS:
    sym_closed = [t for t in closed if t["symbol"] == sym]
    sym_wins   = [t for t in sym_closed if t["status"] == "WIN"]
    sym_wr     = (len(sym_wins) / len(sym_closed) * 100) if sym_closed else 0
    sym_total_pnl = sum(t["pnl_points"] for t in trades
                        if t["symbol"] == sym and t["pnl_points"] is not None)
    breakdown.append({
        "Symbol":   sym,
        "Wins":     len(sym_wins),
        "Losses":   len(sym_closed) - len(sym_wins),
        "Open":     sum(1 for t in opens if t["symbol"] == sym),
        "Reversed": sum(1 for t in revd if t["symbol"] == sym),
        "Win rate": f"{sym_wr:.1f}%",
        "Net P&L (pts)": round(sym_total_pnl, 2),
    })
st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)


# ─── Trade log ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Trade log")

if not trades:
    st.info("No trades yet. Waiting for the first direction change after baseline…")
else:
    rows = []
    for t in reversed(trades):  # newest first
        rows.append({
            "Symbol":     t["symbol"],
            "Direction":  DIR_NAME[t["direction"]],
            "Entry time": to_ist_str(t["entry_time"]),
            "Entry":      round(t["entry"], 2),
            "Target":     round(t["target"], 2),
            "Stop":       round(t["stop"], 2),
            "Status":     t["status"],
            "Exit time":  to_ist_str(t["exit_time"]) if t["exit_time"] else "—",
            "Exit price": round(t["exit_price"], 2) if t["exit_price"] is not None else "—",
            "P&L (pts)":  round(t["pnl_points"], 2) if t["pnl_points"] is not None else "—",
        })
    log_df = pd.DataFrame(rows)

    # Color status column
    def color_status(val):
        if val == "WIN":      return "background-color: #1f7a3a; color: white"
        if val == "LOSS":     return "background-color: #8a1c1c; color: white"
        if val == "OPEN":     return "background-color: #1f4e8a; color: white"
        if val == "REVERSED": return "background-color: #6e6020; color: white"
        return ""
    styled = log_df.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "💡 The accuracy shown here measures how often the spot price reaches the target "
    "before the stop — assumes ideal entry, no slippage, no fees, no spread. "
    "Treat it as a relative quality measure, not expected return."
)
