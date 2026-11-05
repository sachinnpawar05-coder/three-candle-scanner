"""
3 Candle Top Down - Live Scanner (Streamlit)
Run:
    pip install streamlit pandas numpy yfinance plotly
    streamlit run three_candle_scanner_app.py

Same rules as the backtest script:
 1. Day  : 1H candle sweeps PDH/PDL and closes back inside -> bias
 2. 1H   : rejection wick on the sweep side
 3. 5m   : push (displacement) candle -> order block (OB) -> limit entry on OB retest
 4. Exit : SL beyond OB, T1 = push extreme, T2 = 1H sweep-candle extreme
Data: yfinance (delayed ~1-15 min). Educational use only, not investment advice.
"""
import time
from datetime import time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3 Candle Top Down Scanner", layout="wide")


@st.cache_data(ttl=60)
def load(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df[["open", "high", "low", "close"]].dropna().sort_index()


def scan(df, p):
    O, H, L, C = (df[c].values for c in ["open", "high", "low", "close"])
    idx = df.index
    dates = np.array(idx.date)
    days = sorted(set(dates))
    if len(days) < 2:
        return None
    d, prev = days[-1], days[-2]
    pdh, pdl = H[dates == prev].max(), L[dates == prev].min()
    pos = np.where(dates == d)[0]
    ds, de = pos[0], pos[-1]
    last_ts = idx[de]
    avgb = pd.Series(np.abs(C - O)).rolling(20).mean().shift(1).values
    h1 = df[dates == d].resample("60min", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

    res = dict(date=d, pdh=pdh, pdl=pdl, bias=0, status="Abhi koi PDH/PDL sweep nahi hua",
               sweep_time=None, push_time=None, level=None, sl=None, t1=None, t2=None,
               entry=None, de=de, ds=ds)

    for ts, h in h1.iterrows():
        t_close = ts + pd.Timedelta(minutes=60)
        if t_close > last_ts + pd.Timedelta(minutes=5):
            continue  # 1H candle abhi complete nahi hui
        rng = h.high - h.low
        if rng <= 0:
            continue
        bias = 0
        if h.high > pdh and h.close < pdh:
            bias = -1
        elif h.low < pdl and h.close > pdl:
            bias = 1
        if bias == 0:
            continue
        upper = (h.high - max(h.open, h.close)) / rng
        lower = (min(h.open, h.close) - h.low) / rng
        if (lower if bias == 1 else upper) < p["wick"]:
            continue

        res.update(bias=bias, sweep_time=ts, status="Bias set - push candle ka wait")
        t_end = min(t_close + pd.Timedelta(hours=p["window"]),
                    pd.Timestamp.combine(d, dtime(14, 30)))
        wpos = np.where((idx >= t_close) & (idx <= t_end))[0]
        if len(wpos) == 0:
            break

        for j in range(wpos[0], wpos[-1] + 1):
            if j - 6 < ds or np.isnan(avgb[j]):
                continue
            if abs(C[j] - O[j]) < p["disp"] * avgb[j]:
                continue
            if bias == 1:
                if not (C[j] > O[j] and C[j] > H[j - 3:j].max()):
                    continue
                obs = [m for m in range(j - 1, j - 7, -1) if C[m] < O[m]]
            else:
                if not (C[j] < O[j] and C[j] < L[j - 3:j].min()):
                    continue
                obs = [m for m in range(j - 1, j - 7, -1) if C[m] > O[m]]
            if not obs:
                continue
            m = obs[0]
            level = H[m] if bias == 1 else L[m]
            sl = L[m] if bias == 1 else H[m]
            t1 = H[m:j + 1].max() if bias == 1 else L[m:j + 1].min()
            t2 = h.high if bias == 1 else h.low
            risk_est = abs(level - sl)
            if (t2 - t1) * bias <= 0:
                t2 = t1 + bias * risk_est
            res.update(push_time=idx[j], level=level, sl=sl, t1=t1, t2=t2,
                       status=f"Waiting for retest - limit @ {level:.1f}")

            touched_at = None
            for e in range(j + 1, min(j + 1 + p["retest"], de + 1)):
                if (L[e] <= level) if bias == 1 else (H[e] >= level):
                    touched_at = e
                    break
            if touched_at is None:
                if de - j >= p["retest"]:
                    res["status"] = "Expired - retest nahi aaya"
                return res

            e = touched_at
            entry = min(O[e], level) if bias == 1 else max(O[e], level)
            res.update(entry=entry, status="TRIGGERED - trade chal raha hai")
            cur_sl, t1hit = sl, False
            for k in range(e, de + 1):
                adv = L[k] if bias == 1 else H[k]
                fav = H[k] if bias == 1 else L[k]
                if bias * (adv - cur_sl) <= 0:
                    res["status"] = "SL hit" if not t1hit else "Breakeven stop (T1 ke baad)"
                    break
                if not t1hit and bias * (fav - t1) >= 0:
                    t1hit, cur_sl = True, entry
                    res["status"] = "T1 hit - SL cost par"
                if t1hit and bias * (fav - t2) >= 0:
                    res["status"] = "T2 hit"
                    break
            return res
        break
    return res


# ------------------------------ UI ------------------------------
st.title("3 Candle Top Down Scanner")
with st.sidebar:
    symbol = st.selectbox("Symbol", ["^NSEI", "^NSEBANK", "^BSESN"])
    disp = st.slider("Push candle body x avg", 1.0, 3.0, 1.5, 0.1)
    wick = st.slider("Min rejection wick (1H)", 0.1, 0.8, 0.35, 0.05)
    retest = st.slider("Retest bars (5m)", 3, 30, 12)
    window = st.slider("Entry window after 1H close (hrs)", 1, 5, 3)
    auto = st.checkbox("Auto refresh (60 sec)", value=False)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

try:
    df = load(symbol)
except Exception as ex:
    st.error(f"Data nahi aaya: {ex}")
    st.stop()

if df.empty:
    st.warning("Data khaali hai (market band ho sakta hai).")
    st.stop()

p = dict(disp=disp, wick=wick, retest=retest, window=window)
r = scan(df, p)
if r is None:
    st.warning("Kam se kam 2 trading din ka data chahiye.")
    st.stop()

side = {1: "LONG", -1: "SHORT", 0: "-"}[r["bias"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("PDH", f"{r['pdh']:.1f}")
c2.metric("PDL", f"{r['pdl']:.1f}")
c3.metric("Bias", side)
c4.metric("LTP", f"{df.close.iloc[-1]:.1f}")

if r["status"].startswith("TRIGGERED"):
    st.success(f"{side}: {r['status']}")
elif r["status"].startswith("Waiting"):
    st.warning(f"{side}: {r['status']}")
else:
    st.info(r["status"])

if r["level"] is not None:
    rows = {
        "Entry (OB retest)": r["entry"] if r["entry"] else r["level"],
        "Stop Loss": r["sl"],
        "T1 (push high/low)": r["t1"],
        "T2 (1H sweep candle)": r["t2"],
    }
    st.table(pd.DataFrame({"Level": {k: round(float(v), 1) for k, v in rows.items()}}))

# chart: last 2 sessions
show = df.iloc[max(0, r["ds"] - 80):]
fig = go.Figure(go.Candlestick(x=show.index, open=show.open, high=show.high,
                               low=show.low, close=show.close, name=symbol))
fig.add_hline(y=r["pdh"], line_dash="dot", line_color="orange", annotation_text="PDH")
fig.add_hline(y=r["pdl"], line_dash="dot", line_color="orange", annotation_text="PDL")
if r["level"] is not None:
    fig.add_hline(y=r["level"], line_color="blue", annotation_text="Entry")
    fig.add_hline(y=r["sl"], line_color="red", annotation_text="SL")
    fig.add_hline(y=r["t1"], line_color="green", annotation_text="T1")
    fig.add_hline(y=r["t2"], line_color="green", line_dash="dash", annotation_text="T2")
fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
st.plotly_chart(fig, use_container_width=True)

st.caption("Educational use only, investment advice nahi hai. yfinance data delayed ho sakta hai, "
           "isliye live paise lagane se pehle backtest aur paper trade zaroor karo.")

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
