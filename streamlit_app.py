# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app: Find US stocks that have fallen by ≥ X% from their 52-week high (default 70%).

주의
- NASDAQ/NYSE 심볼 파일은 data/ 폴더에 두어야 합니다.
  └ data/nasdaqlisted.txt
  └ data/otherlisted.txt   (현재 코드는 NASDAQ 파일만 사용)
- 가격 데이터는 Yahoo Finance(yfinance)에서 불러옵니다.
- 투자 조언이 아닙니다. 참고용으로만 사용하세요.
"""

import io
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Any

st.set_page_config(
    page_title="US Crash Finder (52W High Drawdown)",
    layout="wide",
)

# -----------------------------
# Utilities
# -----------------------------

def fetch_nasdaq_symbols_from_local() -> pd.DataFrame:
    """data/ 폴더의 nasdaqlisted.txt만 불러오기 (otherlisted는 무시)"""
    nas = pd.read_csv("data/nasdaqlisted.txt", sep="|")

    # Footer 제거
    if nas.columns[-1].startswith("File Creation"):
        nas = nas.iloc[:-1]

    nas.columns = [c.strip().lower() for c in nas.columns]
    return nas


def _clean_symbol_df(nas: pd.DataFrame,
                     include_exchanges: List[str], exclude_etfs: bool) -> pd.DataFrame:
    """NASDAQ 심볼만 정리 (otherlisted 제거)"""
    nas = nas.copy()
    nas['exchange'] = 'NASDAQ'

    # 컬럼명 정리
    nas = nas.rename(columns={'security name': 'security_name'})

    # 파일에 따라 'etf' 컬럼이 없을 수 있음 → 존재할 때만 포함
    nas_cols = ['symbol', 'security_name', 'exchange'] + (['etf'] if 'etf' in nas.columns else [])
    df = nas[nas_cols]

    # 정리
    df['symbol'] = df['symbol'].astype(str).str.strip()
    df['security_name'] = df['security_name'].astype(str)
    df['exchange'] = df['exchange'].astype(str)

    # 거래소 필터 (NASDAQ만 남아있지만, 사이드바 설정과 일치하도록 필터 유지)
    df = df[df['exchange'].isin(include_exchanges)]

    # ETF 제외
    if exclude_etfs and 'etf' in df.columns:
        df = df[df['etf'] != 'Y']

    # 워런트/권리/우선주 등 제거
    mask_bad = df['security_name'].str.contains(
        r"\b(WARRANT|RIGHTS|UNIT|PFD|PREFERRED|NOTE|BOND|DEPOSITARY|ETF)\b",
        case=False, regex=True,
    )
    df = df[~mask_bad]

    df = df.drop_duplicates('symbol').sort_values('symbol').reset_index(drop=True)
    return df


@st.cache_data(ttl=3*60*60, show_spinner=False)
def download_prices(tickers: List[str], period: str = '1y') -> pd.DataFrame:
    """가격 다운로드"""
    if not tickers:
        return pd.DataFrame()
    chunks = [tickers[i:i+800] for i in range(0, len(tickers), 800)]
    frames = []
    for ch in chunks:
        df = yf.download(ch, period=period, auto_adjust=False,
                         group_by='column', threads=True, progress=False)
        frames.append(df)
    out = None
    for df in frames:
        out = df if out is None else out.join(df, how='outer')
    return out


def compute_drawdowns(px: pd.DataFrame) -> pd.DataFrame:
    """52주 고점 대비 하락률 계산"""
    if px.empty:
        return pd.DataFrame()

    adj = px['Adj Close'] if 'Adj Close' in px.columns.get_level_values(0) else px['Close']
    vol = px['Volume'] if 'Volume' in px.columns.get_level_values(0) else None

    tickers = list(adj.columns)

    rows = []
    for tkr in tickers:
        s = adj[tkr].dropna()
        if s.empty:
            continue
        last = s.iloc[-1]
        high_52w = s.max()
        low_52w = s.min()
        dd = (last / high_52w) - 1.0
        ret_1y = (last / s.iloc[0]) - 1.0
        avg_vol = None
        if vol is not None and tkr in vol.columns:
            vv = vol[tkr].dropna()
            avg_vol = float(vv.tail(60).mean()) if not vv.empty else np.nan
        rows.append({
            'Ticker': tkr,
            'Last': float(last),
            '52W High': float(high_52w),
            '52W Low': float(low_52w),
            'Drawdown%': float(dd * 100.0),
            '1Y Return%': float(ret_1y * 100.0),
            'AvgVol(60d)': avg_vol,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Market Cap
# -----------------------------

def _get_single_market_cap(tkr: str) -> float:
    """개별 티커 시가총액 조회. info → fast_info 순서로 시도."""
    try:
        # 우선 info (일부 케이스에서 더 신뢰도 높음)
        info = yf.Ticker(tkr).info
        mc = info.get("marketCap")
        if isinstance(mc, (int, float)) and mc is not None:
            return float(mc)
    except Exception:
        pass
    try:
        # fast_info fallback
        finfo = yf.Ticker(tkr).fast_info
        mc = getattr(finfo, "market_cap", None) if not isinstance(finfo, dict) else finfo.get("market_cap")
        if isinstance(mc, (int, float)) and mc is not None:
            return float(mc)
    except Exception:
        pass
    return np.nan


@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_market_caps(tickers: List[str]) -> pd.DataFrame:
    """여러 티커의 시가총액을 조회하여 DataFrame으로 반환"""
    rows = []
    for i, tkr in enumerate(tickers, 1):
        mc = _get_single_market_cap(tkr)
        rows.append({"Ticker": tkr, "MarketCap": mc})
        # 과도한 호출을 피하기 위해 아주 짧은 대기 (필요시 조정/제거)
        time.sleep(0.01)
    return pd.DataFrame(rows)


def format_market_cap(x: Any) -> str:
    """시가총액 보기 좋게 포맷"""
    try:
        v = float(x)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    # 단위: K, M, B, T
    units = [("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)]
    for u, base in units:
        if abs(v) >= base:
            return f"{v/base:.2f}{u}"
    return f"{int(v):,}"


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("필터")
threshold = st.sidebar.slider("52주 고점 대비 하락률 (이상)", min_value=50, max_value=95, value=70, step=5)
min_price = st.sidebar.number_input("최소 현재가 ($)", value=1.0, step=0.5)
min_avgvol = st.sidebar.number_input("최소 평균 거래량 (60일)", value=50_000, step=10_000)
min_mcap = st.sidebar.number_input("최소 시가총액 ($)", value=1_000_000_000, step=100_000_000)  # ✅ 추가
limit_scan = st.sidebar.number_input("스캔할 티커 수 제한 (0=무제한)", value=600, step=100)
include_exchanges = st.sidebar.multiselect(
    "거래소 선택", options=["NASDAQ", "NYSE", "NYSE American"],
    default=["NASDAQ", "NYSE"],
)
exclude_etfs = st.sidebar.checkbox("ETF/ETN 제외", value=True)
st.sidebar.markdown("---")
run_scan = st.sidebar.button("스캔 실행 🚀")

st.title("대폭락 미국주식 사냥꾼 (52주 고점 대비)")

# -----------------------------
# Main
# -----------------------------
if run_scan:
    with st.spinner("심볼 불러오는 중..."):
        try:
            nas = fetch_nasdaq_symbols_from_local()
        except Exception as e:
            st.error(f"심볼 파일 불러오기 실패: {e}")
            st.stop()
        sym_df = _clean_symbol_df(nas, include_exchanges, exclude_etfs)
        st.write(f"심볼 수: {len(sym_df):,}")

    tickers = sym_df['symbol'].tolist()
    if limit_scan and limit_scan > 0:
        tickers = tickers[: int(limit_scan)]

    st.write(f"다운로드할 티커 수: {len(tickers):,}")

    with st.spinner("가격 다운로드 중…"):
        prices = download_prices(tickers, period='1y')
        if prices.empty:
            st.error("가격 데이터를 불러오지 못했습니다.")
            st.stop()

    with st.spinner("지표 계산 중…"):
        metrics = compute_drawdowns(prices)
        if metrics.empty:
            st.warning("지표 없음")
            st.stop()

    df = metrics.copy()

    # 1) 기본 가격/거래량 필터
    df = df[(df['Last'] >= float(min_price))]
    if min_avgvol > 0 and 'AvgVol(60d)' in df.columns:
        df = df[(df['AvgVol(60d)'].fillna(0) >= float(min_avgvol))]

    # 2) 드로우다운 필터
    df = df[(df['Drawdown%'] <= -float(threshold))]

    if df.empty:
        st.info("가격/거래량/드로우다운 조건에 해당하는 종목이 없습니다.")
        st.stop()

    # 3) ✅ 시가총액 조회 및 필터
    with st.spinner("시가총액 조회 중…"):
        mcaps = fetch_market_caps(df['Ticker'].tolist())
    df = df.merge(mcaps, on="Ticker", how="left")

    df = df[df["MarketCap"].fillna(0) >= float(min_mcap)]

    if df.empty:
        st.info("시가총액 조건에 해당하는 종목이 없습니다.")
        st.stop()

    # 정렬 및 표시용 포맷
    df = df.sort_values(['Drawdown%']).reset_index(drop=True)

    # 표시 컬럼
    show_cols = ['Ticker', 'Last', '52W High', '52W Low',
                 'Drawdown%', '1Y Return%', 'AvgVol(60d)', 'MarketCap']

    # 보기 좋은 포맷으로 표시 (표시 전용)
    df_display = df[show_cols].copy()
    df_display['MarketCap'] = df_display['MarketCap'].apply(format_market_cap)

    st.subheader("후보 종목")
    st.write(f"조건에 맞는 종목 수: {len(df_display):,}")

    if df_display.empty:
        st.info("조건에 해당하는 종목이 없습니다.")
        st.stop()

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # CSV 다운로드 (원시 숫자 포함)
    csv = df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button("CSV 다운로드", csv, "crash_finder_results.csv", "text/csv")

    st.markdown("---")
    sel = st.selectbox("차트로 볼 티커 선택", df['Ticker'].tolist())
    if sel:
        hist = yf.download(sel, period='1y', auto_adjust=True, progress=False)
        if not hist.empty:
            high_52w = float(hist['Close'].max())
            low_52w = float(hist['Close'].min())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
            fig.add_hline(y=high_52w, line_dash='dot', annotation_text='52W High')
            fig.add_hline(y=low_52w, line_dash='dot', annotation_text='52W Low')
            fig.update_layout(title=f"{sel} — 1Y Price", xaxis_title="Date", yaxis_title="Price ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("왼쪽에서 조건 설정 후 ‘스캔 실행’ 버튼을 눌러주세요.")
