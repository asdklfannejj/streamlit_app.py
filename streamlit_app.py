# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app: Find US stocks that have fallen by ≥ X% from their 52-week high (default 70%).

Features
- Pulls US tickers from NASDAQ Trader symbol directories (NASDAQ + NYSE + NYSE American).
- Filters to common stocks, excludes ETFs/ETNs/Preferreds by default.
- Downloads 1-year daily prices via yfinance in batched calls.
- Computes drawdown from 52-week high to latest close, 1Y return, 52W high/low, avg volume.
- Interactive controls (threshold %, min price/volume, exchanges, exclude ETFs, limit tickers).
- Results table with CSV download; select a row to see a price chart with 52W high/low bands.

Notes
- First run can take several minutes because it downloads thousands of tickers. Use the
  "Limit tickers scanned" option during development.
- Requires: streamlit, pandas, numpy, yfinance, requests, plotly

Run
    pip install streamlit pandas numpy yfinance requests plotly
    streamlit run app.py
"""

import io
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="US Crash Finder (52W High Drawdown)",
    layout="wide",
)

# -----------------------------
# Utilities & Caching
# -----------------------------

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

@st.cache_data(ttl=12*60*60, show_spinner=False)
def fetch_symbol_directories() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download NASDAQ + OTHER listed symbol directories.

    Returns two DataFrames with normalized columns.
    """
    def _download(url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # Files are pipe-delimited with a footer line we must drop
        txt = r.text.strip().splitlines()
        # Remove footer (last line often says File Creation Time: ...)
        if txt and txt[-1].lower().startswith("file creation time"):
            txt = txt[:-1]
        df = pd.read_csv(io.StringIO("\n".join(txt)), sep="|")
        return df

    nas = _download(NASDAQ_LISTED_URL)
    oth = _download(OTHER_LISTED_URL)

    # Normalize column names
    nas.columns = [c.strip().lower() for c in nas.columns]
    oth.columns = [c.strip().lower() for c in oth.columns]
    return nas, oth


def _clean_symbol_df(nas: pd.DataFrame, oth: pd.DataFrame, include_exchanges: List[str], exclude_etfs: bool) -> pd.DataFrame:
    """Return a cleaned list of symbols for equities only.
    include_exchanges: subset of ["NASDAQ", "NYSE", "NYSE American"]
    """
    # NASDAQ file columns: symbol, security name, market category, test issue, financial status, round lot size, etf, nextshares
    nas = nas.copy()
    nas['exchange'] = 'NASDAQ'
    # OTHER file columns: act symbol|security name|exchange|cqs symbol|etf|round lot size|test issue|nasdaq symbol
    oth = oth.copy()
    # Keep NYSE & NYSE American; sometimes listed as "N" (NYSE), "A" (NYSE American), "P" (NYSE Arca)
    # We'll map to friendly names and drop NYSE Arca (mostly ETFs) by default.
    exch_map = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca"}
    oth['exchange'] = oth['exchange'].map(exch_map).fillna(oth['exchange'])

    # Concatenate
    nas_cols = ['symbol', 'security name', 'exchange', 'etf']
    oth_cols = ['act symbol', 'security name', 'exchange', 'etf']
    nas_small = nas.rename(columns={'security name':'security_name'})[nas_cols].rename(columns={'symbol':'symbol'})
    oth_small = oth.rename(columns={'security name':'security_name', 'act symbol':'symbol'})[oth_cols].rename(columns={'act symbol':'symbol'})
    df = pd.concat([nas_small, oth_small], ignore_index=True)

    # Basic cleaning
    df['symbol'] = df['symbol'].str.strip()
    df['security_name'] = df['security_name'].astype(str)
    df['exchange'] = df['exchange'].astype(str)

    # Filter by exchange
    df = df[df['exchange'].isin(include_exchanges)]

    # Exclude ETFs/ETNs
    if exclude_etfs and 'etf' in df.columns:
        df = df[df['etf'] != 'Y']

    # Exclude test issues, warrants, units, rights, preferreds (simple heuristic by name)
    mask_bad = df['security_name'].str.contains(
        r"\b(WARRANT|RIGHTS|UNIT|PFD|PREFERRED|NOTE|BOND|DEPOSITARY|ETF)\b",
        case=False,
        regex=True,
    )
    df = df[~mask_bad]

    # Deduplicate
    df = df.drop_duplicates('symbol')
    df = df.sort_values('symbol').reset_index(drop=True)
    return df


@st.cache_data(ttl=3*60*60, show_spinner=False)
def download_prices(tickers: List[str], period: str = '1y') -> pd.DataFrame:
    """Batch-download OHLCV and Adj Close for given tickers.

    Returns a multi-index DataFrame with columns like ('Adj Close', 'AAPL'), etc.
    """
    if not tickers:
        return pd.DataFrame()
    # yfinance can handle up to ~800-1000 tickers per batch reliably; we'll split.
    chunks = [tickers[i:i+800] for i in range(0, len(tickers), 800)]
    frames = []
    for ch in chunks:
        df = yf.download(ch, period=period, auto_adjust=False, group_by='column', threads=True, progress=False)
        frames.append(df)
    # Align on date index
    out = None
    for df in frames:
        out = df if out is None else out.join(df, how='outer')
    return out


def compute_drawdowns(px: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per ticker: current close, 52w high/low, drawdown from 52w high, 1Y return, avg volume."""
    if px.empty:
        return pd.DataFrame()

    # If DataFrame is multi-indexed by columns (field, ticker)
    # Normalize to a dict of per-ticker frames
    def _get(col):
        # Works for both single and multi-level cols
        if isinstance(px.columns, pd.MultiIndex):
            # e.g., ('Adj Close','AAPL') exists
            return px['Adj Close'] if 'Adj Close' in px.columns.get_level_values(0) else px['Close']
        else:
            return px

    adj = _get('Adj Close')
    close = _get('Close')
    vol = _get('Volume') if isinstance(px.columns, pd.MultiIndex) else None

    tickers = list(adj.columns if isinstance(adj, pd.DataFrame) else [adj.name])

    rows = []
    for tkr in tickers:
        try:
            s = adj[tkr].dropna()
        except Exception:
            continue
        if s.empty:
            continue
        last = s.iloc[-1]
        high_52w = s.max()
        low_52w = s.min()
        dd = (last / high_52w) - 1.0  # negative number
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

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("필터")
threshold = st.sidebar.slider("52주 고점 대비 하락률 (이상)", min_value=50, max_value=95, value=70, step=5)
min_price = st.sidebar.number_input("최소 현재가 ($)", value=1.0, step=0.5)
min_avgvol = st.sidebar.number_input("최소 평균 거래량 (60일)", value=50000, step=10000)
limit_scan = st.sidebar.number_input("스캔할 티커 수 제한 (0=무제한)", value=600, step=100)
include_exchanges = st.sidebar.multiselect(
    "거래소 선택",
    options=["NASDAQ", "NYSE", "NYSE American"],
    default=["NASDAQ", "NYSE"],
)
exclude_etfs = st.sidebar.checkbox("ETF/ETN 제외", value=True)

st.sidebar.markdown("---")
run_scan = st.sidebar.button("스캔 실행 🚀")

st.title("미국 주식 대폭락 탐색기 (52주 고점 대비)")
st.caption("최근 1년(52주) 동안 고점 대비 크게 하락(예: 70% 이상)한 종목을 찾습니다. 투자 조언이 아니며, 데이터 제공에 지연/오류가 있을 수 있습니다.")

# -----------------------------
# Main logic
# -----------------------------

if run_scan:
    with st.spinner("심볼 목록 불러오는 중..."):
        try:
            nas, oth = fetch_symbol_directories()
        except Exception as e:
            st.error(f"심볼 디렉토리 다운로드 실패: {e}")
            st.stop()
        sym_df = _clean_symbol_df(nas, oth, include_exchanges, exclude_etfs)
        st.write(f"심볼 수: {len(sym_df):,}")

    # Optionally limit number of tickers (for speed during dev)
    tickers = sym_df['symbol'].tolist()
    if limit_scan and limit_scan > 0:
        tickers = tickers[: int(limit_scan)]

    st.write(f"다운로드할 티커 수: {len(tickers):,}")

    # Download prices
    with st.spinner("가격 데이터 다운로드 중 (1년)…"):
        prices = download_prices(tickers, period='1y')
        if prices is None or prices.empty:
            st.error("가격 데이터를 불러오지 못했습니다.")
            st.stop()

    # Compute metrics
    with st.spinner("지표 계산 중…"):
        metrics = compute_drawdowns(prices)
        if metrics.empty:
            st.warning("계산된 지표가 없습니다.")
            st.stop()

    # Apply filters
    df = metrics.copy()
    df = df[(df['Last'] >= float(min_price))]
    if min_avgvol > 0 and 'AvgVol(60d)' in df.columns:
        df = df[(df['AvgVol(60d)'].fillna(0) >= float(min_avgvol))]
    df = df[(df['Drawdown%'] <= -float(threshold))]  # e.g., -70% or worse

    # Sort by worst drawdown
    df = df.sort_values(['Drawdown%']).reset_index(drop=True)

    st.subheader("후보 종목")
    st.write(f"조건에 맞는 종목 수: {len(df):,}")

    if df.empty:
        st.info("조건에 해당하는 종목이 없습니다. 필터를 조정해보세요.")
        st.stop()

    # Display table
    show_cols = ['Ticker', 'Last', '52W High', '52W Low', 'Drawdown%', '1Y Return%', 'AvgVol(60d)']
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    # Download CSV
    csv = df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button("CSV 다운로드", csv, file_name="crash_finder_results.csv", mime="text/csv")

    # Chart for selection
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
            fig.update_layout(title=f"{sel} — 1Y Price with 52W High/Low", xaxis_title="Date", yaxis_title="Price ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("차트 데이터를 불러오지 못했습니다.")

else:
    st.info("왼쪽 사이드바에서 조건을 설정하고 ‘스캔 실행’ 버튼을 눌러 시작하세요.")

# -----------------------------
# Tips & Disclaimers
# -----------------------------
with st.expander("고급 설정/팁"):
    st.markdown(
        """
        - **스캔 속도 향상**: 개발 단계에선 `스캔할 티커 수 제한`을 낮춰 테스트하세요. 서버 성능과 네트워크 상태에 따라 다운로드 시간이 달라집니다.
        - **저유동성/펌프-앤-덤프 회피**: `최소 현재가`, `최소 평균 거래량`을 적절히 설정해 페니주식을 걸러내세요.
        - **정의 차이**: 여기서는 **52주 고점 대비 현재가 하락률**로 판단합니다. 필요하면 1년 전 대비 수익률(1Y Return)도 함께 확인하세요.
        - **데이터 출처**: 심볼은 NASDAQ Trader, 시세는 Yahoo Finance(yfinance)를 사용합니다. 상장폐지/심볼변경 이슈로 일부 누락이나 오류가 있을 수 있습니다.
        - **확장 아이디어**:
            1) 시가총액/섹터 기반 필터 추가 (별도 재무 데이터 소스 필요)
            2) 최대 낙폭(MDD) 타임스탬프 및 저점부터 반등률 계산
            3) 알림 기능: 조건 충족 시 Slack/이메일 Webhook
            4) 백테스트: 극단적 낙폭 이후 수익률 통계
        """
    )
