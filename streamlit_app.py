# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app: Find US stocks that have fallen by ≥ X% from their 52-week high (default 70%).

주의
- NASDAQ/NYSE 심볼 파일은 data/ 폴더에 두어야 합니다.
  └ data/nasdaqlisted.txt
  └ data/otherlisted.txt
- 가격 데이터는 Yahoo Finance(yfinance)에서 불러옵니다.
- 투자 조언이 아닙니다. 참고용으로만 사용하세요.
"""

import io
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from typing import List

st.set_page_config(
    page_title="US Crash Finder (52W High Drawdown)",
    layout="wide",
)

# -----------------------------
# Utilities
# -----------------------------

def fetch_symbol_directories_from_local() -> tuple[pd.DataFrame, pd.DataFrame]:
    """data/ 폴더에 있는 nasdaqlisted.txt, otherlisted.txt 불러오기"""
    nas = pd.read_csv("data/nasdaqlisted.txt", sep="|")
    oth = pd.read_csv("data/otherlisted.txt", sep="|")

    # Footer 제거
    if nas.columns[-1].startswith("File Creation"):
        nas = nas.iloc[:-1]
    if oth.columns[-1].startswith("File Creation"):
        oth = oth.iloc[:-1]

    nas.columns = [c.strip().lower() for c in nas.columns]
    oth.columns = [c.strip().lower() for c in oth.columns]
    return nas, oth


def _clean_symbol_df(nas: pd.DataFrame, oth: pd.DataFrame,
                     include_exchanges: List[str], exclude_etfs: bool) -> pd.DataFrame:
    """심볼 정리"""
    nas = nas.copy()
    nas['exchange'] = 'NASDAQ'

    exch_map = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca"}
    oth = oth.copy()
    oth['exchange'] = oth['exchange'].map(exch_map).fillna(oth['exchange'])

    # (핵심 수정) 먼저 컬럼명을 바꾼 다음, 바뀐 이름으로 선택합니다.
    nas = nas.rename(columns={'security name': 'security_name'})
    oth = oth.rename(columns={'security name': 'security_name', 'act symbol': 'symbol'})

    # (안전 장치) 파일에 따라 'etf' 컬럼이 없을 수 있으므로, 있는 경우에만 포함합니다.
    nas_cols = ['symbol', 'security_name', 'exchange'] + (['etf'] if 'etf' in nas.columns else [])
    oth_cols = ['symbol', 'security_name', 'exchange'] + (['etf'] if 'etf' in oth.columns else [])

    nas_small = nas[nas_cols]
    oth_small = oth[oth_cols]

    # 합치기
    df = pd.concat([nas_small, oth_small], ignore_index=True)

    # 정리
    df['symbol'] = df['symbol'].astype(str).str.strip()
    df['security_name'] = df['security_name'].astype(str)
    df['exchange'] = df['exchange'].astype(str)

    # 거래소 필터
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
# Sidebar Controls
# -----------------------------
st.sidebar.header("필터")
threshold = st.sidebar.slider("52주 고점 대비 하락률 (이상)", min_value=50, max_value=95, value=70, step=5)
min_price = st.sidebar.number_input("최소 현재가 ($)", value=1.0, step=0.5)
min_avgvol = st.sidebar.number_input("최소 평균 거래량 (60일)", value=50000, step=10000)
limit_scan = st.sidebar.number_input("스캔할 티커 수 제한 (0=무제한)", value=600, step=100)
include_exchanges = st.sidebar.multiselect(
    "거래소 선택", options=["NASDAQ", "NYSE", "NYSE American"],
    default=["NASDAQ", "NYSE"],
)
exclude_etfs = st.sidebar.checkbox("ETF/ETN 제외", value=True)
st.sidebar.markdown("---")
run_scan = st.sidebar.button("스캔 실행 🚀")

st.title("미국 주식 대폭락 탐색기 (52주 고점 대비)")

# -----------------------------
# Main
# -----------------------------
if run_scan:
    with st.spinner("심볼 불러오는 중..."):
        try:
            nas, oth = fetch_symbol_directories_from_local()
        except Exception as e:
            st.error(f"심볼 파일 불러오기 실패: {e}")
            st.stop()
        sym_df = _clean_symbol_df(nas, oth, include_exchanges, exclude_etfs)
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
    df = df[(df['Last'] >= float(min_price))]
    if min_avgvol > 0 and 'AvgVol(60d)' in df.columns:
        df = df[(df['AvgVol(60d)'].fillna(0) >= float(min_avgvol))]
    df = df[(df['Drawdown%'] <= -float(threshold))]
    df = df.sort_values(['Drawdown%']).reset_index(drop=True)

    st.subheader("후보 종목")
    st.write(f"조건에 맞는 종목 수: {len(df):,}")

    if df.empty:
        st.info("조건에 해당하는 종목이 없습니다.")
        st.stop()

    show_cols = ['Ticker', 'Last', '52W High', '52W Low',
                 'Drawdown%', '1Y Return%', 'AvgVol(60d)']
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

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
