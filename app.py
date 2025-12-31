import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import numpy as np
import json
import yfinance as yf  # NÉCESSAIRE POUR LA VERSION LIVE
from streamlit_autorefresh import st_autorefresh # AJOUT DEMANDÉ

# =============================================================================
# 1. CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(
    page_title="AlphaEdge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REFRESH AUTOMATIQUE (AJOUT DEMANDÉ)
st_autorefresh(interval=900000, key="datarefresh")

# CSS Personnalisé
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .kpi-container {
        background-color: #151922;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #262730;
        text-align: left;
    }
    .kpi-minimal { text-align: left; padding: 10px 0; }
    .kpi-label { font-size: 12px; color: #8b92a5; margin-bottom: 4px; }
    .kpi-value { font-size: 24px; font-weight: 700; color: #ffffff; }
    .kpi-delta-pos { color: #00cc96; font-size: 20px; font-weight: 600; }
    .kpi-delta-neg { color: #ef553b; font-size: 20px; font-weight: 600; }
    .disclaimer-box {
        background-color: #1E1E1E;
        color: #888888;
        padding: 20px;
        border-radius: 5px;
        font-size: 12px;
        border-top: 1px solid #333;
        margin-top: 50px;
        text-align: center;
    }
    .disclaimer-title { color: #EF553B; font-weight: bold; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CHARGEMENT DES DONNÉES (VERSION AVEC TTL DEMANDÉ)
# =============================================================================
@st.cache_data(ttl=600) # MODIF TTL DEMANDÉE
def load_all_data():
    """Charge les CSVs avec une gestion robuste des dates (format mixed)"""
    history_file = 'portfolio_history.csv'
    signals_file = 'latest_signals.csv'
    
    df_hist = pd.DataFrame()
    df_signals = pd.DataFrame()
    
    # 1. Historique Portfolio
    if os.path.exists(history_file):
        try:
            df_hist = pd.read_csv(history_file, index_col=0)
            df_hist.index = pd.to_datetime(df_hist.index, format='mixed', errors='coerce')
            df_hist = df_hist[df_hist.index.notna()]
            df_hist.sort_index(ascending=True, inplace=True)
        except Exception as e:
            st.error(f"Erreur lecture historique: {e}")
    
    # 2. Signaux Récents
    if os.path.exists(signals_file):
        try:
            df_signals = pd.read_csv(signals_file)
        except:
            pass

    return df_hist, df_signals

# Chargement immédiat
df_hist, df_signals = load_all_data()

# =============================================================================
# 3. FONCTIONS UTILITAIRES
# =============================================================================
def display_kpi_card(label, value, is_percent=True, color_code=False, prefix="", minimal=False):
    formatted_val = f"{prefix}{value:.1%}" if is_percent else f"{prefix}{value:.2f}"
    if color_code:
        color_class = "kpi-delta-pos" if value >= 0 else "kpi-delta-neg"
        arrow = "▲" if value >= 0 else "▼"
        html_val = f'<span class="{color_class}">{arrow} {formatted_val}</span>'
    else:
        html_val = f'<span class="kpi-value">{formatted_val}</span>'

    css_class = "kpi-minimal" if minimal else "kpi-container"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="kpi-label">{label}</div>
        {html_val}
    </div>
    """, unsafe_allow_html=True)

def calculate_metrics(df):
    if df.empty: return 0, 0, 0, 0
    total_ret = (df['Strategy'].iloc[-1] / df['Strategy'].iloc[0]) - 1
    bench_ret = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) - 1
    alpha = total_ret - bench_ret
    strategy_returns = df['Strategy'].pct_change().dropna()
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
    cum_ret = (1 + strategy_returns).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    return total_ret, alpha, sharpe, max_dd

def calculate_period_return(df, days=None, ytd=False, daily=False): # MODIF AJOUT DAILY
    if df.empty or 'Strategy' not in df.columns: return 0.0
    
    if daily and len(df) >= 2: # LOGIQUE DAILY DEMANDÉE
        return (df['Strategy'].iloc[-1] / df['Strategy'].iloc[-2]) - 1
        
    last_price = df['Strategy'].iloc[-1]
    last_date = df.index[-1]
    
    if ytd: target_date = datetime(last_date.year, 1, 1)
    elif days: target_date = last_date - timedelta(days=days)
    else: target_date = df.index[0]
        
    if target_date < df.index[0]: start_price = df['Strategy'].iloc[0]
    else:
        try:
            idx = df.index.get_indexer([target_date], method='nearest')[0]
            start_price = df['Strategy'].iloc[idx]
        except: start_price = df['Strategy'].iloc[0]
    
    if start_price == 0: return 0.0
    return ((last_price / start_price) - 1)

@st.cache_data(ttl=3600)
def get_live_ticker_data(ticker, period="1y"):
    """Télécharge les données en direct via yfinance"""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = df.columns.str.lower()
        if 'adj close' not in df.columns and 'close' in df.columns:
            df['adj close'] = df['close']
        return df
    except Exception as e:
        return pd.DataFrame()

# =============================================================================
# 4. NAVIGATION & SIDEBAR
# =============================================================================
st.sidebar.title("AlphaEdge")
st.sidebar.caption("Quantitative Asset Allocation")

# BOUTON SYNC (AJOUT DEMANDÉ)
if st.sidebar.button("🔄 Force Sync Pipeline"):
    st.cache_data.clear()
    st.rerun()

github_url = "https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation"
st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GITHUB-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)]({github_url})")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Daily Signals", "Data Explorer", "Model Details"])
st.sidebar.markdown("---")

if not df_hist.empty:
    last_dt = df_hist.index[-1]
    if isinstance(last_dt, pd.Timestamp):
        date_str = last_dt.date()
    else:
        date_str = str(last_dt).split(" ")[0]
        
    st.sidebar.info(f"Last Update: {date_str}")
    st.sidebar.success("● System Online")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Disclaimer:** Not financial advice.")

# =============================================================================
# PAGE 1 : DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.title(" Portfolio Overview")
    if not df_hist.empty:
        tot_ret, alpha, sharpe, max_dd = calculate_metrics(df_hist)
        c1, c2, c3, c4 = st.columns(4)
        with c1: display_kpi_card("Total Return", tot_ret, color_code=True)
        with c2: display_kpi_card("Alpha vs Bench", alpha, color_code=True)
        with c3: display_kpi_card("Sharpe Ratio", sharpe, is_percent=False)
        with c4: display_kpi_card("Max Drawdown", max_dd, color_code=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(" Period Performance")
        k1, k2, k3, k4, k5 = st.columns(5) # CHANGÉ EN 5 COLONNES
        with k1: display_kpi_card("YTD", calculate_period_return(df_hist, ytd=True), color_code=True, minimal=True)
        with k2: display_kpi_card("6 Months", calculate_period_return(df_hist, days=180), color_code=True, minimal=True)
        with k3: display_kpi_card("3 Months", calculate_period_return(df_hist, days=90), color_code=True, minimal=True)
        with k4: display_kpi_card("1 Month", calculate_period_return(df_hist, days=30), color_code=True, minimal=True)
        with k5: display_kpi_card("Daily Return", calculate_period_return(df_hist, daily=True), color_code=True, minimal=True) # AJOUT DEMANDÉ
        
        st.markdown("---")
        col_title, col_filter = st.columns([2, 1])
        with col_title: st.subheader(" Strategy vs Benchmark")
        with col_filter:
            p_sel = st.radio("Zoom:", ["1M", "3M", "6M", "YTD", "1Y", "ALL"], index=5, horizontal=True, label_visibility="collapsed")
        
        df_c = df_hist.copy()
        end = df_c.index[-1]
        if p_sel=="1M": start = end - timedelta(days=30)
        elif p_sel=="3M": start = end - timedelta(days=90)
        elif p_sel=="6M": start = end - timedelta(days=180)
        elif p_sel=="YTD": start = datetime(end.year, 1, 1)
        elif p_sel=="1Y": start = end - timedelta(days=365)
        else: start = df_c.index[0]
        
        if start < df_c.index[0]: start = df_c.index[0]
        df_c = df_c[df_c.index >= pd.Timestamp(start)]
        df_base = df_c.apply(lambda x: x / x.iloc[0] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['Strategy'], mode='lines', name='Hybrid Strategy', line=dict(color='#00CC96', width=2)))
        fig.add_trace(go.Scatter(x=df_base.index, y=df_base['Benchmark'], mode='lines', name='Benchmark', line=dict(color='#8b92a5', width=1, dash='dash')))
        fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), height=380, hovermode="x unified", legend=dict(orientation="h", y=1, x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        c_risk, c_pie = st.columns([3, 2])
        with c_risk:
            st.subheader(" Risk Analysis")
            s_ret = df_c['Strategy'].pct_change().dropna()
            cum = (1 + s_ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd, fill='tozeroy', mode='lines', line=dict(color='#EF553B', width=1), name='Drawdown'))
            fig_dd.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), height=300, yaxis_tickformat='.1%')
            st.plotly_chart(fig_dd, use_container_width=True)
            
        with c_pie:
            st.subheader(" Current Allocation")
            if not df_signals.empty and 'Allocation' in df_signals.columns:
                active = df_signals[df_signals['Allocation'] > 0.001].copy()
                cash = max(0, 1.0 - active['Allocation'].sum())
                final = pd.concat([active, pd.DataFrame([{'Ticker':'CASH', 'Allocation':cash}])], ignore_index=True) if cash > 0.001 else active
                fig_p = px.pie(final, values='Allocation', names='Ticker', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                fig_p.update_traces(textposition='outside', textinfo='percent+label')
                fig_p.update_layout(template="plotly_dark", margin=dict(l=20,r=20,t=0,b=0), showlegend=False, height=350)
                st.plotly_chart(fig_p, use_container_width=True)
            else: st.info("Waiting for signals...")
    else: st.warning("No data found. Please run the pipeline.")

# =============================================================================
# PAGE 2 : DAILY SIGNALS
# =============================================================================
elif page == "Daily Signals":
    st.title("📡 Daily Trading Signals")
    if not df_signals.empty:
        d = df_signals.copy()
        if 'Allocation' in d.columns: d = d.sort_values('Allocation', ascending=False)
        st.dataframe(d, use_container_width=True, height=600, hide_index=True, 
                    column_config={"Allocation": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1)})
    else: st.info("No signals found.")

# =============================================================================
# PAGE 3 : DATA EXPLORER (Live Version)
# =============================================================================
elif page == "Data Explorer":
    st.title("🔎 Market Data Explorer")
    
    last_close = 0.0
    daily_var = 0.0
    total_ret_period = 0.0
    volatility = 0.0
    
    default_tickers = ["AI.PA", "AIR.PA", "BNP.PA", "MC.PA", "OR.PA", "TTE.PA"]
    if not df_signals.empty and 'Ticker' in df_signals.columns:
        tickers = df_signals['Ticker'].unique().tolist()
    else:
        tickers = default_tickers

    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        selected_ticker = st.selectbox("Select Asset", tickers, index=0)
    with col_sel2:
        period_exp = st.selectbox("Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], index=2)
    
    yf_period_map = {"1 Month":"1mo", "3 Months":"3mo", "6 Months":"6mo", "1 Year":"1y", "5 Years":"5y"}
    
    with st.spinner(f"Downloading data for {selected_ticker}..."):
        df_asset = get_live_ticker_data(selected_ticker, period=yf_period_map[period_exp])
    
    if not df_asset.empty and len(df_asset) > 1:
        try:
            last_close = df_asset['adj close'].iloc[-1]
            prev_close = df_asset['adj close'].iloc[-2]
            daily_var = (last_close / prev_close) - 1
            
            first_p = df_asset['adj close'].iloc[0]
            if first_p != 0:
                total_ret_period = (last_close / first_p) - 1
            
            ret_series = df_asset['adj close'].pct_change().dropna()
            if not ret_series.empty:
                volatility = ret_series.std() * np.sqrt(252)
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")

        m1, m2, m3, m4 = st.columns(4)
        with m1: display_kpi_card("Last Price", last_close, is_percent=False, prefix="€ ")
        with m2: display_kpi_card("Daily Change", daily_var, color_code=True)
        with m3: display_kpi_card(f"Return ({period_exp})", total_ret_period, color_code=True)
        with m4: display_kpi_card("Annualized Volatility", volatility, is_percent=True)
        
        st.subheader(f"Price Action : {selected_ticker}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_asset.index, open=df_asset['open'], high=df_asset['high'],
            low=df_asset['low'], close=df_asset['close'], name='OHLC'), row=1, col=1)
        colors = ['#00CC96' if r >= 0 else '#EF553B' for r in df_asset['adj close'].pct_change().fillna(0)]
        fig.add_trace(go.Bar(x=df_asset.index, y=df_asset['volume'], name='Volume', marker_color=colors), row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No sufficient data found for {selected_ticker} (Period: {period_exp}). Try a longer timeframe.")

# =============================================================================
# PAGE 4 : MODEL DETAILS
# =============================================================================
elif page == "Model Details":
    st.title("⚙️ Model Configuration & Performance")
    tab1, tab2 = st.tabs(["📊 Performance Metrics", "🌐 Cluster Analysis"])
    
    with tab1:
        st.markdown("""
        ###  Hybrid Strategy Components
        **1. Machine Learning Filter (XGBoost)**: Predicts 1-month upside probability based on Momentum, Volatility & Macro factors.
        **2. Clustering (K-Means)**: Segments stocks by market regime (Target: Cluster 3).
        **3. Portfolio Optimization (Markowitz)**: Maximizes Sharpe Ratio with constraints.
        """)
        st.markdown("---")
        try:
            with open("src/models/metrics.json", "r") as f:
                metrics = json.load(f)
            st.markdown("### Model Performance (Test Set)")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Accuracy", f"{metrics.get('accuracy',0):.2%}")
            with col2: st.metric("Precision", f"{metrics.get('precision',0):.2%}")
            with col3: st.metric("Recall", f"{metrics.get('recall',0):.2%}")
            with col4: st.metric("ROC AUC", f"{metrics.get('auc_score',0):.4f}")
        except:
            st.info("Training metrics not available yet.")
            
    with tab2:
        st.subheader("Market Regime Clustering")
        st.markdown("Groups assets based on RSI behavior to identify Momentum vs Reversal regimes.")
        
        if not df_signals.empty and 'RSI' in df_signals.columns and 'Return_3M' in df_signals.columns:
            fig_scatter = px.scatter(
                df_signals, x="RSI", y="Return_3M", color="Cluster", hover_name="Ticker",
                title="Cluster Distribution: RSI vs 3-Month Return", 
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"RSI": "RSI (Strength)", "Return_3M": "Momentum (3M Return)"}
            )
            fig_scatter.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ Data for clustering visualization missing.")

# =============================================================================
# DISCLAIMER FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">⚠️ AVIS DE NON-RESPONSABILITÉ (DISCLAIMER)</div>
    <p>
        Les informations, données et signaux présentés sur ce tableau de bord (AlphaEdge) sont fournis 
        <strong>à titre informatif et éducatif uniquement</strong>. Ils ne constituent en aucun cas un conseil en investissement.
    </p>
</div>
""", unsafe_allow_html=True)