import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import numpy as np
import json
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import time

# =============================================================================
# 1. CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(
    page_title="AlphaEdge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REFRESH AUTOMATIQUE (15 min)
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
    .disclaimer-title { 
        color: #EF553B; 
        font-weight: bold; 
        margin-bottom: 10px; 
        font-size: 14px; 
        text-transform: uppercase; 
    }
    .error-box {
        background-color: #2D1B1B;
        border-left: 4px solid #EF553B;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #2D2B1B;
        border-left: 4px solid #FFA726;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CHARGEMENT DES DONNÉES (AVEC VALIDATION)
# =============================================================================

@st.cache_data(ttl=600)
def load_all_data():
    """Charge les CSVs avec validation complète"""
    history_file = 'portfolio_history.csv'
    signals_file = 'latest_signals.csv'
    rebalance_file = 'rebalance_history.csv'
    
    df_hist = pd.DataFrame()
    df_signals = pd.DataFrame()
    df_rebalance = pd.DataFrame()
    errors = []
    
    # 1. Historique Portfolio
    if os.path.exists(history_file):
        try:
            df_hist = pd.read_csv(history_file, index_col=0)
            df_hist.index = pd.to_datetime(df_hist.index, format='mixed', errors='coerce')
            df_hist = df_hist[df_hist.index.notna()]
            df_hist.sort_index(ascending=True, inplace=True)
            
            if not df_hist.empty:
                last_date = df_hist.index[-1]
                days_old = (datetime.now() - last_date).days
                if days_old > 7:
                    errors.append(f"⚠️ Portfolio data is {days_old} days old")
        except Exception as e:
            errors.append(f"❌ Error loading portfolio history: {e}")
    else:
        errors.append(f"❌ File not found: {history_file}")
    
    # 2. Signaux
    if os.path.exists(signals_file):
        try:
            df_signals = pd.read_csv(signals_file)
            required_cols = ['Ticker', 'Signal']
            missing = [c for c in required_cols if c not in df_signals.columns]
            if missing:
                errors.append(f"⚠️ Missing columns in signals: {missing}")
        except Exception as e:
            errors.append(f"❌ Error loading signals: {e}")
    else:
        errors.append(f"❌ File not found: {signals_file}")
    
    # 3. Rebalance history
    if os.path.exists(rebalance_file):
        try:
            df_rebalance = pd.read_csv(rebalance_file, index_col=0)
            df_rebalance.index = pd.to_datetime(df_rebalance.index)
            df_rebalance.sort_index(inplace=True)
        except Exception as e:
            errors.append(f"⚠️ Could not load rebalance history: {e}")
    
    return df_hist, df_signals, df_rebalance, errors

# Chargement
with st.spinner("Loading data..."):
    df_hist, df_signals, df_rebalance, load_errors = load_all_data()

# =============================================================================
# 3. FONCTIONS UTILITAIRES (OPTIMISÉES)
# =============================================================================

def display_kpi_card(label, value, is_percent=True, color_code=False, prefix="", minimal=False):
    """KPI Card avec gestion robuste des valeurs NaN/Inf"""
    if pd.isna(value) or np.isinf(value):
        formatted_val = "N/A"
        html_val = f'<span class="kpi-value">{formatted_val}</span>'
    else:
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
    """Calcul métriques avec validation edge cases"""
    if df.empty or len(df) < 2:
        return 0, 0, 0, 0
    
    try:
        total_ret = (df['Strategy'].iloc[-1] / df['Strategy'].iloc[0]) - 1
        bench_ret = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) - 1
        alpha = total_ret - bench_ret
        
        strategy_returns = df['Strategy'].pct_change().dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() != 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        cum_ret = (1 + strategy_returns).cumprod()
        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        
        return total_ret, alpha, sharpe, max_dd
    except Exception as e:
        return 0, 0, 0, 0

def calculate_period_return(df, days=None, ytd=False, daily=False):
    """Calcul rendements par période avec validation"""
    if df.empty or 'Strategy' not in df.columns or len(df) < 2:
        return 0.0
    
    try:
        if daily and len(df) >= 2:
            return (df['Strategy'].iloc[-1] / df['Strategy'].iloc[-2]) - 1
        
        last_price = df['Strategy'].iloc[-1]
        last_date = df.index[-1]
        
        if ytd:
            target_date = datetime(last_date.year, 1, 1)
        elif days:
            target_date = last_date - timedelta(days=days)
        else:
            target_date = df.index[0]
        
        if target_date < df.index[0]:
            start_price = df['Strategy'].iloc[0]
        else:
            idx = df.index.get_indexer([target_date], method='nearest')[0]
            start_price = df['Strategy'].iloc[idx]
        
        if start_price == 0:
            return 0.0
        
        return ((last_price / start_price) - 1)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_live_ticker_data(ticker, period="1y"):
    """Télécharge données yfinance avec RETRY mechanism"""
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=10)
            
            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = df.columns.str.lower()
            if 'adj close' not in df.columns and 'close' in df.columns:
                df['adj close'] = df['close']
            
            return df
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()

# =============================================================================
# 4. SIDEBAR (AMÉLIORÉE AVEC TICKER HEALTH)
# =============================================================================

st.sidebar.title("AlphaEdge")
st.sidebar.caption("Quantitative Asset Allocation")

if st.sidebar.button("🔄 Force Sync Pipeline"):
    with st.spinner("Clearing cache..."):
        st.cache_data.clear()
        time.sleep(0.5)
    st.success("✅ Cache cleared!", icon="✅")
    st.rerun()

github_url = "https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation"
st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GITHUB-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)]({github_url})")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Dashboard", 
    "Daily Signals", 
    "Data Explorer", 
    "Model Details",
    "Rebalance History"
])
st.sidebar.markdown("---")

# STATUS INDICATOR 
if not df_hist.empty:
    last_dt = df_hist.index[-1]
    date_str = last_dt.date() if isinstance(last_dt, pd.Timestamp) else str(last_dt).split(" ")[0]
    
    days_old = (datetime.now() - last_dt).days
    if days_old == 0:
        status_icon = "🟢"
        status_text = "● System Online"
    elif days_old <= 3:
        status_icon = "🟡"
        status_text = "○ Data Slightly Old"
    else:
        status_icon = "🔴"
        status_text = "○ Data Outdated"
    
    st.sidebar.info(f"Last Update: {date_str}")
    st.sidebar.markdown(f"{status_icon} {status_text}")
else:
    st.sidebar.error("❌ No Data Available")

st.sidebar.markdown("---")

# TICKER HEALTH 
if os.path.exists('ticker_validation.json'):
    with st.sidebar.expander("🔍 Ticker Health", expanded=False):
        try:
            with open('ticker_validation.json', 'r') as f:
                validation = json.load(f)
            
            alerts = validation.get('alerts', {})
            
            if alerts.get('delisted'):
                st.error(f"❌ {len(alerts['delisted'])} delistés")
                st.caption(", ".join(alerts['delisted'][:3]))
            
            if alerts.get('stale'):
                st.warning(f"⚠️ {len(alerts['stale'])} obsolètes")
            
            if alerts.get('warnings'):
                st.info(f"ℹ️ {len(alerts['warnings'])} warnings")
            
            st.metric("Valid Tickers", validation.get('valid_tickers', 0))
        except:
            st.caption("Validation data unavailable")

st.sidebar.markdown("---")

# SYSTEM HEALTH
if os.path.exists('pipeline_health.json'):
    with st.sidebar.expander(" System Health", expanded=False):
        try:
            with open('pipeline_health.json', 'r') as f:
                health = json.load(f)
            
            if isinstance(health, list) and len(health) > 0:
                latest = health[-1]
                st.metric("Pipeline Duration", f"{latest.get('pipeline_duration_seconds', 0):.0f}s")
                st.metric("Data Freshness", f"{latest.get('data_freshness_days', 0)} days")
                st.metric("Selection Rate", f"{latest.get('selection_rate', 0):.1%}")
        except:
            st.caption("Health metrics unavailable")

# DATA ISSUES
if load_errors:
    with st.sidebar.expander("⚠️ Data Issues", expanded=False):
        for err in load_errors:
            st.warning(err, icon="⚠️")

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
        st.subheader("Period Performance")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: display_kpi_card("YTD", calculate_period_return(df_hist, ytd=True), color_code=True, minimal=True)
        with k2: display_kpi_card("6 Months", calculate_period_return(df_hist, days=180), color_code=True, minimal=True)
        with k3: display_kpi_card("3 Months", calculate_period_return(df_hist, days=90), color_code=True, minimal=True)
        with k4: display_kpi_card("1 Month", calculate_period_return(df_hist, days=30), color_code=True, minimal=True)
        with k5: display_kpi_card("Daily Return", calculate_period_return(df_hist, daily=True), color_code=True, minimal=True)
        
        st.markdown("---")
        
        col_title, col_filter = st.columns([2, 1])
        with col_title: 
            st.subheader("Strategy vs Benchmark")
        with col_filter:
            p_sel = st.radio("Zoom:", ["1M", "3M", "6M", "YTD", "1Y", "ALL"], index=5, horizontal=True, label_visibility="collapsed")
        
        df_c = df_hist.copy()
        end = df_c.index[-1]
        
        if p_sel == "1M": start = end - timedelta(days=30)
        elif p_sel == "3M": start = end - timedelta(days=90)
        elif p_sel == "6M": start = end - timedelta(days=180)
        elif p_sel == "YTD": start = datetime(end.year, 1, 1)
        elif p_sel == "1Y": start = end - timedelta(days=365)
        else: start = df_c.index[0]
        
        if start < df_c.index[0]: start = df_c.index[0]
        df_c = df_c[df_c.index >= pd.Timestamp(start)]
        df_base = df_c.apply(lambda x: x / x.iloc[0] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_base.index, y=df_base['Strategy'], 
            mode='lines', name='Hybrid Strategy', 
            line=dict(color='#00CC96', width=2.5),
            fill='tonexty', fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=df_base.index, y=df_base['Benchmark'], 
            mode='lines', name='Benchmark', 
            line=dict(color='#8b92a5', width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            margin=dict(l=0, r=0, t=10, b=0), 
            height=400, 
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            yaxis_title="Indexed Value (Base 100)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        c_risk, c_pie = st.columns([3, 2])
        
        with c_risk:
            st.subheader("⚠️ Risk Analysis")
            s_ret = df_c['Strategy'].pct_change().dropna()
            cum = (1 + s_ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, 
                fill='tozeroy', mode='lines', 
                line=dict(color='#EF553B', width=1.5), 
                name='Drawdown',
                fillcolor='rgba(239, 85, 59, 0.3)'
            ))
            fig_dd.update_layout(
                template="plotly_dark", 
                margin=dict(l=0, r=0, t=10, b=0), 
                height=320, 
                yaxis_tickformat='.1%',
                yaxis_title="Drawdown"
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with c_pie:
            st.subheader("Current Allocation")
            if not df_signals.empty and 'Allocation' in df_signals.columns:
                active = df_signals[df_signals['Allocation'] > 0.001].copy()
                cash = max(0, 1.0 - active['Allocation'].sum())
                
                if cash > 0.001:
                    final = pd.concat([active, pd.DataFrame([{'Ticker': 'CASH', 'Allocation': cash}])], ignore_index=True)
                else:
                    final = active
                
                fig_p = px.pie(
                    final, values='Allocation', names='Ticker', 
                    hole=0.5, 
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_p.update_traces(textposition='outside', textinfo='percent+label')
                fig_p.update_layout(
                    template="plotly_dark", 
                    margin=dict(l=20, r=20, t=0, b=0), 
                    showlegend=False, 
                    height=370
                )
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("⏳ Waiting for signals...")
    else:
        st.warning("⚠️ No data. Run `python daily_run.py`")

# =============================================================================
# PAGE 2 : DAILY SIGNALS
# =============================================================================

elif page == "Daily Signals":
    st.title("📡 Daily Trading Signals")
    
    if not df_signals.empty:
        d = df_signals.copy()
        if 'Allocation' in d.columns:
            d = d.sort_values('Allocation', ascending=False)
        
        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            filter_opt = st.selectbox("Filter", ["All Signals", "BUY Only", "NEUTRAL Only"])
        
        if filter_opt == "BUY Only":
            d = d[d['Signal'] == 'BUY']
        elif filter_opt == "NEUTRAL Only":
            d = d[d['Signal'] == 'NEUTRAL']
        
        
        st.dataframe(
            d, 
            use_container_width=True, 
            height=600, 
            hide_index=True,
            column_config={
                "Allocation": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1),
                "Proba_Hausse": st.column_config.NumberColumn("Probability ↑", format="%.1f%%")
            }
        )
        
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: st.metric("Total Tickers", len(d))
        with col_s2:
            n_buy = len(d[d['Signal'] == 'ACHAT']) if 'Signal' in d.columns else 0
            st.metric("BUY Signals", n_buy)
        with col_s3:
            if 'Allocation' in d.columns:
                st.metric("Total Allocated", f"{d['Allocation'].sum():.1%}")
    else:
        st.info("⏳ No signals. Run pipeline first.")

# =============================================================================
# PAGE 3 : DATA EXPLORER
# =============================================================================

elif page == "Data Explorer":
    st.title("🔎 Market Data Explorer")
    
    default_tickers = ["AI.PA", "AIR.PA", "BNP.PA", "MC.PA", "OR.PA", "TTE.PA"]
    tickers = df_signals['Ticker'].unique().tolist() if not df_signals.empty and 'Ticker' in df_signals.columns else default_tickers
    
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        selected_ticker = st.selectbox("Select Asset", tickers, index=0)
    with col_sel2:
        period_exp = st.selectbox("Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], index=2)
    
    yf_period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
    
    with st.spinner(f"📥 Downloading {selected_ticker}..."):
        df_asset = get_live_ticker_data(selected_ticker, period=yf_period_map[period_exp])
    
    if not df_asset.empty and len(df_asset) > 1:
        try:
            last_close = df_asset['adj close'].iloc[-1]
            prev_close = df_asset['adj close'].iloc[-2]
            daily_var = (last_close / prev_close) - 1
            first_p = df_asset['adj close'].iloc[0]
            total_ret_period = (last_close / first_p) - 1 if first_p != 0 else 0
            ret_series = df_asset['adj close'].pct_change().dropna()
            volatility = ret_series.std() * np.sqrt(252) if not ret_series.empty else 0
        except:
            last_close = daily_var = total_ret_period = volatility = 0
        
        m1, m2, m3, m4 = st.columns(4)
        with m1: display_kpi_card("Last Price", last_close, is_percent=False, prefix="€ ")
        with m2: display_kpi_card("Daily Change", daily_var, color_code=True)
        with m3: display_kpi_card(f"Return ({period_exp})", total_ret_period, color_code=True)
        with m4: display_kpi_card("Ann. Volatility", volatility, is_percent=True)
        
        st.subheader(f"Price Action: {selected_ticker}")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_asset.index, open=df_asset['open'], high=df_asset['high'], low=df_asset['low'], close=df_asset['close'], name='OHLC'), row=1, col=1)
        
        colors = ['#00CC96' if r >= 0 else '#EF553B' for r in df_asset['adj close'].pct_change().fillna(0)]
        fig.add_trace(go.Bar(x=df_asset.index, y=df_asset['volume'], name='Volume', marker_color=colors), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No data for {selected_ticker}")

# =============================================================================
# PAGE 4 : MODEL DETAILS
# =============================================================================

elif page == "Model Details":
    st.title("⚙️ Model Configuration")
    tab1, tab2 = st.tabs(["📊 Performance", "🌐 Clusters"])
    
    with tab1:
        st.markdown("""
        ### Hybrid Strategy Components
        **1. XGBoost**: Predicts 1-month upside probability  
        **2. K-Means**: Market regime detection (RSI-based)  
        **3. Markowitz**: Portfolio optimization (Max Sharpe)
        """)
        st.markdown("---")
        
        try:
            with open("src/models/metrics.json", "r") as f:
                metrics = json.load(f)
            
            st.markdown("### Model Performance (Test Set)")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with col2: st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            with col3: st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            with col4: st.metric("ROC AUC", f"{metrics.get('auc_score', 0):.4f}")
        except:
            st.info("⏳ Metrics not available. Train model first.")
    
    with tab2:
        st.subheader("🌐 Cluster Analysis")
        st.markdown("Segmentation based on RSI to identify momentum vs reversal regimes.")
        
        if not df_signals.empty and 'RSI' in df_signals.columns and 'Return_3M' in df_signals.columns:
            fig = px.scatter(
                df_signals, 
                x="RSI", 
                y="Return_3M", 
                color="Cluster", 
                hover_name="Ticker",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"RSI": "RSI (20)", "Return_3M": "3-Month Momentum"}
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No cluster data available")

# =============================================================================
# PAGE 5 : REBALANCE HISTORY
# =============================================================================

elif page == "Rebalance History":
    st.title(" Monthly Rebalancing History")
    
    if not df_rebalance.empty:
        st.markdown("""
        This page shows the **monthly rebalancing decisions** made by the strategy.
        Each month, the model selects new assets and recalculates optimal weights.
        """)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1: st.metric("Total Rebalances", len(df_rebalance))
        with col_r2:
            avg_stocks = df_rebalance['N_Stocks'].mean() if 'N_Stocks' in df_rebalance.columns else 0
            st.metric("Avg. Stocks/Month", f"{avg_stocks:.1f}")
        with col_r3:
            last_rebal = df_rebalance.index[-1].date() if len(df_rebalance) > 0 else "N/A"
            st.metric("Last Rebalance", str(last_rebal))
        
        st.markdown("---")
        
        if 'N_Stocks' in df_rebalance.columns:
            st.subheader(" Portfolio Size Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_rebalance.index, 
                y=df_rebalance['N_Stocks'], 
                mode='lines+markers', 
                name='N° Stocks', 
                line=dict(color='#00CC96', width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                template="plotly_dark", 
                height=350, 
                yaxis_title="Number of Stocks",
                xaxis_title="Date"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Detailed Rebalancing Log")
        st.dataframe(df_rebalance, use_container_width=True, height=400)
    else:
        st.info("⏳ No rebalance history. Run the corrected pipeline first.")

# =============================================================================
# DISCLAIMER FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">⚠️ AVIS DE NON-RESPONSABILITÉ</div>
    <p>
        Les informations présentées sur ce tableau de bord sont fournies 
        <strong>à titre informatif et éducatif uniquement</strong>. 
        Elles ne constituent en aucun cas un conseil en investissement.
    </p>
    <p>
        <strong>Risques :</strong> Tout investissement comporte des risques. 
        Les performances passées ne garantissent pas les résultats futurs.
    </p>
    <p>
        <strong>Responsabilité :</strong> Consultez un conseiller financier agréé 
        avant toute décision d'investissement.
    </p>
</div>
""", unsafe_allow_html=True)
