# =============================================================================
# STREAMLIT APP - Portfolio Optimization (Hybrid Strategy)
# Filename: app.py
# Repo: SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime
import pytz
import json

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="📊 CAC40 Smart Portfolio",
    page_icon="favicon.jpg", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Hedge Fund" look (Dark Mode Tech)
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
    }
    .stMetric label { color: #aaaaaa; }
    .stMetric value { color: #ffffff; }
    div[data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (ENHANCED - LOCAL + GITHUB FALLBACK)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900)  # 15 min cache
def load_data():
    """
    Loads data with fallback strategy:
    1. Try LOCAL files (if running locally with data/)
    2. Fallback to GITHUB raw URLs (for production/cloud)
    """
    base_url = "https://raw.githubusercontent.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation/main/"
    
    # A. Portfolio History (CSV)
    try:
        # Try local first
        history_df = pd.read_csv("portfolio_history.csv", index_col=0, parse_dates=True)
    except:
        try:
            # Fallback to GitHub
            history_url = base_url + "portfolio_history.csv"
            history_df = pd.read_csv(history_url, index_col=0, parse_dates=True)
        except:
            history_df = pd.DataFrame(columns=['Strategy', 'Benchmark'])
    
    if not history_df.empty:
        history_df.index.name = 'Date'
        history_df = history_df.sort_index()

    # B. Latest Signals (CSV)
    try:
        signals_df = pd.read_csv("latest_signals.csv")
    except:
        try:
            signals_url = base_url + "latest_signals.csv"
            signals_df = pd.read_csv(signals_url)
        except:
            signals_df = pd.DataFrame()

    # C. Metadata (JSON) - NEW
    try:
        with open("data_metadata.json", "r") as f:
            metadata = json.load(f)
    except:
        try:
            metadata_url = base_url + "data_metadata.json"
            metadata = pd.read_json(metadata_url, typ='series').to_dict()
        except:
            metadata = {
                'last_update': datetime.now().isoformat(),
                'n_stocks': 40,
                'n_selected': 0
            }

    # D. Daily Raw Data (Parquet) - NEW
    df_daily_raw = None
    try:
        df_daily_raw = pd.read_parquet("data/raw/cac40_daily_raw.parquet")
    except:
        try:
            daily_url = base_url + "data/raw/cac40_daily_raw.parquet"
            df_daily_raw = pd.read_parquet(daily_url)
        except Exception as e:
            st.sidebar.warning("⚠️ Daily data not available")

    # E. Monthly Features (Parquet) - NEW
    df_monthly_features = None
    try:
        df_monthly_features = pd.read_parquet("data/raw/cac40_monthly_features.parquet")
    except:
        try:
            monthly_url = base_url + "data/raw/cac40_monthly_features.parquet"
            df_monthly_features = pd.read_parquet(monthly_url)
        except Exception as e:
            st.sidebar.warning("⚠️ Monthly features not available")

    return history_df, signals_df, metadata, df_daily_raw, df_monthly_features

# Load all data
history_df, latest_signals, metadata, df_daily_raw, df_monthly_features = load_data()

# Calculate Real-Time Metrics
if not history_df.empty and len(history_df) > 1:
    cum_strat = history_df['Strategy'].iloc[-1] - 100  # Base 100
    cum_bench = history_df['Benchmark'].iloc[-1] - 100
    alpha = cum_strat - cum_bench
    
    peak = history_df['Strategy'].cummax()
    drawdown = (history_df['Strategy'] - peak) / peak
    max_dd = drawdown.min()
    
    daily_ret = history_df['Strategy'].pct_change()
    if daily_ret.std() != 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0
else:
    cum_strat, cum_bench, alpha, max_dd, sharpe = 0, 0, 0, 0, 0
    drawdown = pd.Series(dtype=float)
    daily_ret = pd.Series(dtype=float)

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2621/2621020.png", width=80)

st.sidebar.title("")
st.sidebar.markdown("**CAC40 Quantitative Optimizer**")
st.sidebar.caption("*Powered by XGBoost + K-Means + Markowitz*")

# GitHub Badge
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <a href='https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation' target='_blank' style='text-decoration: none;'>
            <img src='https://img.shields.io/badge/GitHub-SORADATA%2FCAC40-blue?logo=github&style=for-the-badge' />
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard & Performance", 
    "🤖 Daily Signals", 
    "⚙️ Model Details",
    "📈 Data Explorer"  # NEW PAGE
])

st.sidebar.markdown("---")
st.sidebar.success("✅ **System Status**: ONLINE")

# Display Last Update from Metadata
last_update = metadata.get('last_update', 'Unknown')
if isinstance(last_update, str):
    try:
        last_update_dt = pd.to_datetime(last_update)
        last_update_str = last_update_dt.strftime('%d/%m/%Y %H:%M')
    except:
        last_update_str = last_update
else:
    last_update_str = datetime.now().strftime('%d/%m/%Y %H:%M')

st.sidebar.info(f"📅 **Last Update**: {last_update_str}")
st.sidebar.caption(f"📦 **Stocks Analyzed**: {metadata.get('n_stocks', 40)}")
st.sidebar.caption(f"✅ **Selected Today**: {metadata.get('n_selected', 0)}")

# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -----------------------------------------------------------------------------
if page == "📊 Dashboard & Performance":
    st.title("📊 Live Performance")
    
    if history_df.empty or len(history_df) < 2:
        st.info("👋 Welcome! System initialized. History will appear after the next daily update.")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: 
        st.metric("Alpha (vs Bench)", f"{alpha:.1%}", delta=f"{alpha*100:.1f} pts")
    with kpi2: 
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Risk Adj.")
    with kpi3: 
        st.metric(
            label="📉 Max Historical Loss", 
            value=f"{max_dd:.1%}", 
            delta=f"{max_dd:.1%}",
            delta_color="inverse"  # Red when negative (which is bad for drawdown)
        )
    with kpi4: 
        st.metric("Total Return", f"{cum_strat:.1%}", delta="Net of fees")

    st.markdown("---")

    if not history_df.empty:
        st.subheader("📈 Strategy Evolution")
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df['Benchmark'],
            mode='lines', name='Benchmark (CAC40)',
            line=dict(color='gray', width=1.5)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df['Strategy'],
            mode='lines', name='Hybrid Strategy',
            line=dict(color='#2E86AB', width=2.5)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df['Strategy'],
            fill='tonexty', fillcolor='rgba(46, 134, 171, 0.1)',
            line=dict(width=0), showlegend=False
        ))

        fig_perf.update_layout(template="plotly_dark", height=500, legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_perf, use_container_width=True)

        col_dd, col_dist = st.columns(2)
        with col_dd:
            st.subheader("📉 Drawdown Analysis")
            if not drawdown.empty:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown * 100,  # Convert to %
                    fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.5)',
                    line=dict(color='#E74C3C', width=1), name='Drawdown'
                ))
                fig_dd.update_layout(
                    template="plotly_dark", 
                    height=350,
                    yaxis_title="Drawdown (%)"
                )
                st.plotly_chart(fig_dd, use_container_width=True)

        with col_dist:
            st.subheader("📊 Return Distribution")
            if not daily_ret.empty:
                fig_hist = px.histogram(
                    daily_ret.dropna() * 100,  # Convert to %
                    nbins=50, 
                    color_discrete_sequence=['#4ECDC4']
                )
                fig_hist.update_layout(
                    template="plotly_dark", 
                    height=350, 
                    showlegend=False,
                    xaxis_title="Daily Return (%)"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: SIGNALS 
# -----------------------------------------------------------------------------
elif page == "🤖 Daily Signals":
    st.title("🤖 ML Investment Signals")
    st.markdown("Signals generated at **18:00 UTC** via GitHub Actions.")
    
    if latest_signals.empty:
        st.warning("⚠️ No signals available yet (Run the GitHub Action first).")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💼 Investment Watchlist (Active Only)")
            
            # --- INPUT CAPITAL ---
            capital = st.number_input(
                "💰 Capital to Invest (€)", 
                min_value=100, 
                value=1000, 
                step=100,
                help="Enter your total capital to calculate exact share quantities."
            )
            st.markdown("---")

            # --- FILTERING LOGIC ---
            if 'Allocation' in latest_signals.columns:
                active_signals = latest_signals[latest_signals['Allocation'] > 0.001].copy()
            else:
                active_signals = latest_signals[latest_signals['Signal'].isin(['ACHAT', 'BUY'])].copy()

            if active_signals.empty:
                st.info("😴 Strategy is currently **100% CASH**. No stocks to buy today.")
            else:
                # --- FETCH LIVE PRICES ---
                with st.spinner('🔄 Fetching live prices & calculating...'):
                    try:
                        tickers_list = active_signals['Ticker'].tolist()
                        if tickers_list:
                            live_data = yf.download(tickers_list, period="1d", progress=False)
                            
                            if isinstance(live_data, pd.Series):
                                current_prices = {tickers_list[0]: live_data.iloc[-1]}
                            elif not live_data.empty:
                                if 'Close' in live_data.columns:
                                    current_prices = live_data['Close'].iloc[-1].to_dict()
                                else:
                                    current_prices = live_data.iloc[-1].to_dict()
                            else:
                                current_prices = {}
                            
                            active_signals['Last Price'] = active_signals['Ticker'].map(current_prices)
                        else:
                            active_signals['Last Price'] = 0.0
                    except Exception as e:
                        st.error(f"Error fetching prices: {e}")
                        active_signals['Last Price'] = 0.0

                # --- CALCULATIONS ---
                if 'Allocation' in active_signals.columns:
                    active_signals['Invest (€)'] = active_signals['Allocation'] * capital
                    active_signals['Shares (Qté)'] = np.where(
                        active_signals['Last Price'] > 0,
                        np.floor(active_signals['Invest (€)'] / active_signals['Last Price']),
                        0
                    )
                else:
                    active_signals['Invest (€)'] = 0
                    active_signals['Shares (Qté)'] = 0

                # --- DISPLAY ---
                desired_order = ['Ticker', 'Last Price', 'Proba_Hausse', 'Allocation', 'Invest (€)', 'Shares (Qté)']
                final_cols = [c for c in desired_order if c in active_signals.columns]
                active_signals = active_signals[final_cols]

                # Custom Color Function
                def color_proba(val):
                    if val >= 0.70: 
                        return 'background-color: #145A32; color: white; font-weight: bold;'
                    elif val >= 0.60: 
                        return 'background-color: #28B463; color: white;'
                    else: 
                        return 'background-color: #D5F5E3; color: black;'

                # Style Config
                st.dataframe(
                    active_signals.style
                    .format({
                        'Last Price': '{:.2f} €',
                        'Proba_Hausse': '{:.1%}', 
                        'Allocation': '{:.1%}',
                        'Invest (€)': '**{:.2f} €**',
                        'Shares (Qté)': '{:.0f}'
                    }, na_rep="-")
                    .map(color_proba, subset=['Proba_Hausse'])
                    .background_gradient(subset=['Invest (€)'], cmap='BuGn'),
                    
                    use_container_width=True,
                    height=400
                )
            
        with col2:
            st.subheader("🎯 Asset Allocation")
            if not active_signals.empty and 'Allocation' in active_signals.columns:
                fig_pie = px.pie(
                    active_signals, 
                    values='Allocation', 
                    names='Ticker',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Tealgrn
                )
                fig_pie.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                total_invested = active_signals['Invest (€)'].sum()
                st.success(f"**Total Invested:** {total_invested:.2f} €")
                st.caption(f"💵 Remaining Cash: {capital - total_invested:.2f} €")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL DETAILS
# -----------------------------------------------------------------------------
elif page == "⚙️ Model Details":
    st.title("⚙️ Model Architecture")
    
    st.markdown("""
    ### 🧠 "Hybrid" Approach
    This model combines Machine Learning and Quantitative Finance:
    """)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Filtering (XGBoost)**")
        st.markdown("Predicts 1-month upside probability (Target > 0%). Confidence Threshold: **55-60%**.")
    with c2:
        st.warning("**2. Profiling (K-Means)**")
        st.markdown("Clusters stocks by market regime. We target **Cluster 3 (Momentum)**.")
    with c3:
        st.success("**3. Allocation (Markowitz)**")
        st.markdown("Optimizes weights (Max Sharpe Ratio) with diversification constraint (Max 25%).")

    st.markdown("---")
    
    # --- MODEL METRICS SECTION (JSON) ---
    st.subheader("📊 Model Performance (Backtest)")
    st.markdown("Metrics from Cross-Validation (GridSearch) on test data.")

    @st.cache_data(ttl=900)
    def load_metrics():
        base_url = "https://raw.githubusercontent.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation/main/"
        try:
            url = base_url + "src/models/metrics.json"
            metrics = pd.read_json(url, typ='series')
            return metrics
        except:
            try:
                with open("src/models/metrics.json", "r") as f:
                    metrics = json.load(f)
                return pd.Series(metrics)
            except:
                return None

    metrics = load_metrics()

    if metrics is not None:
        k1, k2, k3 = st.columns(3)
        
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('auc_score', 0)
        date_train = metrics.get('training_date', 'Unknown')

        with k1:
            st.metric("Accuracy (Test)", f"{accuracy:.1%}", delta="vs Random (50%)")
        with k2:
            st.metric("ROC AUC Score", f"{auc:.3f}", delta="Discriminant Quality")
        with k3:
            st.metric("Last Training", date_train)
            
        with st.expander("🔍 View Optimal Hyperparameters (Best Params)"):
            st.json(metrics.get('best_params', {}))
            
    else:
        st.warning("⚠️ Model metrics (metrics.json) are not yet available.")

    st.markdown("---")
    st.subheader("📊 Fundamental Cluster Analysis (Simulation)")
    
    # Pedagogical Simulation
    np.random.seed(42)
    n_points = 200
    df_analysis = pd.DataFrame({
        'Cluster': np.random.choice([0, 1, 2, 3], n_points),
        'Ticker': [f'STOCK_{i}' for i in range(n_points)]
    })
    def generate_return(cluster):
        if cluster == 0: return np.random.normal(0.05, 0.1)
        if cluster == 1: return np.random.normal(0.02, 0.15)
        if cluster == 2: return np.random.normal(0.10, 0.2)
        if cluster == 3: return np.random.normal(0.25, 0.25)
        return 0
    df_analysis['return_6m'] = df_analysis['Cluster'].apply(generate_return)
    cluster_map = {0: '0 - Defensive 🛡️', 1: '1 - Value 💰', 2: '2 - Growth 📈', 3: '3 - Momentum 🚀'}
    df_analysis['Label'] = df_analysis['Cluster'].map(cluster_map)

    fig_box = px.box(
        df_analysis, x='Label', y='return_6m', color='Label',
        title="Return Distribution by Cluster (Theoretical)",
        color_discrete_map={
            '0 - Defensive 🛡️': '#95a5a6', 
            '1 - Value 💰': '#3498db', 
            '2 - Growth 📈': '#f1c40f', 
            '3 - Momentum 🚀': '#2ecc71'
        }
    )
    fig_box.update_layout(template="plotly_dark", showlegend=False, height=450)
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: DATA EXPLORER (NEW)
# -----------------------------------------------------------------------------
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("Explore the underlying data powering the models.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Daily Prices", "📅 Monthly Features", "🔍 Stock Details"])
    
    with tab1:
        st.subheader("📊 Daily Historical Data")
        if df_daily_raw is not None and not df_daily_raw.empty:
            st.info(f"**Period**: {df_daily_raw.index.get_level_values('date').min().date()} → {df_daily_raw.index.get_level_values('date').max().date()}")
            
            # Stock Selector
            available_tickers = df_daily_raw.index.get_level_values('ticker').unique().tolist()
            selected_ticker = st.selectbox("Select a stock:", available_tickers, key='daily_ticker')
            
            # Filter data for selected stock
            stock_data = df_daily_raw.xs(selected_ticker, level='ticker')
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Price Chart
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['adj close'],
                    mode='lines',
                    name='Adj Close',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig_price.update_layout(
                    template="plotly_dark",
                    title=f"{selected_ticker} - Price History",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Price (€)"
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                # Latest Stats
                latest = stock_data.iloc[-1]
                st.metric("Latest Close", f"{latest['adj close']:.2f} €")
                st.metric("Volume", f"{latest['volume']:,.0f}")
                if 'rsi' in stock_data.columns:
                    st.metric("RSI", f"{latest['rsi']:.1f}")
                
            # Show raw data
            with st.expander("📋 View Raw Data (Last 100 rows)"):
                st.dataframe(stock_data.tail(100), use_container_width=True)
                
        else:
            st.warning("⚠️ Daily data not available. Run the pipeline to generate it.")
    
    with tab2:
        st.subheader("📅 Monthly Features (ML Input)")
        if df_monthly_features is not None and not df_monthly_features.empty:
            st.info(f"**Period**: {df_monthly_features.index.get_level_values('date').min().date()} → {df_monthly_features.index.get_level_values('date').max().date()}")
            
            # Stock Selector
            available_tickers_monthly = df_monthly_features.index.get_level_values('ticker').unique().tolist()
            selected_ticker_monthly = st.selectbox("Select a stock:", available_tickers_monthly, key='monthly_ticker')
            
            # Filter
            stock_monthly = df_monthly_features.xs(selected_ticker_monthly, level='ticker')
            
            # Feature categories
            momentum_features = [c for c in stock_monthly.columns if 'return_' in c]
            technical_features = ['rsi', 'macd', 'atr', 'bb_low', 'bb_high']
            ff_features = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Momentum Indicators**")
                if momentum_features:
                    fig_momentum = go.Figure()
                    for feat in momentum_features[:6]:  # Limit to 6
                        if feat in stock_monthly.columns:
                            fig_momentum.add_trace(go.Scatter(
                                x=stock_monthly.index,
                                y=stock_monthly[feat],
                                mode='lines',
                                name=feat
                            ))
                    fig_momentum.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_momentum, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Technical Indicators**")
                tech_available = [f for f in technical_features if f in stock_monthly.columns]
                if tech_available:
                    latest_tech = stock_monthly[tech_available].iloc[-1]
                    for feat in tech_available:
                        st.metric(feat.upper(), f"{latest_tech[feat]:.2f}")
            
            # Show raw data
            with st.expander("📋 View Feature Matrix (Last 24 months)"):
                st.dataframe(stock_monthly.tail(24), use_container_width=True)
                
        else:
            st.warning("⚠️ Monthly features not available. Run the pipeline to generate them.")
    
    with tab3:
        st.subheader("🔍 Stock Comparison")
        if df_daily_raw is not None and not df_daily_raw.empty:
            # Multi-select for comparison
            available_tickers_comp = df_daily_raw.index.get_level_values('ticker').unique().tolist()
            selected_tickers_comp = st.multiselect(
                "Select stocks to compare (max 5):",
                available_tickers_comp,
                default=available_tickers_comp[:3],
                max_selections=5
            )
            
            if selected_tickers_comp:
                # Normalized price chart (Base 100)
                fig_comp = go.Figure()
                for ticker in selected_tickers_comp:
                    stock_data = df_daily_raw.xs(ticker, level='ticker')
                    normalized = (stock_data['adj close'] / stock_data['adj close'].iloc[0]) * 100
                    fig_comp.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized,
                        mode='lines',
                        name=ticker
                    ))
                
                fig_comp.update_layout(
                    template="plotly_dark",
                    title="Normalized Performance (Base 100)",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Performance (Base 100)"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Performance Table
                perf_data = []
                for ticker in selected_tickers_comp:
                    stock_data = df_daily_raw.xs(ticker, level='ticker')
                    total_return = (stock_data['adj close'].iloc[-1] / stock_data['adj close'].iloc[0] - 1) * 100
                    volatility = stock_data['adj close'].pct_change().std() * np.sqrt(252) * 100
                    perf_data.append({
                        'Ticker': ticker,
                        'Total Return (%)': total_return,
                        'Volatility (%)': volatility,
                        'Sharpe Ratio': (total_return / volatility) if volatility > 0 else 0
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(
                    perf_df.style.format({
                        'Total Return (%)': '{:.2f}',
                        'Volatility (%)': '{:.2f}',
                        'Sharpe Ratio': '{:.3f}'
                    }).background_gradient(subset=['Total Return (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Data not available for comparison.")

# -----------------------------------------------------------------------------
# DISCLAIMER (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("⚠️ Disclaimer", expanded=False):
    st.caption("""
    **Educational Project - Master 2**
    
    This application is developed for strictly educational and academic research purposes.
    
    Predictions (Signals, Allocations) are generated by AI models (XGBoost, K-Means) and do not constitute financial investment advice.
    
    **Risks:** Investing in the stock market involves risks of capital loss. Past performance is not indicative of future results.
    
    The author declines all responsibility regarding the use of information provided here.
    """)