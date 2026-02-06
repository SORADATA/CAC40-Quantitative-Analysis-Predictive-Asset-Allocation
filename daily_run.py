import os
import pickle
import warnings
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# --- DATA SCIENCE STACK ---
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# --- TECHNICAL ANALYSIS ---
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

# --- PORTFOLIO OPTIMIZATION ---
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# =============================================================================
# 1. SYSTEM CONFIGURATION & PATH SETUP
# =============================================================================

# Auto-detect project root by looking for 'app.py'
current_path = Path(os.getcwd())
project_root = current_path
while not (project_root / 'app.py').exists():
    if project_root == project_root.parent:
        project_root = Path(os.getcwd())
        break
    project_root = project_root.parent

# Define standard paths
MODEL_DIR = project_root / "src" / "models"
DATA_DIR = project_root / "data" / "raw"
CONFIG_FILE = project_root / "config" / "market_config.json"
LOG_DIR = project_root / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. MARKET CONFIGURATION LOADER (DYNAMIC)
# =============================================================================

def load_market_config():
    """
    Loads market configuration from JSON file.
    Allows switching between CAC40, S&P500, Crypto, etc. without changing code.
    """
    default_config = {
        "market_name": "Default (CAC40)",
        "benchmark_ticker": "^FCHI",
        "assets": ["AI.PA", "AIR.PA", "SAN.PA", "MC.PA"] # Fallback
    }

    if CONFIG_FILE.exists():
        print(f"⚙️  Loading configuration from: {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading config file: {e}. Using default fallback.")
            return default_config
    else:
        print(f"⚠️  Config file not found at {CONFIG_FILE}. Using default fallback.")
        return default_config

# Load Global Variables
market_config = load_market_config()
TICKERS = market_config.get('assets', [])
BENCHMARK_TICKER = market_config.get('benchmark_ticker', '^FCHI')
MARKET_NAME = market_config.get('market_name', 'Unknown Market')

print(f"📋 Market: {MARKET_NAME} | Assets: {len(TICKERS)} | Benchmark: {BENCHMARK_TICKER}")

# =============================================================================
# 3. TICKER MANAGEMENT & VALIDATION
# =============================================================================

def handle_ticker_changes():
    """
    Handles known ticker changes (mergers, renames).
    Update this dictionary based on corporate actions in your specific market.
    """
    TICKER_CHANGES = {
        # 'OLD_TICKER': 'NEW_TICKER'
    }
    
    DELISTED_TICKERS = [
        # 'BANKRUPT_CO.PA'
    ]
    
    return TICKER_CHANGES, DELISTED_TICKERS

def validate_and_clean_tickers(df, tickers_list, max_days_stale=30):
    """
    Validates data quality: checks for missing tickers, stale data, and liquidity issues.
    Returns cleaned dataframe and list of valid tickers.
    """
    print("🔍 Validating ticker data integrity...")
    
    alerts = {
        'delisted': [],
        'stale': [],
        'missing': [],
        'warnings': []
    }
    
    # 1. Check for missing tickers
    tickers_in_data = df.index.get_level_values('ticker').unique().tolist()
    missing_tickers = set(tickers_list) - set(tickers_in_data)
    
    if missing_tickers:
        alerts['missing'] = list(missing_tickers)
        print(f"   ⚠️ {len(missing_tickers)} tickers missing from download.")
    
    # 2. Check for stale data (no updates in X days)
    last_date = df.index.get_level_values('date').max()
    stale_tickers = []
    
    for ticker in tickers_in_data:
        ticker_data = df.xs(ticker, level='ticker')
        ticker_last_date = ticker_data.index.max()
        days_stale = (last_date - ticker_last_date).days
        
        if days_stale > max_days_stale:
            stale_tickers.append({
                'ticker': ticker,
                'days_stale': days_stale,
                'last_date': ticker_last_date.date()
            })
    
    if stale_tickers:
        alerts['stale'] = stale_tickers
        print(f"   ⚠️ {len(stale_tickers)} stale tickers detected (>{max_days_stale} days).")
        
        # Remove stale tickers
        tickers_to_remove = [t['ticker'] for t in stale_tickers]
        df = df[~df.index.get_level_values('ticker').isin(tickers_to_remove)]
        alerts['delisted'] = tickers_to_remove
        print(f"   ✅ Removed {len(tickers_to_remove)} stale tickers from dataset.")
    
    # 3. Liquidity Check (Low Volume)
    if 'volume' in df.columns:
        for ticker in df.index.get_level_values('ticker').unique():
            ticker_data = df.xs(ticker, level='ticker')
            # Check average volume of last 20 records
            recent_volume = ticker_data['volume'].tail(20).mean()
            
            if recent_volume < 1000: 
                alerts['warnings'].append(f"{ticker}: Low liquidity ({recent_volume:.0f} avg vol)")
    
    valid_tickers = df.index.get_level_values('ticker').unique().tolist()
    print(f"   ✅ Valid Tickers: {len(valid_tickers)} / {len(tickers_list)}")
    
    return df, valid_tickers, alerts

# =============================================================================
# 4. ETL & FEATURE ENGINEERING PIPELINE
# =============================================================================

def load_models():
    """Loads pre-trained XGBoost and KMeans models."""
    print(f"📂 Loading ML models from {MODEL_DIR}...")
    try:
        with open(MODEL_DIR / 'xgboost_model.pkl', 'rb') as f:
            xgb = pickle.load(f)
        with open(MODEL_DIR / 'kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        return xgb, kmeans
    except FileNotFoundError:
        print("❌ Error: Models not found. Please run training notebook first.")
        return None, None

def compute_technical_indicators(df):
    """Computes technical indicators (RSI, Bollinger, MACD, ATR, Volatility)."""
    print("📊 Computing Technical Indicators...")

    # 1. Garman-Klass Volatility (more efficient than Close-to-Close)
    df['garman_klass_vol'] = (
        (np.log(df['high']) - np.log(df['low']))**2 / 2 - 
        (2*np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open']))**2
    )

    # 2. RSI (Relative Strength Index)
    # Applied per ticker group
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = df.loc[idx, 'adj close']
        if len(close_series) > 20:
            rsi_indicator = RSIIndicator(close=close_series, window=20)
            df.loc[idx, 'rsi'] = rsi_indicator.rsi().values

            # Bollinger Bands
            close_log = np.log1p(close_series)
            bb = BollingerBands(close=close_log, window=20, window_dev=2)
            df.loc[idx, 'bb_low'] = bb.bollinger_lband().values
            df.loc[idx, 'bb_mid'] = bb.bollinger_mavg().values
            df.loc[idx, 'bb_high'] = bb.bollinger_hband().values

    # 3. ATR (Average True Range)
    def compute_atr(stock_data):
        if len(stock_data) < 15: return pd.Series(np.nan, index=stock_data.index)
        atr_ind = AverageTrueRange(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], window=14)
        atr = atr_ind.average_true_range()
        return atr.sub(atr.mean()).div(atr.std()) # Normalize

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    # 4. MACD
    def compute_macd(stock_data):
        if len(stock_data) < 26: return pd.Series(np.nan, index=stock_data.index)
        macd_ind = MACD(close=stock_data['adj close'], window_slow=26, window_fast=12, window_sign=9)
        macd_val = macd_ind.macd()
        return macd_val.sub(macd_val.mean()).div(macd_val.std()) # Normalize

    df['macd'] = df.groupby(level=1, group_keys=False).apply(compute_macd)

    # 5. Euro/Dollar Volume
    df['euro_volume'] = (df['adj close'] * df['volume']) / 1e6

    return df

def calculate_returns(df):
    """Calculates momentum returns for various lookback periods."""
    print("📈 Computing Momentum Returns...")
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    min_periods = 12

    for lag in lags:
        returns_raw = df['adj close'].pct_change(lag)
        # Winsorize outliers to prevent model instability
        lower_bound = returns_raw.expanding(min_periods=min_periods).quantile(outlier_cutoff)
        upper_bound = returns_raw.expanding(min_periods=min_periods).quantile(1 - outlier_cutoff)
        df[f'return_{lag}m'] = returns_raw.clip(lower=lower_bound, upper=upper_bound)

    return df

def get_fama_french_betas(data):
    """Retrieves Fama-French factors and computes rolling betas."""
    print("🌍 Retrieving Fama-French Factors (Europe 5 Factors)...")
    try:
        # Note: This pulls Europe data. If switching to US markets, change to 'F-F_Research_Data_5_Factors_2x3'
        factor_data = web.DataReader('Europe_5_Factors', 'famafrench', start='2010')[0].drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample('BM').last().div(100)
        factor_data.index.name = 'date'

        data_ff = data.copy()
        if 'return_1m' not in data_ff.columns: return data

        betas_list = []
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        for ticker in data_ff.index.get_level_values(1).unique():
            try:
                y = data_ff.xs(ticker, level=1)['return_1m']
                X = factor_data.loc[factor_data.index.intersection(y.index)]
                y = y.loc[X.index]

                if len(y) > 24:
                    exog = sm.add_constant(X[factors])
                    rols = RollingOLS(y, exog, window=24)
                    rres = rols.fit()
                    params = rres.params.drop('const', axis=1)
                    params['ticker'] = ticker
                    betas_list.append(params)
            except Exception:
                continue

        if not betas_list: return data

        betas_df = pd.concat(betas_list).set_index('ticker', append=True)
        data = data.join(betas_df.groupby('ticker').shift()) # Shift to avoid lookahead bias
        
        # Fill missing betas with mean
        data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

        return data

    except Exception as e:
        print(f"⚠️ Warning: Fama-French retrieval failed ({e}). Filling with zeros.")
        for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            data[f] = 0.0
        return data

def get_data_pipeline():
    """Executes the full ETL pipeline: Download, Clean, Feature Engineering."""
    
    # 1. Handle ticker changes
    ticker_changes, delisted = handle_ticker_changes()
    active_tickers = [t for t in TICKERS if t not in delisted]
    
    # Apply renames
    for old, new in ticker_changes.items():
        if old in active_tickers:
            active_tickers[active_tickers.index(old)] = new
    
    print(f"📋 Active Tickers for Download: {len(active_tickers)}")
    
    # 2. Download Data (Yahoo Finance)
    end_date = (datetime.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(datetime.today()) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    
    print(f"⬇️ Downloading Market Data ({start_date} -> {end_date})...")
    
    MAX_RETRIES = 3
    df = None
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                active_tickers, start=start_date, end=end_date,
                progress=False, auto_adjust=False, threads=True
            )
            if not df.empty:
                print(f"   ✅ Download successful (Attempt {attempt + 1})")
                break
            else:
                print(f"   ⚠️ Empty data received (Attempt {attempt + 1})")
        except Exception as e:
            print(f"   ❌ Download error (Attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print("   ❌ FATAL: Failed to download data after retries.")
                return None, None

    if df is None or df.empty: return None, None

    # 3. Structure Data
    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    
    # Normalize column names
    if 'adj close' not in df.columns and 'close' in df.columns:
        df['adj close'] = df['close']

    # 4. Validate & Clean
    df, valid_tickers, alerts = validate_and_clean_tickers(df, active_tickers)
    
    # Save validation report (CORRECTED)
    with open(project_root / 'ticker_validation.json', 'w') as f:
        json.dump({
            'date': str(datetime.now()), 
            'alerts': alerts,
            'valid_tickers': len(valid_tickers) # <--- ADDED THIS LINE
        }, f, indent=2, default=str)

    # 5. Save Raw Data (Parquet)
    print(f"💾 Saving raw data to {DATA_DIR}...")
    df.to_parquet(DATA_DIR / 'daily_raw.parquet', compression='gzip')

    # 6. Feature Engineering
    df = compute_technical_indicators(df)

    # 7. Monthly Resampling
    print("📅 Resampling to Monthly Frequency...")
    last_cols = [c for c in df.columns if c not in ['euro_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    data_monthly = pd.concat([
        df.unstack('ticker')['euro_volume'].resample('BM').mean().stack('ticker').to_frame('euro_volume'),
        df.unstack()[last_cols].resample('BM').last().stack('ticker')
    ], axis=1).dropna()

    data_monthly = data_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
    data_monthly = get_fama_french_betas(data_monthly)

    # Lag variables for prediction (avoid lookahead bias)
    vars_to_lag = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'euro_volume', 'garman_klass_vol']
    for col in vars_to_lag:
        if col in data_monthly.columns:
            data_monthly[f'{col}_lag1'] = data_monthly.groupby('ticker')[col].shift(1)

    print(f"💾 Saving processed monthly features...")
    data_monthly.to_parquet(DATA_DIR / 'monthly_features.parquet', compression='gzip')

    return df, data_monthly

# =============================================================================
# 5. PORTFOLIO OPTIMIZATION
# =============================================================================

def get_optimal_weights(prices_df):
    """
    Calculates Mean-Variance Optimization weights using Ledoit-Wolf shrinkage.
    """
    try:
        # Expected returns & Covariance
        mu = expected_returns.mean_historical_return(prices_df, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=252).ledoit_wolf()

        n_stocks = len(prices_df.columns)
        max_weight = max(0.25, 1.0 / n_stocks * 2.0) # Dynamic constraint

        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
        weights = ef.max_sharpe(risk_free_rate=0.03) # Maximize Sharpe
        cleaned_weights = ef.clean_weights()

        return cleaned_weights, True
    except Exception as e:
        print(f"⚠️ Optimization Failed: {e}")
        return {}, False

# =============================================================================
# 6. BACKTESTING ENGINE
# =============================================================================

def backtest_strategy_with_rebalancing(df_daily, df_monthly, xgb_model, kmeans_model):
    """
    Simulates the strategy historically with monthly rebalancing.
    Now supports dynamic Benchmarks via Config.
    """
    print("📊 Starting Backtest with Monthly Rebalancing...")
    
    initial_capital = 100.0
    current_portfolio_value = initial_capital
    current_benchmark_value = initial_capital
    
    portfolio_values = []
    rebalance_log = []
    
    daily_prices = df_daily['adj close'].unstack().ffill()
    
    # --- DOWNLOAD DYNAMIC BENCHMARK ---
    print(f"⬇️ Downloading Benchmark: {BENCHMARK_TICKER}")
    try:
        start_bench = df_daily.index.get_level_values('date').min()
        end_bench = df_daily.index.get_level_values('date').max() + pd.DateOffset(days=1)
        
        bench_data = yf.download(BENCHMARK_TICKER, start=start_bench, end=end_bench, progress=False, auto_adjust=False)
        
        if isinstance(bench_data.columns, pd.MultiIndex):
            bench_prices = bench_data['Close'].iloc[:, 0]
        else:
            bench_prices = bench_data['Close']
            
        benchmark_returns = bench_prices.reindex(daily_prices.index, method='ffill').pct_change().fillna(0)
        print(f"   ✓ Benchmark loaded ({len(benchmark_returns)} days)")
    except Exception as e:
        print(f"   ⚠️ Benchmark download failed ({e}). Using average market returns.")
        benchmark_returns = daily_prices.mean(axis=1).pct_change().fillna(0)

    # Features for XGBoost
    feature_cols = [
        'rsi', 'macd', 'bb_low', 'bb_high', 'atr',
        'return_2m', 'return_3m', 'return_6m',
        'euro_volume_lag1', 'garman_klass_vol_lag1',
        'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'RMW_lag1', 'CMA_lag1',
        'cluster'
    ]
    
    monthly_dates = df_monthly.index.get_level_values('date').unique().sort_values()
    print(f"   🔄 Simulating {len(monthly_dates)-1} rebalancing events...")

    # === MAIN SIMULATION LOOP ===
    for i, month_date in enumerate(monthly_dates[:-1]):
        
        # A. Prepare Data
        month_data = df_monthly.xs(month_date, level='date').copy()
        
        # B. Cluster Prediction
        if 'rsi' in month_data.columns:
            X_cluster = month_data[['rsi']].fillna(50)
            month_data['cluster'] = kmeans_model.predict(X_cluster)
        
        # C. XGBoost Prediction
        missing_cols = [c for c in feature_cols if c not in month_data.columns]
        if missing_cols: continue
        
        X_pred = month_data[feature_cols].fillna(0)
        month_data['proba_upside'] = xgb_model.predict_proba(X_pred)[:, 1]
        
        # D. Asset Selection (High proba + specific cluster)
        selected = month_data[
            (month_data['cluster'] == 3) & 
            (month_data['proba_upside'] > 0.55)
        ]
        
        # E. Optimization
        new_allocation = {}
        if not selected.empty:
            tickers = selected.index.tolist()
            prices_subset = daily_prices[tickers].iloc[-252:].dropna(axis=1)
            
            if not prices_subset.empty and len(prices_subset.columns) >= 3:
                weights, success = get_optimal_weights(prices_subset)
                new_allocation = weights if success else {t: 1.0/len(tickers) for t in tickers}
        
        # F. Step Forward (Daily Returns)
        next_month_start = monthly_dates[i + 1]
        trading_mask = (daily_prices.index >= month_date) & (daily_prices.index < next_month_start)
        trading_days = daily_prices.index[trading_mask]
        
        if len(trading_days) == 0: continue
        
        for date in trading_days:
            bench_ret = benchmark_returns.get(date, 0.0)
            
            if new_allocation:
                try:
                    portfolio_tickers = list(new_allocation.keys())
                    weights_array = np.array(list(new_allocation.values()))
                    
                    # Calculate daily return for portfolio
                    daily_rets = []
                    for ticker in portfolio_tickers:
                        if ticker in daily_prices.columns:
                            price_series = daily_prices[ticker]
                            idx = price_series.index.get_loc(date)
                            if idx > 0:
                                ret = (price_series.iloc[idx] / price_series.iloc[idx-1]) - 1
                                daily_rets.append(ret if not pd.isna(ret) else 0.0)
                            else:
                                daily_rets.append(0.0)
                        else:
                            daily_rets.append(0.0)
                    
                    strat_ret = np.average(daily_rets, weights=weights_array) if daily_rets else 0.0
                except Exception:
                    strat_ret = 0.0
            else:
                strat_ret = 0.0 # Cash position
            
            current_benchmark_value *= (1 + bench_ret)
            current_portfolio_value *= (1 + strat_ret)
            
            portfolio_values.append({
                'Date': date,
                'Strategy': current_portfolio_value,
                'Benchmark': current_benchmark_value
            })
        
        rebalance_log.append({
            'Date': month_date,
            'N_Stocks': len(new_allocation),
            'Allocation': new_allocation
        })
        
        if (i + 1) % 12 == 0:
            print(f"   📅 {month_date.date()} | Portfolio: {current_portfolio_value:.2f} | Bench: {current_benchmark_value:.2f}")

    # Export Results
    history_df = pd.DataFrame(portfolio_values).set_index('Date')
    rebalance_df = pd.DataFrame(rebalance_log).set_index('Date')
    
    print(f"   ✅ Backtest Complete. Final Value: {current_portfolio_value:.2f}")
    return history_df, rebalance_df

# =============================================================================
# 7. MAIN ORCHESTRATOR
# =============================================================================

def run_pipeline():
    """Main execution entry point."""
    start_time = datetime.now()
    print("\n" + "="*60)
    print(f"🚀 STARTING DAILY PIPELINE | {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # A. Load Models
        xgb_model, kmeans_model = load_models()
        if xgb_model is None: raise Exception("ML Models not found.")

        # B. Run ETL
        df_daily, df_monthly = get_data_pipeline()
        if df_daily is None: raise Exception("Data Pipeline Failed.")

        # C. Generate Latest Signals (For Dashboard)
        last_date = df_monthly.index.get_level_values('date').max()
        print(f"📅 Generating signals for: {last_date.date()}")
        
        today_data = df_monthly.xs(last_date, level=0).copy()
        
        if 'rsi' in today_data.columns:
            X_cluster = today_data[['rsi']].fillna(50)
            today_data['cluster'] = kmeans_model.predict(X_cluster)
            
        feature_cols = [
            'rsi', 'macd', 'bb_low', 'bb_high', 'atr',
            'return_2m', 'return_3m', 'return_6m',
            'euro_volume_lag1', 'garman_klass_vol_lag1',
            'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'RMW_lag1', 'CMA_lag1',
            'cluster'
        ]
        
        X_pred = today_data[feature_cols].fillna(0)
        today_data['proba_upside'] = xgb_model.predict_proba(X_pred)[:, 1]
        
        selected_stocks = today_data[
            (today_data['cluster'] == 3) & 
            (today_data['proba_upside'] > 0.55)
        ]
        
        print(f"✅ Selected Assets: {selected_stocks.index.tolist()}")
        
        # Calculate current target allocation
        final_alloc = {}
        if not selected_stocks.empty:
            tickers = selected_stocks.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-252:].dropna(axis=1)
            weights, success = get_optimal_weights(prices_subset)
            final_alloc = weights if success else {t: 1.0/len(tickers) for t in tickers}

        # D. Save Signals
        export_df = today_data[['cluster', 'proba_upside', 'rsi', 'return_3m']].reset_index()
        export_df.rename(columns={'ticker': 'Ticker', 'proba_upside': 'Proba_Hausse', 'return_3m': 'Return_3M', 'cluster': 'Cluster', 'rsi': 'RSI'}, inplace=True)
        export_df['Proba_Hausse'] *= 100
        export_df['Allocation'] = export_df['Ticker'].map(final_alloc).fillna(0.0)
        export_df['Signal'] = np.where(export_df['Allocation'] > 0, 'BUY', 'NEUTRAL')
        
        export_df.to_csv(project_root / 'latest_signals.csv', index=False)
        print("   ✓ Saved latest_signals.csv")

        # E. Run Full Backtest (Recalculate history)
        hist_df, rebal_df = backtest_strategy_with_rebalancing(df_daily, df_monthly, xgb_model, kmeans_model)
        
        hist_df.to_csv(project_root / 'portfolio_history.csv')
        rebal_df.to_csv(project_root / 'rebalance_history.csv')
        print("   ✓ Saved portfolio_history.csv & rebalance_history.csv")

        # F. Save Metadata
        metadata = {
            'market_name': MARKET_NAME,
            'last_update': datetime.now().isoformat(),
            'data_start': df_daily.index.get_level_values('date').min().isoformat(),
            'n_assets_tracked': len(TICKERS),
            'current_allocation': final_alloc
        }
        with open(project_root / 'data_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n🏁 PIPELINE COMPLETED SUCCESSFULLY in {duration:.1f}s")
        
        # Debug file
        with open(project_root / 'debug_run.txt', 'w') as f:
            f.write(f"Success: {datetime.now()}\nMarket: {MARKET_NAME}\nAssets: {len(TICKERS)}")

    except Exception as e:
        import traceback
        error_msg = f"CRITICAL FAILURE: {str(e)}\n\n{traceback.format_exc()}"
        print(f"\n❌ {error_msg}")
        
        with open(LOG_DIR / 'pipeline_errors.log', 'a') as f:
            f.write(f"[{datetime.now()}] {error_msg}\n")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()