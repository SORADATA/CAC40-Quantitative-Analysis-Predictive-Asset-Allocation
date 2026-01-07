import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import json 
import time

# --- IMPORTS LIBRAIRIE 'TA' ---
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Ignorer les warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CAC40_TICKERS = [
    "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "ATO.PA", "CS.PA", "BNP.PA",
    "EN.PA", "CAP.PA", "CA.PA", "DSY.PA", "EL.PA", "ENGI.PA", "ERF.PA",
    "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA", "ORA.PA",
    "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA",
    "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA",
    "URW.PA", "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA", "FR.PA"
]

# Détection automatique des chemins
current_path = Path(os.getcwd())
project_root = current_path
while not (project_root / 'app.py').exists():
    if project_root == project_root.parent:
        project_root = Path(os.getcwd())
        break
    project_root = project_root.parent

MODEL_DIR = project_root / "src" / "models"
DATA_DIR = project_root / "data" / "raw"

# Créer les dossiers si nécessaire
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. GESTION DES TICKERS
# =============================================================================

def handle_ticker_changes():
    """
    Gère les changements de tickers connus (fusions, renommages)
    🔧 À mettre à jour manuellement quand nécessaire
    """
    # Dictionnaire des changements de tickers
    TICKER_CHANGES = {
        # Format: 'OLD_TICKER': 'NEW_TICKER'
        # Exemple: 'ALU.PA': 'NOK.PA'  # Alcatel-Lucent -> Nokia
    }
    
    # Tickers définitivement delistés (à exclure)
    DELISTED_TICKERS = [
        # Exemple: 'XXX.PA'
    ]
    
    return TICKER_CHANGES, DELISTED_TICKERS


def validate_and_clean_tickers(df, tickers_list, max_days_stale=30):
    """
    Nettoie les tickers obsolètes et gère les changements
    🔧 Détecte: delisting, fusion, suspension, changement de ticker
    
    Returns:
        - df nettoyé
        - liste des tickers valides
        - dict des alertes
    """
    print("🔍 Validation des tickers...")
    
    alerts = {
        'delisted': [],
        'stale': [],
        'missing': [],
        'warnings': []
    }
    
    # 1. Vérifier la présence de tous les tickers
    tickers_in_data = df.index.get_level_values('ticker').unique().tolist()
    missing_tickers = set(tickers_list) - set(tickers_in_data)
    
    if missing_tickers:
        alerts['missing'] = list(missing_tickers)
        print(f"   ⚠️ {len(missing_tickers)} tickers manquants: {missing_tickers}")
    
    # 2. Détecter les tickers sans données récentes
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
        print(f"   ⚠️ {len(stale_tickers)} tickers obsolètes (>{max_days_stale}j):")
        for item in stale_tickers:
            print(f"      - {item['ticker']}: {item['days_stale']} jours (dernier: {item['last_date']})")
        
        # Suppression des tickers obsolètes
        tickers_to_remove = [t['ticker'] for t in stale_tickers]
        df = df[~df.index.get_level_values('ticker').isin(tickers_to_remove)]
        alerts['delisted'] = tickers_to_remove
        print(f"   ✅ {len(tickers_to_remove)} tickers supprimés")
    
    # 3. Vérifier les volumes anormalement bas (signe de problème)
    if 'volume' in df.columns:
        for ticker in df.index.get_level_values('ticker').unique():
            ticker_data = df.xs(ticker, level='ticker')
            recent_volume = ticker_data['volume'].tail(20).mean()
            
            if recent_volume < 1000:  # Volume quotidien < 1000
                alerts['warnings'].append(f"{ticker}: Volume très faible ({recent_volume:.0f})")
    
    # 4. Liste finale des tickers valides
    valid_tickers = df.index.get_level_values('ticker').unique().tolist()
    
    print(f"   ✅ {len(valid_tickers)}/{len(tickers_list)} tickers valides")
    
    return df, valid_tickers, alerts


# =============================================================================
# 3. FONCTIONS DE CHARGEMENT ET FEATURE ENGINEERING
# =============================================================================

def load_models():
    """Charge les modèles XGBoost et KMeans"""
    print(f"📂 Chargement des modèles depuis {MODEL_DIR}...")
    try:
        with open(MODEL_DIR / 'xgboost_model.pkl', 'rb') as f:
            xgb = pickle.load(f)
        with open(MODEL_DIR / 'kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        return xgb, kmeans
    except FileNotFoundError:
        print("❌ Erreur : Modèles introuvables.")
        return None, None


def compute_technical_indicators(df):
    """Calcul des indicateurs techniques avec la librairie 'ta'"""
    print("📊 Calcul des indicateurs techniques...")

    # 1. Garman Klass Volatility
    df['garman_klass_vol'] = (
        (np.log(df['high']) - np.log(df['low']))**2 / 2 - 
        (2*np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open']))**2
    )

    # 2. RSI
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = df.loc[idx, 'adj close']
        if len(close_series) > 20:
            rsi_indicator = RSIIndicator(close=close_series, window=20)
            df.loc[idx, 'rsi'] = rsi_indicator.rsi().values

    # 3. Bollinger Bands
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = np.log1p(df.loc[idx, 'adj close'])
        if len(close_series) > 20:
            bb = BollingerBands(close=close_series, window=20, window_dev=2)
            df.loc[idx, 'bb_low'] = bb.bollinger_lband().values
            df.loc[idx, 'bb_mid'] = bb.bollinger_mavg().values
            df.loc[idx, 'bb_high'] = bb.bollinger_hband().values

    # 4. ATR
    def compute_atr(stock_data):
        if len(stock_data) < 15:
            return pd.Series(np.nan, index=stock_data.index)
        atr_indicator = AverageTrueRange(
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            window=14
        )
        atr = atr_indicator.average_true_range()
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    # 5. MACD
    def compute_macd(stock_data):
        if len(stock_data) < 26:
            return pd.Series(np.nan, index=stock_data.index)
        close_series = stock_data['adj close']
        macd_indicator = MACD(
            close=close_series,
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        macd_val = macd_indicator.macd()
        return macd_val.sub(macd_val.mean()).div(macd_val.std())

    df['macd'] = df.groupby(level=1, group_keys=False).apply(compute_macd)

    # 6. Euro Volume
    df['euro_volume'] = (df['adj close'] * df['volume']) / 1e6

    return df


def calculate_returns(df):
    """Calcule les rendements momentum sur données mensuelles"""
    print("📈 Calcul des Returns Momentum...")
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    min_periods_monthly = 12

    for lag in lags:
        returns_raw = df['adj close'].pct_change(lag)
        lower_bound = returns_raw.expanding(min_periods=min_periods_monthly).quantile(outlier_cutoff)
        upper_bound = returns_raw.expanding(min_periods=min_periods_monthly).quantile(1 - outlier_cutoff)
        df[f'return_{lag}m'] = returns_raw.clip(lower=lower_bound, upper=upper_bound)

    return df


def get_fama_french_betas(data):
    """Récupère les facteurs Fama-French et calcule les Rolling Betas"""
    print("🌍 Récupération Fama-French Europe...")
    try:
        factor_data = web.DataReader('Europe_5_Factors', 'famafrench', start='2010')[0].drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample('BM').last().div(100)
        factor_data.index.name = 'date'

        data_ff = data.copy()
        if 'return_1m' not in data_ff.columns:
            return data

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
            except Exception as e:
                continue

        if not betas_list:
            return data

        betas_df = pd.concat(betas_list).set_index('ticker', append=True)
        data = data.join(betas_df.groupby('ticker').shift())
        data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(
            lambda x: x.fillna(x.mean())
        )

        return data

    except Exception as e:
        print(f"⚠️ Erreur Fama-French : {e}")
        for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            data[f] = 0.0
        return data


def get_data_pipeline():
    """Pipeline complet avec validation des tickers et retry mechanism"""
    
    # 1. Gestion des changements de tickers
    ticker_changes, delisted = handle_ticker_changes()
    
    # Appliquer les changements
    active_tickers = [t for t in CAC40_TICKERS if t not in delisted]
    for old, new in ticker_changes.items():
        if old in active_tickers:
            active_tickers[active_tickers.index(old)] = new
    
    print(f"📋 Tickers actifs: {len(active_tickers)}/{len(CAC40_TICKERS)}")
    
    # 2. DOWNLOAD avec retry mechanism
    end_date = (datetime.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(datetime.today()) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    print(f"⬇️ Téléchargement des données ({start_date} → {end_date})...")
    
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    df = None
    
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                active_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=True
            )
            
            if not df.empty:
                print(f"   ✅ Téléchargement réussi (tentative {attempt + 1}/{MAX_RETRIES})")
                break
            else:
                print(f"   ⚠️ Données vides (tentative {attempt + 1}/{MAX_RETRIES})")
        except Exception as e:
            print(f"   ❌ Erreur téléchargement (tentative {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"   ⏳ Nouvelle tentative dans {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"   ❌ ÉCHEC DÉFINITIF après {MAX_RETRIES} tentatives")
                return None, None

    if df is None or df.empty:
        print("❌ Erreur : Aucune donnée téléchargée après tous les retries.")
        return None, None

    # 3. Structuration
    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()

    if 'adj close' not in df.columns and 'close' in df.columns:
        df['adj close'] = df['close']

    # 4. VALIDATION DES DONNÉES TÉLÉCHARGÉES
    print("🔍 Validation des données...")
    
    # Vérifier le nombre de tickers récupérés
    n_tickers_downloaded = df.index.get_level_values('ticker').nunique()
    n_tickers_expected = len(active_tickers)
    
    if n_tickers_downloaded < n_tickers_expected * 0.7:  # Minimum 70%
        print(f"   ⚠️ WARNING: Seulement {n_tickers_downloaded}/{n_tickers_expected} tickers téléchargés")
        missing_tickers = set(active_tickers) - set(df.index.get_level_values('ticker').unique())
        print(f"   ❌ Tickers manquants: {missing_tickers}")
    else:
        print(f"   ✅ {n_tickers_downloaded}/{n_tickers_expected} tickers OK")
    
    # Vérifier la fraîcheur des données
    last_date = df.index.get_level_values('date').max()
    days_old = (datetime.today() - last_date).days
    
    if days_old > 5:  # Plus de 5 jours = suspect
        print(f"   ⚠️ WARNING: Dernières données datent de {days_old} jours ({last_date.date()})")
    else:
        print(f"   ✅ Données à jour ({last_date.date()})")

    # 5. NETTOYAGE DES TICKERS OBSOLÈTES
    df, valid_tickers, alerts = validate_and_clean_tickers(df, active_tickers)
    
    # 6. Sauvegarde rapport de validation
    validation_report = {
        'date': datetime.now().isoformat(),
        'total_tickers_requested': len(active_tickers),
        'valid_tickers': len(valid_tickers),
        'alerts': alerts
    }
    
    with open(project_root / 'ticker_validation.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"   📄 Rapport de validation sauvegardé")

    # 7. Sauvegarde données brutes
    print(f"💾 Sauvegarde des données brutes (compressées)...")
    df.to_parquet(DATA_DIR / 'cac40_daily_raw.parquet', compression='gzip')

    # 8. Indicateurs Techniques
    df = compute_technical_indicators(df)

    # 9. Resampling Mensuel
    print("📅 Resampling Mensuel (Business Days)...")
    last_cols = [c for c in df.columns if c not in ['euro_volume', 'volume', 'open', 'high', 'low', 'close']]

    data_monthly = pd.concat([
        df.unstack('ticker')['euro_volume'].resample('BM').mean().stack('ticker').to_frame('euro_volume'),
        df.unstack()[last_cols].resample('BM').last().stack('ticker')
    ], axis=1).dropna()

    # 10. Momentum Returns
    data_monthly = data_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)

    # 11. Fama-French
    data_monthly = get_fama_french_betas(data_monthly)

    # 12. Lag Variables
    vars_to_lag = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'euro_volume', 'garman_klass_vol']
    for col in vars_to_lag:
        if col in data_monthly.columns:
            data_monthly[f'{col}_lag1'] = data_monthly.groupby('ticker')[col].shift(1)

    # Sauvegarde données mensuelles
    print(f"💾 Sauvegarde des données mensuelles (compressées)...")
    data_monthly.to_parquet(DATA_DIR / 'cac40_monthly_features.parquet', compression='gzip')

    data_monthly = data_monthly.drop(columns=['adj close'], errors='ignore')

    return df, data_monthly


# =============================================================================
# 4. OPTIMISATION MARKOWITZ
# =============================================================================
def get_optimal_weights(prices_df):
    """Optimisation Markowitz avec Ledoit-Wolf shrinkage"""
    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=252).ledoit_wolf()

        n_stocks = len(prices_df.columns)
        max_weight = max(0.25, 1.0 / n_stocks * 2.0)

        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
        weights = ef.max_sharpe(risk_free_rate=0.03)
        cleaned_weights = ef.clean_weights()

        # Validation des poids
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"   ⚠️ Warning: Poids totaux = {total_weight:.4f}")

        return cleaned_weights, True
    except Exception as e:
        print(f"⚠️ Erreur Optimisation : {e}")
        return {}, False


# =============================================================================
# 5. BACKTEST AVEC REBALANCEMENT MENSUEL
# =============================================================================
def backtest_strategy_with_rebalancing(df_daily, df_monthly, xgb_model, kmeans_model):
    
    print("📊 Backtest avec rebalancement mensuel...")
    
    initial_capital = 100.0
    current_portfolio_value = initial_capital
    current_benchmark_value = initial_capital
    
    portfolio_values = []
    rebalance_log = []
    
    # Prix journaliers
    daily_prices = df_daily['adj close'].unstack().ffill()
    
    # Benchmark CAC40
    try:
        start_bench = df_daily.index.get_level_values('date').min()
        end_bench = df_daily.index.get_level_values('date').max() + pd.DateOffset(days=1)
        cac40 = yf.download('^FCHI', start=start_bench, end=end_bench, progress=False, auto_adjust=False)
        
        if isinstance(cac40.columns, pd.MultiIndex):
            cac40_prices = cac40['Close'].iloc[:, 0]
        else:
            cac40_prices = cac40['Close']
        
        # ✅ CORRECTION : Aligner les dates exactement
        benchmark_returns = cac40_prices.reindex(daily_prices.index, method='ffill').pct_change().fillna(0)
        print(f"   ✓ CAC40 téléchargé ({len(benchmark_returns)} jours)")
    except Exception as e:
        print(f"   ⚠️ Erreur CAC40, fallback : {e}")
        benchmark_returns = daily_prices.mean(axis=1).pct_change().fillna(0)
    
    # Features XGBoost
    feature_cols = [
        'rsi', 'macd', 'bb_low', 'bb_high', 'atr',
        'return_2m', 'return_3m', 'return_6m',
        'euro_volume_lag1', 'garman_klass_vol_lag1',
        'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'RMW_lag1', 'CMA_lag1',
        'cluster'
    ]
    
    # Dates mensuelles
    monthly_dates = df_monthly.index.get_level_values('date').unique().sort_values()
    
    print(f"   🔄 Rebalancement sur {len(monthly_dates)-1} mois...")
    
    # === BOUCLE PRINCIPALE ===
    for i, month_date in enumerate(monthly_dates[:-1]):
        
        # 1. Récupération données du mois
        month_data = df_monthly.xs(month_date, level='date').copy()
        
        # 2. Clustering
        if 'rsi' in month_data.columns:
            X_cluster = month_data[['rsi']].fillna(50)
            month_data['cluster'] = kmeans_model.predict(X_cluster)
        
        # 3. Prédiction XGBoost
        missing_cols = [c for c in feature_cols if c not in month_data.columns]
        if missing_cols:
            continue
        
        X_pred = month_data[feature_cols].fillna(0)
        month_data['proba_hausse'] = xgb_model.predict_proba(X_pred)[:, 1]
        
        # 4. Sélection actions
        selected = month_data[
            (month_data['cluster'] == 3) &
            (month_data['proba_hausse'] > 0.55)
        ]
        
        # 5. Optimisation Markowitz
        new_allocation = {}
        if not selected.empty:
            tickers = selected.index.tolist()
            prices_subset = daily_prices[tickers].iloc[-252:].dropna(axis=1)
            
            if not prices_subset.empty and len(prices_subset.columns) >= 3:
                weights, success = get_optimal_weights(prices_subset)
                if success:
                    new_allocation = weights
                else:
                    new_allocation = {t: 1.0/len(prices_subset.columns) for t in prices_subset.columns}
        
        # 6. Application pour le mois suivant
        next_month_start = monthly_dates[i + 1]
        trading_mask = (daily_prices.index >= month_date) & (daily_prices.index < next_month_start)
        trading_days = daily_prices.index[trading_mask]
        
        if len(trading_days) == 0:
            continue
        
        # ✅ CORRECTION : Calcul CORRECT des rendements quotidiens
        for date in trading_days:
            # Benchmark
            bench_ret = benchmark_returns.get(date, 0.0)
            if pd.isna(bench_ret):
                bench_ret = 0.0
            
            # Stratégie
            if new_allocation:
                try:
                    portfolio_tickers = list(new_allocation.keys())
                    weights_array = np.array(list(new_allocation.values()))
                    
                    # ✅ CORRECTION CRITIQUE : Utiliser .get() avec valeur par défaut
                    daily_rets = []
                    for ticker in portfolio_tickers:
                        if ticker in daily_prices.columns:
                            # Calcul du rendement entre hier et aujourd'hui
                            price_series = daily_prices[ticker]
                            idx = price_series.index.get_loc(date)
                            if idx > 0:
                                ret = (price_series.iloc[idx] / price_series.iloc[idx-1]) - 1
                                daily_rets.append(ret if not pd.isna(ret) else 0.0)
                            else:
                                daily_rets.append(0.0)
                        else:
                            daily_rets.append(0.0)
                    
                    if daily_rets:
                        strat_ret = np.average(daily_rets, weights=weights_array)
                    else:
                        strat_ret = 0.0
                        
                except (KeyError, IndexError) as e:
                    strat_ret = 0.0
            else:
                strat_ret = 0.0  # Cash
            
            # Mise à jour
            current_benchmark_value *= (1 + bench_ret)
            current_portfolio_value *= (1 + strat_ret)
            
            portfolio_values.append({
                'Date': date,
                'Strategy': current_portfolio_value,
                'Benchmark': current_benchmark_value
            })
        
        # Log rebalancement
        rebalance_log.append({
            'Date': month_date,
            'N_Stocks': len(new_allocation),
            'Allocation': new_allocation
        })
        
        # Affichage périodique
        if (i + 1) % 12 == 0:
            print(f"   {month_date.date()} | Portfolio: {current_portfolio_value:.2f} | Bench: {current_benchmark_value:.2f}")
    
    # Conversion DataFrames
    history_df = pd.DataFrame(portfolio_values).set_index('Date')
    rebalance_df = pd.DataFrame(rebalance_log).set_index('Date')
    
    print(f"   ✅ Backtest terminé : {len(history_df)} jours, {len(rebalance_df)} rebalancements")
    print(f"   📈 Performance finale : Strategy={current_portfolio_value:.2f} vs Bench={current_benchmark_value:.2f}")
    
    return history_df, rebalance_df

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """Pipeline principal avec gestion d'erreurs complète"""
    start_time = datetime.now()
    print("🚀 Démarrage du Daily Run...")
    print(f"📅 Timestamp : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # A. Chargement Modèles
        xgb_model, kmeans_model = load_models()
        if xgb_model is None:
            raise Exception("Modèles ML introuvables")

        # B. Récupération & Feature Engineering
        df_daily, df_monthly = get_data_pipeline()

        if df_daily is None or df_monthly is None:
            raise Exception("Échec téléchargement données")

        # C. Prédiction sur dernières données mensuelles (pour latest_signals.csv)
        last_date = df_monthly.index.get_level_values('date').max()
        print(f"📅 Analyse basée sur les données mensuelles du : {last_date.date()}")

        today_data = df_monthly.xs(last_date, level=0).copy()

        # D. Clustering
        if 'rsi' in today_data.columns:
            X_cluster = today_data[['rsi']].fillna(50)
            today_data['cluster'] = kmeans_model.predict(X_cluster)

        # E. Prédiction XGBoost
        feature_cols = [
            'rsi', 'macd', 'bb_low', 'bb_high', 'atr',
            'return_2m', 'return_3m', 'return_6m',
            'euro_volume_lag1', 'garman_klass_vol_lag1',
            'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'RMW_lag1', 'CMA_lag1',
            'cluster'
        ]

        missing_cols = [c for c in feature_cols if c not in today_data.columns]
        if missing_cols:
            raise Exception(f"Colonnes manquantes : {missing_cols}")

        X_pred = today_data[feature_cols].fillna(0)

        print("🔮 Prédictions XGBoost...")
        today_data['proba_hausse'] = xgb_model.predict_proba(X_pred)[:, 1]

        # F. Filtrage Stratégie
        selected_stocks = today_data[
            (today_data['cluster'] == 3) &
            (today_data['proba_hausse'] > 0.55)
        ]

        print(f"✅ Actions sélectionnées ({len(selected_stocks)}) : {selected_stocks.index.tolist()}")

        # G. Optimisation Markowitz (pour affichage seulement)
        final_alloc = {}
        if not selected_stocks.empty:
            tickers = selected_stocks.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-252:].dropna(axis=1)

            if not prices_subset.empty:
                weights, success = get_optimal_weights(prices_subset)
                if success:
                    final_alloc = weights
                else:
                    final_alloc = {t: 1.0/len(prices_subset.columns) for t in prices_subset.columns}

        # H. Sauvegarde Résultats
        print("💾 Sauvegarde des résultats...")

        # 1. Signaux (Latest)
        if not today_data.empty:
            cols_to_export = ['cluster', 'proba_hausse']
            if 'rsi' in today_data.columns:
                cols_to_export.append('rsi')
            if 'return_3m' in today_data.columns:
                cols_to_export.append('return_3m')

            export_df = today_data[cols_to_export].reset_index()
            col_mapping = {
                'ticker': 'Ticker',
                'cluster': 'Cluster',
                'proba_hausse': 'Proba_Hausse',
                'rsi': 'RSI',
                'return_3m': 'Return_3M'
            }
            export_df.rename(columns=col_mapping, inplace=True)
            export_df['Proba_Hausse'] = export_df['Proba_Hausse'] * 100
            export_df['Allocation'] = export_df['Ticker'].map(final_alloc).fillna(0.0)
            export_df['Signal'] = np.where(export_df['Allocation'] > 0, 'ACHAT', 'NEUTRE')
            export_df = export_df.sort_values(by='Allocation', ascending=False)

            signals_file = str(project_root / 'latest_signals.csv')
            export_df.to_csv(signals_file, index=False)
            print(f"   ✓ latest_signals.csv")

        # =============================================================================
        # 2. BACKTEST COMPLET AVEC REBALANCEMENT
        # =============================================================================
       
        
        history_file = str(project_root / 'portfolio_history.csv')
        rebalance_file = str(project_root / 'rebalance_history.csv')
        
        # ✅ TOUJOURS RECALCULER
        print("   🔄 Recalcul COMPLET de l'historique (2-3 min)...")
        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily, df_monthly, xgb_model, kmeans_model
        )
        
        hist_df.to_csv(history_file, date_format='%Y-%m-%d')
        rebal_df.to_csv(rebalance_file, date_format='%Y-%m-%d')
        
        print(f"   ✅ portfolio_history.csv ({len(hist_df)} jours)")
        print(f"   ✅ rebalance_history.csv ({len(rebal_df)} mois)")

    
        # 3. Metadata
        metadata = {
            'last_update': datetime.now().isoformat(),
            'data_start': df_daily.index.get_level_values('date').min().isoformat(),
            'data_end': df_daily.index.get_level_values('date').max().isoformat(),
            'n_stocks': len(CAC40_TICKERS),
            'n_selected': len(selected_stocks),
            'allocations': final_alloc,
            'data_files': {
                'daily_raw': str(DATA_DIR / 'cac40_daily_raw.parquet'),
                'monthly_features': str(DATA_DIR / 'cac40_monthly_features.parquet'),
                'signals': str(project_root / 'latest_signals.csv'),
                'history': str(project_root / 'portfolio_history.csv'),
                'rebalance': str(project_root / 'rebalance_history.csv')
            }
        }

        metadata_file = str(project_root / 'data_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ data_metadata.json")

        # 4. Debug file
        debug_file = str(project_root / 'debug_run.txt')
        with open(debug_file, "w") as f:
            f.write(f"Run finished at {datetime.now()}\n")
            f.write(f"Daily data shape: {df_daily.shape}\n")
            f.write(f"Monthly data shape: {df_monthly.shape}\n")
            f.write(f"Selected stocks: {len(selected_stocks)}\n")
            f.write(f"Allocations: {final_alloc}\n")
            f.write(f"\n=== Latest Values ===\n")
            if not hist_df.empty:
                f.write(f"Strategy: {hist_df['Strategy'].iloc[-1]:.2f}\n")
                f.write(f"Benchmark: {hist_df['Benchmark'].iloc[-1]:.2f}\n")
            f.write(f"\n=== File Paths ===\n")
            f.write(f"Project root: {project_root}\n")
            f.write(f"History file: {history_file}\n")
        print(f"   ✓ debug_run.txt")

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n🏁 Pipeline terminé avec succès !")
        print(f"⏱️ Durée totale : {duration:.1f}s")
        print(f"\n📂 Fichiers sauvegardés dans: {project_root}")
        
    except Exception as e:
        import traceback
        error_msg = f"Pipeline Critical Failure: {str(e)}\n\n{traceback.format_exc()}"
        print(f"\n❌ ERREUR CRITIQUE:\n{error_msg}")
        
        # Log l'erreur
        log_dir = project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        error_file = log_dir / 'pipeline_errors.log'
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.now()}] CRITICAL ERROR\n")
            f.write(f"{'='*80}\n")
            f.write(f"{error_msg}\n")
        
        raise


if __name__ == "__main__":
    run_pipeline()
