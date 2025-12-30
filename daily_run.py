import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import json 

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
# 2. FONCTIONS DE CHARGEMENT ET FEATURE ENGINEERING
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
        factor_data = factor_data.resample('BM').last().div(100)  # BM au lieu de M
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
                print(f"   ⚠️ Beta échoué pour {ticker}: {e}")
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
    """Pipeline complet : Download -> Tech -> Monthly -> Macro -> Clean"""
    
    # 1. DOWNLOAD (CORRECTION: +1 jour pour inclusion de today)
    end_date = (datetime.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(datetime.today()) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    
    print(f"⬇️ Téléchargement des données ({start_date} → {end_date})...")
    df = yf.download(CAC40_TICKERS, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if df.empty:
        print("❌ Erreur : Aucune donnée téléchargée.")
        return None, None
    
    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    
    if 'adj close' not in df.columns and 'close' in df.columns:
        df['adj close'] = df['close']
    
    # Sauvegarde données brutes (compression pour réduire taille)
    print(f"💾 Sauvegarde des données brutes (compressées)...")
    df.to_parquet(DATA_DIR / 'cac40_daily_raw.parquet', compression='gzip')
        
    # 2. Indicateurs Techniques
    df = compute_technical_indicators(df)
    
    # 3. Resampling Mensuel (BM = Business Month)
    print("📅 Resampling Mensuel (Business Days)...")
    last_cols = [c for c in df.columns if c not in ['euro_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    data_monthly = pd.concat([
        df.unstack('ticker')['euro_volume'].resample('BM').mean().stack('ticker').to_frame('euro_volume'),
        df.unstack()[last_cols].resample('BM').last().stack('ticker')
    ], axis=1).dropna()
    
    # 4. Momentum Returns
    data_monthly = data_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
    
    # 5. Fama-French
    data_monthly = get_fama_french_betas(data_monthly)
    
    # 6. Lag Variables
    vars_to_lag = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'euro_volume', 'garman_klass_vol']
    for col in vars_to_lag:
        if col in data_monthly.columns:
            data_monthly[f'{col}_lag1'] = data_monthly.groupby('ticker')[col].shift(1)
    
    # Sauvegarde données mensuelles (compressées)
    print(f"💾 Sauvegarde des données mensuelles (compressées)...")
    data_monthly.to_parquet(DATA_DIR / 'cac40_monthly_features.parquet', compression='gzip')
    
    data_monthly = data_monthly.drop(columns=['adj close'], errors='ignore')
    
    return df, data_monthly

# =============================================================================
# 3. OPTIMISATION MARKOWITZ
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
# 4. MAIN PIPELINE
# =============================================================================
def run_pipeline():
    start_time = datetime.now()
    print("🚀 Démarrage du Daily Run...")
    print(f"📅 Timestamp : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # A. Chargement Modèles
    xgb_model, kmeans_model = load_models()
    if xgb_model is None:
        return

    # B. Récupération & Feature Engineering
    df_daily, df_monthly = get_data_pipeline()
    
    if df_daily is None or df_monthly is None:
        print("❌ Pipeline arrêté : Données manquantes")
        return
    
    # C. Prédiction sur dernières données mensuelles
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
        print(f"❌ Colonnes manquantes : {missing_cols}")
        return

    X_pred = today_data[feature_cols].fillna(0)
    
    print("🔮 Prédictions XGBoost...")
    today_data['proba_hausse'] = xgb_model.predict_proba(X_pred)[:, 1]
    
    # F. Filtrage Stratégie
    selected_stocks = today_data[
        (today_data['cluster'] == 3) &
        (today_data['proba_hausse'] > 0.55)
    ]
    
    print(f"✅ Actions sélectionnées ({len(selected_stocks)}) : {selected_stocks.index.tolist()}")
    
    # G. Optimisation Markowitz
    final_alloc = {}
    if not selected_stocks.empty:
        tickers = selected_stocks.index.tolist()
        prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-252:].dropna(axis=1)
        
        if not prices_subset.empty:
            weights, success = get_optimal_weights(prices_subset)
            if success:
                final_alloc = weights
            else:
                # Fallback équipondéré
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
        
        export_df['Allocation'] = export_df['Ticker'].map(final_alloc).fillna(0.0)
        export_df['Signal'] = np.where(export_df['Allocation'] > 0, 'ACHAT', 'NEUTRE')
        export_df = export_df.sort_values(by='Allocation', ascending=False)
        
        export_df.to_csv('latest_signals.csv', index=False)
        print("   ✓ latest_signals.csv")

    # -------------------------------------------------------------
    # 2. Historique (Mise à jour incrémentale OPTIMISÉE)
    # -------------------------------------------------------------
    print("📜 Mise à jour de l'historique Strategy vs Benchmark...")

    history_file = 'portfolio_history.csv'

    # Préparation des données de prix
    daily_prices = df_daily['adj close'].unstack()
    market_returns = daily_prices.pct_change().mean(axis=1)

    # Chargement historique
    if os.path.exists(history_file):
        hist_df = pd.read_csv(history_file, index_col=0)
        # --- CORRECTIF : Format mixed pour gérer les formats de dates inconsistants ---
        hist_df.index = pd.to_datetime(hist_df.index, format='mixed', errors='coerce')
        # On supprime les lignes où la date n'a pas pu être lue (NaT)
        hist_df = hist_df[hist_df.index.notna()]
        hist_df.sort_index(inplace=True)
        last_recorded_date = hist_df.index[-1]
        print(f"   📖 Historique chargé : {len(hist_df)} jours (dernier: {last_recorded_date.date()})")
    else:
        print("   ⚠️ Création nouveau fichier historique")
        start_date = market_returns.index[0]
        hist_df = pd.DataFrame(
            {'Strategy': 100.0, 'Benchmark': 100.0},
            index=[start_date]
        )
        last_recorded_date = start_date

    # Identification des nouveaux jours
    new_dates = market_returns[market_returns.index > last_recorded_date]

    if new_dates.empty:
        print(f"   ✓ Historique à jour")
    else:
        print(f"   → Ajout de {len(new_dates)} jour(s)...")
        
        current_strat_val = hist_df['Strategy'].iloc[-1]
        current_bench_val = hist_df['Benchmark'].iloc[-1]
        
        new_rows = []
        
        # OPTIMISATION : Calcul vectorisé si portefeuille actif
        weights = None
        portfolio_returns = None
        
        if final_alloc:
            portfolio_tickers = list(final_alloc.keys())
            weights = np.array(list(final_alloc.values()))
            
            # Préparation des rendements du portefeuille
            portfolio_prices = daily_prices[portfolio_tickers]
            portfolio_returns = portfolio_prices.pct_change()
        
        for date in new_dates.index:
            # Vérification alignement
            if date not in daily_prices.index:
                print(f"   ⚠️ Date {date.date()} absente des prix quotidiens")
                continue
            
            # Benchmark
            ret_bench = market_returns.loc[date]
            if pd.isna(ret_bench):
                ret_bench = 0.0

            # === CALCUL DU RENDEMENT STRATÉGIE ===
            if final_alloc and portfolio_returns is not None:
                # On récupère les returns des actions sélectionnées pour ce jour
                try:
                    ticker_returns = portfolio_returns.loc[date].fillna(0)
                    ret_strat = np.average(ticker_returns, weights=weights)
                except KeyError:
                    # Si une date manque pour le portfolio spécifique
                    ret_strat = 0.0
            else:
                ret_strat = 0.0

            # Mise à jour valeurs
            current_bench_val = current_bench_val * (1 + ret_bench)
            current_strat_val = current_strat_val * (1 + ret_strat)
            
            new_rows.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Strategy': round(current_strat_val, 2),
                'Benchmark': round(current_bench_val, 2)
            })
            
            print(f"      {date.date()}: Strat {current_strat_val:.2f} ({ret_strat*100:+.2f}%) | "
                  f"Bench {current_bench_val:.2f} ({ret_bench*100:+.2f}%)")

        # Sauvegarde
        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index('Date')
            new_df.index = pd.to_datetime(new_df.index)
            hist_df = pd.concat([hist_df, new_df])
            # Nettoyage final pour avoir un format de date propre dans le CSV
            hist_df.index = pd.to_datetime(hist_df.index) 
            hist_df.to_csv(history_file, date_format='%Y-%m-%d')
            print(f"   ✓ portfolio_history.csv ({len(hist_df)} jours)")

    # 3. Metadata
    metadata = {
        'last_update': datetime.now().isoformat(),
        'data_start': df_daily.index.get_level_values('date').min().isoformat(),
        'data_end': df_daily.index.get_level_values('date').max().isoformat(),
        'n_stocks': len(CAC40_TICKERS),
        'n_selected': len(selected_stocks),
        'allocations': final_alloc,
        'data_files': {
            'daily_raw': 'data/raw/cac40_daily_raw.parquet',
            'monthly_features': 'data/raw/cac40_monthly_features.parquet',
            'signals': 'latest_signals.csv',
            'history': 'portfolio_history.csv'
        }
    }
    
    with open('data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✓ data_metadata.json")
    
    # 4. Debug file
    with open("debug_run.txt", "w") as f:
        f.write(f"Run finished at {datetime.now()}\n")
        f.write(f"Daily data shape: {df_daily.shape}\n")
        f.write(f"Monthly data shape: {df_monthly.shape}\n")
        f.write(f"Selected stocks: {len(selected_stocks)}\n")
        f.write(f"Allocations: {final_alloc}\n")
        f.write(f"\n=== Latest Values ===\n")
        if not hist_df.empty:
            f.write(f"Strategy: {hist_df['Strategy'].iloc[-1]:.2f}\n")
            f.write(f"Benchmark: {hist_df['Benchmark'].iloc[-1]:.2f}\n")
    print("   ✓ debug_run.txt")
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n🏁 Pipeline terminé avec succès !")
    print(f"⏱️ Durée totale : {duration:.1f}s")


if __name__ == "__main__":
    run_pipeline()