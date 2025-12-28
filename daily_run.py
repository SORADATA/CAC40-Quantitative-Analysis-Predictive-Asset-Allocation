import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Ignorer les warnings pour garder les logs propres
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

# Détection automatique des chemins (comme vu précédemment)
current_path = Path(os.getcwd())
project_root = current_path
while not (project_root / 'app.py').exists():
    if project_root == project_root.parent:
        project_root = Path(os.getcwd()) # Fallback
        break
    project_root = project_root.parent

MODEL_DIR = project_root / "src" / "models"
DATA_DIR = project_root / "data" / "raw"

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
        print("❌ Erreur : Modèles introuvables. Lancez les notebooks d'abord.")
        return None, None

def compute_technical_indicators(df):
    """Calcul des indicateurs techniques sur données journalières"""
    print("📊 Calcul des indicateurs techniques (RSI, BB, ATR, Vol, MACD)...")
    
    # 1. Garman Klass Volatility
    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    # 2. RSI avec ta
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = df.loc[idx, 'adj close']
        if len(close_series) > 20:
            rsi_indicator = RSIIndicator(close=close_series, window=20)
            df.loc[idx, 'rsi'] = rsi_indicator.rsi()

    # 3. Bollinger Bands avec ta
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close_series = np.log1p(df.loc[idx, 'adj close'])
        if len(close_series) > 20:
            bb = BollingerBands(close=close_series, window=20, window_dev=2)
            df.loc[idx, 'bb_low'] = bb.bollinger_lband()
            df.loc[idx, 'bb_mid'] = bb.bollinger_mavg()
            df.loc[idx, 'bb_high'] = bb.bollinger_hband()

    # 4. ATR avec ta
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

    # 5. MACD avec ta
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
    df['euro_volume'] = (df['adj close']*df['volume'])/1e6
    
    return df

def calculate_returns(df):
    """Calcule les rendements cumulés sur données MENSUELLES"""
    print("📈 Calcul des Returns Momentum...")
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    
    # NOTE: J'ai réduit min_periods de 252 à 12. 
    # Raison : Sur des données MENSUELLES, 252 signifie 21 ans d'historique requis.
    # Avec 10 ans téléchargés, cela donnerait des NaN partout. 12 = 1 an min.
    min_periods_monthly = 12 

    for lag in lags:
        returns_raw = df['adj close'].pct_change(lag)
        lower_bound = returns_raw.expanding(min_periods=min_periods_monthly).quantile(outlier_cutoff)
        upper_bound = returns_raw.expanding(min_periods=min_periods_monthly).quantile(1 - outlier_cutoff)
        df[f'return_{lag}m'] = returns_raw.clip(lower=lower_bound, upper=upper_bound)
    
    return df

def get_fama_french_betas(data):
    """Récupère les facteurs FF et calcule les Rolling Betas"""
    print("🌍 Récupération Fama-French Europe et calcul des Betas...")
    try:
        factor_data = web.DataReader('Europe_5_Factors', 'famafrench', start='2010')[0].drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        factor_data = factor_data.resample('M').last().div(100)
        factor_data.index.name = 'date'
        
        # Join
        # On s'assure que data a un index date/ticker pour le join
        data_ff = data.copy()
        if 'return_1m' not in data_ff.columns:
            print("⚠️ 'return_1m' manquant pour FF calc.")
            return data
            
        # Fusion
        # On reset l'index ticker pour merger sur la date
        factor_data_joined = factor_data.merge(data_ff['return_1m'].unstack('ticker'), on='date', how='right')
        
        # On recalcule proprement
        betas_list = []
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
        # Calcul par ticker
        for ticker in data_ff.index.get_level_values(1).unique():
             # Extraire série du ticker
             try:
                 y = data_ff.xs(ticker, level=1)['return_1m']
                 # Extraire facteurs correspondants aux dates
                 X = factor_data.loc[factor_data.index.intersection(y.index)]
                 y = y.loc[X.index] # Alignement
                 
                 if len(y) > 24: # Besoin d'historique
                     exog = sm.add_constant(X[factors])
                     rols = RollingOLS(y, exog, window=24)
                     rres = rols.fit()
                     params = rres.params.drop('const', axis=1)
                     params['ticker'] = ticker
                     betas_list.append(params)
             except Exception as e:
                 continue

        if not betas_list:
            print("⚠️ Calcul des betas échoué (pas assez de data ?)")
            return data

        betas_df = pd.concat(betas_list).set_index('ticker', append=True)
        
        # Join back to data
        # On doit décaler les betas (shift) car on utilise les betas passés pour prédire
        data = data.join(betas_df.groupby('ticker').shift())
        
        # FillNA avec la moyenne (important pour le jour J)
        data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
        
        return data

    except Exception as e:
        print(f"⚠️ Erreur Fama-French : {e}")
        # Fallback : Créer colonnes vides pour ne pas casser XGBoost
        for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            data[f] = 0.0
        return data

def get_data_pipeline():
    """Pipeline complet : Download -> Tech -> Monthly -> Macro -> Clean"""
    
    # 1. DOWNLOAD
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    
    print(f"⬇️ Téléchargement des données ({start_date} -> {end_date})...")
    df = yf.download(CAC40_TICKERS, start=start_date, end=end_date, progress=False, auto_adjust=False)
    # Gestion du MultiIndex de yfinance (Price, Ticker) -> (Date, Ticker)
    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    # Sécurité : Si 'adj close' n'existe pas, on utilise 'close'
    if 'adj close' not in df.columns and 'close' in df.columns:
        print("⚠️ 'adj close' manquant, utilisation de 'close' comme fallback.")
        df['adj close'] = df['close']
        
    # 2. INDICATEURS TECHNIQUES (Daily)
    df = compute_technical_indicators(df)
    
    # 3. RESAMPLING MENSUEL
    print("📅 Resampling Mensuel...")
    last_cols = [c for c in df.columns.unique(0) if c not in ['euro_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    # Logique spécifique du notebook fourni
    data_monthly = (pd.concat([
        df.unstack('ticker')['euro_volume'].resample('M').mean().stack('ticker').to_frame('euro_volume'),
        df.unstack()[last_cols].resample('M').last().stack('ticker')
    ], axis=1)).dropna()
    
    # 4. MOMENTUM RETURNS (Monthly)
    data_monthly = data_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
    
    # 5. FAMA-FRENCH (Monthly)
    data_monthly = get_fama_french_betas(data_monthly)
    
    # 6. LAG VARIABLES (Pour XGBoost)
    # Le modèle a été entraîné avec des variables _lag1
    vars_to_lag = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'euro_volume', 'garman_klass_vol']
    for col in vars_to_lag:
        if col in data_monthly.columns:
            data_monthly[f'{col}_lag1'] = data_monthly.groupby('ticker')[col].shift(1)
            
    # Nettoyage final
    data_monthly = data_monthly.drop(columns=['adj close'], errors='ignore')
    # On ne drop pas les NA tout de suite, on veut garder la dernière ligne (Today)
    # même si elle est incomplète sur le futur return
    
    return df, data_monthly # On retourne aussi df (Daily) pour Markowitz

# =============================================================================
# 3. OPTIMISATION MARKOWITZ
# =============================================================================
def get_optimal_weights(prices_df):
    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=252).ledoit_wolf()
        
        n_stocks = len(prices_df.columns)
        max_weight = max(0.25, 1.0 / n_stocks * 2.0)
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
        weights = ef.max_sharpe(risk_free_rate=0.03)
        return ef.clean_weights(), True
    except Exception as e:
        print(f"⚠️ Erreur Optimisation : {e}")
        return {}, False

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def run_pipeline():
    print("🚀 Démarrage du Daily Run...")
    
    # A. Chargement Modèles
    xgb_model, kmeans_model = load_models()
    if xgb_model is None: return

    # B. Récupération & Feature Engineering
    df_daily, df_monthly = get_data_pipeline()
    
    # C. Sélection de la dernière ligne (LE MOIS EN COURS / TODAY)
    # On prend la dernière date disponible dans les données mensuelles
    last_date = df_monthly.index.get_level_values('date').max()
    print(f"📅 Analyse basée sur les données du : {last_date.date()}")
    
    today_data = df_monthly.xs(last_date, level=0).copy()
    
    # D. Clustering
    # KMeans attend 'rsi'. On utilise les centroids fixes si besoin ou le modèle chargé.
    if 'rsi' in today_data.columns:
        # Nettoyage des NaNs pour le predict
        X_cluster = today_data[['rsi']].fillna(50) # FillNa 50 (Neutre) par sécurité
        today_data['cluster'] = kmeans_model.predict(X_cluster)
        
    # E. Prédiction XGBoost
    # On prépare les colonnes exactes attendues par XGBoost
    # Liste des features basée sur votre notebook
    feature_cols = [
        'rsi', 'macd', 'bb_low', 'bb_high', 'atr',          
        'return_2m', 'return_3m', 'return_6m',              
        'euro_volume_lag1', 'garman_klass_vol_lag1',        
        'Mkt-RF_lag1', 'SMB_lag1', 'HML_lag1', 'RMW_lag1', 'CMA_lag1',
        'cluster'
    ]
    
    # Vérification présence colonnes
    missing_cols = [c for c in feature_cols if c not in today_data.columns]
    if missing_cols:
        print(f"❌ Colonnes manquantes pour XGBoost: {missing_cols}")
        return

    X_pred = today_data[feature_cols].fillna(0) # Sécurité ultime
    
    print("🔮 Prédictions XGBoost...")
    today_data['proba_hausse'] = xgb_model.predict_proba(X_pred)[:, 1]
    
    # F. Filtrage (Stratégie)
    # Cluster 3 (Momentum) + Proba > 0.55 (par exemple)
    selected_stocks = today_data[
        (today_data['cluster'] == 3) & 
        (today_data['proba_hausse'] > 0.55)
    ]
    
    print(f"✅ Actions sélectionnées ({len(selected_stocks)}) : {selected_stocks.index.tolist()}")
    
    # G. Optimisation
    final_alloc = {}
    if not selected_stocks.empty:
        # On récupère les prix daily pour Markowitz (sur 1 an)
        tickers = selected_stocks.index.tolist()
        
        # yfinance multi-index clean up pour Markowitz
        prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-252:]
        prices_subset = prices_subset.dropna(axis=1) # On vire ceux qui ont des trous
        
        if not prices_subset.empty:
            weights, success = get_optimal_weights(prices_subset)
            if success:
                final_alloc = weights
            else:
                # Équipondéré
                final_alloc = {t: 1.0/len(prices_subset.columns) for t in prices_subset.columns}
    
    # H. Sauvegarde
    print("💾 Sauvegarde des résultats...")
    
    # 1. Signaux (Latest)
    signals_df = pd.DataFrame(columns=['Ticker', 'Cluster', 'Proba_Hausse', 'Allocation'])
    if not today_data.empty:
        # On prépare tout le monde
        export_df = today_data[['cluster', 'proba_hausse']].reset_index()
        export_df.columns = ['Ticker', 'Cluster', 'Proba_Hausse']
        export_df['Allocation'] = export_df['Ticker'].map(final_alloc).fillna(0.0)
        export_df['Signal'] = np.where(export_df['Allocation'] > 0, 'ACHAT', 'NEUTRE')
        
        export_df.to_csv('latest_signals.csv', index=False)
        print(" -> latest_signals.csv OK")

    # 2. Historique (Append simulation)
    # Pour l'historique, on va juste ajouter une ligne dummy ou la vraie perf si on trackait
    # Ici on met juste à jour le fichier pour dire "le script a tourné"
    try:
        hist_df = pd.read_csv('portfolio_history.csv', index_col=0, parse_dates=True)
    except:
        hist_df = pd.DataFrame(columns=['Strategy', 'Benchmark'])
        
    # On n'ajoute pas de perf fictive ici pour ne pas fausser le graph, 
    # mais vous pourriez calculer la perf de la veille ici.
    
    print("🏁 Pipeline terminé avec succès !")

if __name__ == "__main__":
    run_pipeline()