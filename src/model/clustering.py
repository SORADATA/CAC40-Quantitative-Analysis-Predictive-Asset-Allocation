import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import os

# --- CHEMINS ABSOLUS STATIQUES ---
PROJECT_ROOT_ABSOLUTE = "/home/onyxia/work/Gestion-portefeuille/" 

try:
    ROOT_DIR = Path(PROJECT_ROOT_ABSOLUTE)
except Exception:
    ROOT_DIR = Path.cwd() 

INTERIM_DATA_PATH = ROOT_DIR / "data" / "interim"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"
INPUT_FILENAME = "cac40_interim_features.csv"
OUTPUT_FILENAME = "cac40_clustered_data.csv"
K_CLUSTERS = 4 # Nombre de clusters cible (à ajuster après la méthode du coude)


def load_data_for_clustering(input_path: Path) -> pd.DataFrame:
    """ Charge les données enrichies et fusionnées. """
    filepath = input_path / INPUT_FILENAME
    if not filepath.exists():
        print(f"❌ Erreur : Fichier d'entrée non trouvé à {filepath}. Relancez l'étape 2.")
        return pd.DataFrame()
    return pd.read_csv(filepath)


def prepare_data_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """ Prépare, sélectionne la dernière date, sélectionne les features et normalise. """
    
    # 1. Sélectionner la dernière date disponible (Scoring actualisé)
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    df_latest = df[df['Date'] == latest_date].copy().set_index('Ticker')
    
    # 2. Définir les features pour la segmentation
    # Features clés de Risque, Performance et Taille (Volume est une proxy de la taille/liquidité)
    FEATURES = ['Volatility', 'Sharpe_Ratio_20D', 'Performance_20D', 'Volume', 'Dividends']
    
    X = df_latest[FEATURES].copy()
    
    # 3. Nettoyage des NaN/Infinites (crucial pour K-means)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Imputation simple par la moyenne (puisque ce sont des indicateurs de performance)
    X = X.fillna(X.mean()) 

    # 4. Normalisation (StandardScaler)
    # ESSENTIEL pour que Volatility et Volume n'écrasent pas les autres features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=X.index)
    
    return X_scaled, df_latest, scaler


def run_clustering(k_clusters: int = K_CLUSTERS):
    """ Orchestre le chargement, le clustering et la sauvegarde. """
    print("🚀 Étape 3 : Clustering et Segmentation (K-Means)...")

    # 1. Chargement et préparation des données
    df_interim = load_data_for_clustering(INTERIM_DATA_PATH)
    if df_interim.empty: return

    X_scaled, df_latest, scaler = prepare_data_and_scale(df_interim)
    if X_scaled.empty: 
        print("❌ Processus interrompu: Aucune donnée restante après nettoyage.")
        return

    # 2. Application du K-means
    optimal_k = k_clusters
    print(f"🔧 Application de K-Means avec K = {optimal_k}.")

    try:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_latest['Cluster'] = kmeans.fit_predict(X_scaled)
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution de K-Means: {e}.")
        return

    # 3. Calcul des profils moyens des clusters (le SCORING final)
    cluster_profiles = df_latest.groupby('Cluster')[
        ['Volatility', 'Sharpe_Ratio_20D', 'Performance_20D', 'Dividends']
    ].mean().sort_values(by='Volatility', ascending=False)
    
    print("\n✅ Profils Moyens des Clusters (Scoring) :")
    print(cluster_profiles)
    
    # 4. Sauvegarde des données finales (pour Superset)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_filepath = PROCESSED_DATA_PATH / OUTPUT_FILENAME
    
    df_latest.to_csv(output_filepath, index=True)
    print(f"\n💾 Clustering terminé. Données sauvées vers : {output_filepath}")


if __name__ == "__main__":
    run_clustering()