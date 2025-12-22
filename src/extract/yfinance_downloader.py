import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


# --- CHEMIN ABSOLU STATIQUE POUR ONYXIA ---
# 🛑 ATTENTION : Ce chemin doit correspondre EXACTEMENT à la racine de votre projet.
PROJECT_ROOT_ABSOLUTE = "/home/onyxia/work/Gestion-portefeuille/" 

try:
    ROOT_DIR = Path(PROJECT_ROOT_ABSOLUTE)
    # Chemin absolu de sauvegarde des fichiers bruts (data/raw/)
    OUTPUT_PATH_FINAL = ROOT_DIR / "data" / "raw"
except Exception as e:
    print(f"Erreur de chemin statique: {e}. Utilisation du chemin courant.")
    OUTPUT_PATH_FINAL = Path.cwd() / "data" / "raw"


# 1. Liste des tickers CAC40 (avec corrections STLA.MI)
CAC40_TICKERS = [
    "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "ATO.PA", "CS.PA", "BNP.PA",
    "EN.PA", "CAP.PA", "CA.PA", "DSY.PA", "EL.PA", "ENGI.PA", "ERF.PA",
    "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA", "ORA.PA",
    "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA",
    "GLE.PA", "STLA.MI", "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA",
    "URW.AS", "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA", "FR.PA"
]


def download_cac40_history(start="2010-01-01", end=None, interval="1d"):
    """ Télécharge les données historiques pour toutes les entreprises du CAC40. """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    
    all_data = {}

    for ticker in CAC40_TICKERS:
        try:
            print(f"📥 Téléchargement : {ticker}")
            # On passe auto_adjust à True pour capturer la performance réelle (dividendes inclus
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
            
            if df.empty:
                print(f"⚠️ Aucune donnée pour {ticker}")
                continue
            
            # --- CORRECTION CLE : Transforme l'index 'Date' en colonne nommée 'Date' ---
            df = df.reset_index(names=['Date']) 
            
            df["Ticker"] = ticker
            df["Date"] = df["Date"].dt.tz_localize(None) # Supprime le timezone
            
            all_data[ticker] = df

        except Exception as e:
            print(f"❌ Erreur sur {ticker} : {e}")
            continue

    return all_data


def save_to_csv(data_dict):
    """ Sauvegarde chaque DataFrame dans un fichier CSV individuel en utilisant le chemin absolu. """
    output_path_str = str(OUTPUT_PATH_FINAL) 
    
    # Création du dossier de sauvegarde
    os.makedirs(output_path_str, exist_ok=True) 
    print(f"Création du dossier de sauvegarde : {output_path_str}")

    for ticker, df in data_dict.items():
        # Construit le chemin complet du fichier
        filename = os.path.join(output_path_str, f"{ticker.replace('.','_')}.csv")
        # On sauvegarde SANS index car la date est maintenant une colonne nommée
        df.to_csv(filename, index=False) 
        print(f"💾 Sauvegardé : {filename}")


if __name__ == "__main__":
    print("🚀 Lancement du téléchargement CAC40...")
    data = download_cac40_history(start="2015-01-01")
    save_to_csv(data) 
    print("✔️ Terminé !")