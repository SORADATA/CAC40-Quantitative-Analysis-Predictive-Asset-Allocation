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
from datetime import datetime
import pytz

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AlphaEdge | CAC40 Smart Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un look "Hedge Fund" (Dark Mode Tech)
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
# 2. CHARGEMENT DES DONNÉES (MODE RÉEL VIA GITHUB)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900)  # Cache de 15 min
def load_data():
    """
    Charge les VRAIES données générées par GitHub Actions.
    """
    # --- URL OFFICIELLE BASÉE SUR TON REPO ---
    base_url = "https://raw.githubusercontent.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation/main/"

    # A. Historique du Portfolio
    try:
        history_url = base_url + "portfolio_history.csv"
        history_df = pd.read_csv(history_url, index_col=0, parse_dates=True)
        history_df.index.name = 'Date'
        history_df = history_df.sort_index()
    except Exception as e:
        st.warning(f"⚠️ Historique non chargé (Attente du premier run): {e}")
        history_df = pd.DataFrame(columns=['Strategy', 'Benchmark'])

    # B. Derniers Signaux
    try:
        signals_url = base_url + "latest_signals.csv"
        signals_df = pd.read_csv(signals_url)
    except Exception as e:
        st.warning(f"⚠️ Signaux non chargés : {e}")
        signals_df = pd.DataFrame()

    return history_df, signals_df

# Chargement des données
history_df, latest_signals = load_data()

# Calcul des métriques en temps réel
if not history_df.empty and len(history_df) > 1:
    cum_strat = history_df['Strategy'].iloc[-1] - 1
    cum_bench = history_df['Benchmark'].iloc[-1] - 1
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
st.sidebar.title("🤖 AlphaML")
st.sidebar.markdown("**CAC40 Quantitative Optimizer**")
st.sidebar.caption("*Powered by XGBoost + K-Means + Markowitz*")
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


page = st.sidebar.radio("Navigation", ["🏠 Dashboard & Performance", "🚀 Signaux du Jour", "⚙️ Détails du Modèle"])

st.sidebar.markdown("---")
st.sidebar.success("✅ **Status Système** : EN LIGNE")
st.sidebar.info(f"📅 **Date Données** : {datetime.now().strftime('%d/%m/%Y')}")

# -----------------------------------------------------------------------------
# PAGE 1 : DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard & Performance":
    st.title("📊 Performance Live")
    
    if history_df.empty or len(history_df) < 2:
        st.info("👋 Bienvenue ! Le système est initialisé. L'historique s'affichera après la prochaine mise à jour quotidienne.")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.metric("Alpha (vs Bench)", f"{alpha:.1%}", delta=f"{alpha*100:.1f} pts")
    with kpi2: st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="Risk Adj.")
    with kpi3: st.metric("Max Drawdown", f"{max_dd:.1%}", delta_color="inverse")
    with kpi4: st.metric("Gain Total", f"{cum_strat:.1%}", delta="Net de frais")

    st.markdown("---")

    if not history_df.empty:
        st.subheader("📈 Évolution de la Stratégie")
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df['Benchmark'],
            mode='lines', name='Benchmark (CAC40)',
            line=dict(color='gray', width=1.5)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=history_df.index, y=history_df['Strategy'],
            mode='lines', name='Stratégie Hybride',
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
            st.subheader("📉 Analyse du Drawdown")
            if not drawdown.empty:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown,
                    fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.5)',
                    line=dict(color='#E74C3C', width=1), name='Drawdown'
                ))
                fig_dd.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_dd, use_container_width=True)

        with col_dist:
            st.subheader("📊 Distribution des Rendements")
            if not daily_ret.empty:
                fig_hist = px.histogram(daily_ret.dropna(), nbins=50, color_discrete_sequence=['#4ECDC4'])
                fig_hist.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2 : SIGNAUX
# -----------------------------------------------------------------------------
elif page == "🚀 Signaux du Jour":
    st.title("🚀 Signaux Générés par l'IA")
    st.markdown("Ces signaux sont générés chaque soir à **18:00 UTC** par GitHub Actions.")
    
    if latest_signals.empty:
        st.warning("⚠️ Aucun signal disponible (vérifiez que daily_run.py a bien tourné sur GitHub).")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Liste de Surveillance (Cluster 3)")
            
            def highlight_signal(val):
                if val == 'ACHAT': return 'color: #2ecc71; font-weight: bold'
                if val == 'VENTE': return 'color: #e74c3c; font-weight: bold'
                return 'color: #95a5a6'

            st.dataframe(
                latest_signals.style.map(highlight_signal, subset=['Signal'])
                .format({'Proba_Hausse': '{:.1%}', 'Allocation': '{:.1%}'})
                .background_gradient(subset=['Proba_Hausse'], cmap='Greens'),
                use_container_width=True,
                height=500
            )
            
        with col2:
            st.subheader("🍰 Allocation Actuelle")
            portfolio_active = latest_signals[latest_signals['Allocation'] > 0]
            
            if not portfolio_active.empty:
                fig_pie = px.pie(
                    portfolio_active, values='Allocation', names='Ticker',
                    hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("😴 Le portefeuille est actuellement 100% Cash.")

# -----------------------------------------------------------------------------
# PAGE 3 : DÉTAILS
# -----------------------------------------------------------------------------
elif page == "⚙️ Détails du Modèle":
    st.title("⚙️ Architecture du Modèle")
    
    st.markdown("""
    ### 🧠 Approche "Hybride"
    Ce modèle combine Machine Learning et Finance Quantitative :
    """)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Filtrage (XGBoost)**")
        st.markdown("Prédit la probabilité de hausse à 1 mois (Target > 0%). Seuil de confiance : **55-60%**.")
    with c2:
        st.warning("**2. Profilage (K-Means)**")
        st.markdown("Clusterise les actions par régime de marché. On cible le **Cluster 3 (Momentum)**.")
    with c3:
        st.success("**3. Allocation (Markowitz)**")
        st.markdown("Optimise les poids (Max Sharpe Ratio) avec contrainte de diversification (Max 25%).")

    st.markdown("---")
    
    # --- AJOUT : SECTION PERFORMANCE DU MODÈLE (JSON) ---
    st.subheader("📊 Performance du Modèle (Backtest)")
    st.markdown("Métriques issues de la validation croisée (GridSearch) sur données de test.")

    # Chargement du JSON depuis GitHub
    @st.cache_data(ttl=900)
    def load_metrics():
        base_url = "https://raw.githubusercontent.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation/main/"
        try:
            # Attention au chemin : src/models/metrics.json (Pluriel)
            url = base_url + "src/models/metrics.json"
            metrics = pd.read_json(url, typ='series')
            return metrics
        except Exception as e:
            return None

    metrics = load_metrics()

    if metrics is not None:
        # Affichage des KPIs
        k1, k2, k3 = st.columns(3)
        
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('auc_score', 0)
        date_train = metrics.get('training_date', 'Inconnue')

        with k1:
            st.metric("Accuracy (Test)", f"{accuracy:.1%}", delta="vs Random (50%)")
        with k2:
            st.metric("ROC AUC Score", f"{auc:.3f}", delta="Qualité Discriminante")
        with k3:
            st.metric("Dernier Entraînement", date_train)
            
        # Affichage des hyperparamètres
        with st.expander("🔍 Voir les Hyperparamètres Optimaux (Best Params)"):
            st.json(metrics['best_params'])
            
    else:
        st.warning("⚠️ Les métriques du modèle (metrics.json) ne sont pas encore disponibles sur GitHub.")

    st.markdown("---")
    st.subheader("3. Analyse Fondamentale des Clusters (Simulation)")
    
    # Simulation pédagogique
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
        title="Distribution des Rendements par Cluster (Théorique)",
        color_discrete_map={'0 - Defensive 🛡️': '#95a5a6', '1 - Value 💰': '#3498db', '2 - Growth 📈': '#f1c40f', '3 - Momentum 🚀': '#2ecc71'}
    )
    fig_box.update_layout(template="plotly_dark", showlegend=False, height=450)
    st.plotly_chart(fig_box, use_container_width=True)