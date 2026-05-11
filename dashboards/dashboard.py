import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Shto direktorinë root në sys.path për të importuar utils nga dashboards
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dashboards.utils import (
    load_spotify_data, 
    compute_yearly_averages, 
    get_audio_features, 
    get_comparison_metrics
)

# Konfigurimi i faqes
st.set_page_config(page_title="Spotify Explorer", layout="wide", page_icon="🎧")
st.title("🎧 Evolucioni i Muzikës në Spotify (1980-2025)")
st.markdown("Ky dashboard interaktiv tregon se si kanë ndryshuar tiparet e muzikës (Audio Features) ndër dekada, së bashku me analizën e modeleve dhe grupimit (Clustering).")

# Ngarkimi i të dhënave duke përdorur utils
try:
    df = load_spotify_data()
    audio_features = get_audio_features()
    comparison_metrics = get_comparison_metrics()
    yearly_avg = compute_yearly_averages(df, audio_features)
except Exception as e:
    st.error(f"Gabim gjatë ngarkimit të të dhënave: {e}. Sigurohuni që file-i i pastruar ekziston.")
    st.stop()

# Ndarja në Tabs për organizim më të mirë
tab1, tab2, tab3 = st.tabs(["📊 Eksplorimi i Trendeve", "🎵 Scatter Plot", "🤖 Modelimi & Clustering"])

# --- TAB 1: Eksplorimi i Trendeve ---
with tab1:
    st.sidebar.header("Opsionet e Grafikëve")
    st.sidebar.markdown("Përdor menunë për të filtruar të dhënat.")

    selected_features = st.sidebar.multiselect(
        "Zgjidh Karakteristikat Audio për t'i parë në kohë:",
        audio_features,
        default=['danceability', 'energy']
    )

    st.subheader("📈 Trendi i Karakteristikave Ndër Vite")
    if selected_features:
        fig_line = px.line(
            yearly_avg, x='year', y=selected_features, 
            title="Ndryshimi i Tipareve Audio ndër vite",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Të lutem zgjidh të paktën një karakteristikë nga menuja majtas.")

# --- TAB 2: Eksplorimi i Këngëve ---
with tab2:
    st.subheader("🎵 Zbulo Këngët (Scatter Plot Interaktiv)")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Zgjidh boshtin X:", audio_features, index=1)
    with col2:
        y_axis = st.selectbox("Zgjidh boshtin Y:", comparison_metrics)

    # Marrja e kampionit dhe drop na
    sample_df = df.dropna(subset=[x_axis, y_axis])
    
    # Këtu mund të modifikosh sampling logic (p.sh. stratifikim)
    if len(sample_df) > 5000:
        sample_df = sample_df.sample(n=5000, random_state=42)

    if len(sample_df) < 100:
        st.warning("⚠️ Shumë pak këngë me këto karakteristika. Provo kombinime të tjera!")
    else:
        fig_scatter = px.scatter(
            sample_df, x=x_axis, y=y_axis, color='year', 
            hover_data=['name', 'artists', 'year'], 
            title=f"{y_axis.capitalize()} vs {x_axis.capitalize()} ({len(sample_df)} këngë)",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: Modelimi & Clustering ---
with tab3:
    st.subheader("🤖 Analiza e Modeleve & Clustering")
    st.markdown("Integrimi i rezultateve nga KMeans dhe modeleve të Machine Learning.")
    
    # Logjika e Clustering
    cluster_file = os.path.join(project_root, 'reports', 'clustering', 'clustered_tracks.csv')
    if os.path.exists(cluster_file):
        st.write("### Grupimet e Këngëve (KMeans Clusters)")
        cluster_df = pd.read_csv(cluster_file)
        
        if 'cluster' in cluster_df.columns:
            cluster_df['cluster'] = cluster_df['cluster'].astype(str)
            
            sample_cluster_df = cluster_df.sample(min(2000, len(cluster_df)), random_state=42)
            fig_cluster = px.scatter_3d(
                sample_cluster_df, x='energy', y='danceability', z='valence',
                color='cluster', hover_data=['name', 'artists'],
                title="Këngët e grupuara sipas Energjisë, Kërcyeshmërisë dhe Valencës"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    elif 'cluster' in df.columns or 'kmeans_cluster' in df.columns:
        st.write("### Grupimet e Këngëve (KMeans Clusters)")
        cluster_col = 'cluster' if 'cluster' in df.columns else 'kmeans_cluster'
        df[cluster_col] = df[cluster_col].astype(str)
        fig_cluster = px.scatter_3d(
            df.sample(min(2000, len(df))), x='energy', y='danceability', z='valence',
            color=cluster_col, hover_data=['name', 'artists']
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("💡 Skedari i Clustering nuk u gjet. Ekzekutoni fillimisht: `python -m src.models.clustering_analysis`")

    st.divider()

    # Logjika e Feature Importance
    st.write("### Rëndësia e Veçorive (Feature Importance)")
    importance_file = os.path.join(project_root, 'reports', 'popularity_model', 'popularity_feature_importance.csv')
    
    if os.path.exists(importance_file):
        importance_df = pd.read_csv(importance_file)
        importance_df = importance_df.sort_values(by='importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df, x='importance', y='feature', orientation='h',
            title="Cilat tipare ndikojnë më shumë në popullaritet?",
            color='importance', color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("💡 Skedari i Feature Importance nuk u gjet. Për të shfaqur këtë grafik, ekzekutoni fillimisht: `python -m src.models.train_popularity_model`")
