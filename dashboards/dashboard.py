import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurimi i faqes
st.set_page_config(page_title="Spotify Explorer", layout="wide")
st.title("🎧 Evolucioni i Muzikës në Spotify (1980-2025)")
st.markdown("Ky dashboard interaktiv tregon se si kanë ndryshuar tiparet e muzikës (Audio Features) ndër dekada.")

# Funksioni për ngarkimin e të dhënave (përdor cache që të jetë super i shpejtë)
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/Spotify_1980_2025_Final_clean.csv")
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'liveness', 'speechiness']
    
    # Llogarisim mesataren e tipareve për çdo vit
    yearly_avg = df.groupby('year')[audio_features].mean().reset_index()
    return df, yearly_avg, audio_features

df, yearly_avg, audio_features = load_data()

# --- PJESA 1: Trendet Interaktive ---
st.sidebar.header("Opsionet e Grafikëve")
st.sidebar.markdown("Përdor menunë për të filtruar të dhënat.")

selected_features = st.sidebar.multiselect(
    "Zgjidh Karakteristikat Audio për t'i parë në kohë:",
    audio_features,
    default=['danceability', 'energy']
)

st.subheader("📈 Trendi i Karakteristikave Ndër Vite")
if selected_features:
    # Krijojmë një grafik interaktiv me Plotly
    fig_line = px.line(
        yearly_avg, x='year', y=selected_features, 
        title="Ndryshimi i Tipareve Audio (Zmadho dhe shiko vlerat duke kaluar mausin)",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("Të lutem zgjidh të paktën një karakteristikë nga menuja majtas.")

st.divider()

# --- PJESA 2: Eksplorimi i Këngëve ---
st.subheader("🎵 Zbulo Këngët (Scatter Plot Interaktiv)")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("Zgjidh boshtin X:", audio_features, index=1) # Default: energy
with col2:
    y_axis = st.selectbox("Zgjidh boshtin Y:", ['popularity', 'duration_ms', 'tempo']) # Default: popularity

# Marrim një kampion (sample) dhe filtroni vlerat null
sample_df = df.dropna(subset=[x_axis, y_axis])  # Remove rows with missing values
if len(sample_df) > 5000:
    sample_df = sample_df.sample(n=5000, random_state=42)

if len(sample_df) < 100:
    st.warning(f"⚠️ Shumë pak këngë me këto karakteristika ({len(sample_df)} këngë). Provo kombinime të tjera!")
else:
    fig_scatter = px.scatter(
        sample_df, x=x_axis, y=y_axis, color='year', 
        hover_data=['name', 'artists', 'year'], # Kur kalon mausin, tregon emrin e këngës!
        title=f"{y_axis.capitalize()} vs {x_axis.capitalize()} ({len(sample_df)} këngë)",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
