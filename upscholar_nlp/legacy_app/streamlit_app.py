import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_clean_data
from src.preprocessor import TextPreprocessor
from src.similarity import SimilarityEngine
from src.search_engine import SearchEngine
from src.recommender import Recommender

# Set page config for aesthetics
st.set_page_config(
    page_title="UPScholar",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        padding-bottom: 2rem;
    }
    .result-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .result-meta {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .score-badge {
        background-color: #e0e7ff;
        color: #4338ca;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .rec-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #e2e8f0;
    }
    .rec-title {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    data_dir = "data"
    input_file = f"{data_dir}/DataSetCompleto.csv"
    
    if not os.path.exists(input_file):
        return None, None, None, None, None
        
    df = load_and_clean_data(input_file)
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df, data_dir)
    
    sim_engine = SimilarityEngine()
    final_matrix = sim_engine.build_and_save_matrices(df, data_dir)
    
    search_engine = SearchEngine(preprocessor)
    recommender = Recommender(final_matrix, df)
    
    return df, preprocessor, final_matrix, search_engine, recommender

st.markdown("<h1 class='main-header'>📚 UPScholar - Buscador Inteligente de Documentos Científicos</h1>", unsafe_allow_html=True)

with st.spinner('Cargando el sistema y calculando matrices (esto puede tardar unos segundos la primera vez)...'):
    df, preprocessor, final_matrix, search_engine, recommender = load_system()

if df is None:
    st.error("No se encontró el dataset en data/DataSetCompleto.csv")
    st.stop()

# Search UI
st.write("### Encuentra artículos relevantes y descubre nuevas recomendaciones")
query = st.text_input("Ingresa tu consulta de búsqueda:", placeholder="Ej: deep learning medical image segmentation")
search_button = st.button("Buscar", type="primary")

if search_button and query:
    with st.spinner('Buscando...'):
        top10_df = search_engine.search(query, df, "data")
        rec_df = recommender.recommend(top10_df, "data")
        
        st.write(f"### Resultados principales para: *{query}*")
        
        for _, row in top10_df.iterrows():
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">#{row['rank']} - {row['Title']}</div>
                <div class="result-meta">
                    <b>Sesión:</b> {row['Session']} | <b>Año:</b> {row['year']}<br>
                    <b>Keywords:</b> {row['Keywords'] if row['Keywords'] else 'N/A'}
                </div>
                <p><b>Abstract:</b> {row['Abstract'][:300]}...</p>
                <div>
                    <span class="score-badge">Score Final: {row['score_final']:.4f}</span>
                    <span style="font-size: 0.8rem; color: #64748b; margin-left: 10px;">
                        (Title: {row['score_title']:.4f} | Keywords: {row['score_keywords']:.4f} | Abstract: {row['score_abstract']:.4f})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show recommendations
            recs = rec_df[rec_df['source_paper_id'] == row['paper_id']]
            if not recs.empty:
                with st.expander(f"Ver 3 Artículos Recomendados basados en este resultado"):
                    for _, rec in recs.iterrows():
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-title">💡 {rec['recommended_title']}</div>
                            <div class="result-meta" style="margin-bottom: 5px;">Sesión: {rec['recommended_session']}</div>
                            <span class="score-badge" style="background-color: #f1f5f9; color: #475569;">
                                Similitud: {rec['similarity_score']:.4f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
