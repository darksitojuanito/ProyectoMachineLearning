import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_clean_data
from src.preprocessor import TextPreprocessor
from src.similarity import SimilarityEngine
from src.search_engine import SearchEngine
from src.recommender import Recommender
from src.query_processor import build_vocabulary, process_query

# Emdedding engine imports
try:
    from src.embedding_engine import (
        load_embedding_model,
        load_embeddings,
        search_by_embeddings,
        recommend_by_embeddings
    )
except ImportError:
    pass

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

df = None
preprocessor = None
final_matrix = None
search_engine = None
recommender = None
vocabulary = None

# Embeddings state
embedding_model = None
abstract_embeddings = None
embedding_similarity_matrix = None

def load_system():
    global df, preprocessor, final_matrix, search_engine, recommender, vocabulary
    global embedding_model, abstract_embeddings, embedding_similarity_matrix
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_file = os.path.join(data_dir, 'DataSetArticulos.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run main.py first to generate it.")
        return
        
    print("Loading dataset...")
    df = load_and_clean_data(input_file)
    
    print("Initializing NLP preprocessor...")
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df, data_dir)
    
    print("Building vocabulary for query processing...")
    vocabulary = build_vocabulary(df)
    
    print("Building similarity matrices...")
    sim_engine = SimilarityEngine(w_title=0.10, w_keywords=0.20, w_abstract=0.70)
    final_matrix = sim_engine.build_and_save_matrices(df, data_dir)
    
    search_engine = SearchEngine(preprocessor, w_title=0.10, w_keywords=0.20, w_abstract=0.70)
    recommender = Recommender(final_matrix, df)
    
    print("Loading embeddings subsystem (Phase 3)...")
    emb_model_path = os.path.join(base_dir, 'models', 'all-MiniLM-L6-v2')
    emb_npy_path = os.path.join(data_dir, 'abstract_embeddings.npy')
    emb_matrix_path = os.path.join(data_dir, 'matriz_embeddings_coseno.csv')
    
    if os.path.exists(emb_npy_path) and os.path.exists(emb_matrix_path):
        print("Embeddings files found. Loading model and matrices into memory...")
        embedding_model = load_embedding_model(emb_model_path)
        abstract_embeddings = load_embeddings(emb_npy_path)
        embedding_similarity_matrix = pd.read_csv(emb_matrix_path).values
        print("Embeddings subsystem loaded successfully.")
    else:
        print("Embeddings files NOT found. Run scripts/generate_embeddings.py to enable Phase 3.")
        
    print("System loaded successfully.")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/search', methods=['POST'])
def search_api():
    if df is None:
        return jsonify({"error": "System not loaded."}), 500
        
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter."}), 400
        
    original_query = data.get("query", "").strip()
    if not original_query:
        return jsonify({"error": "La consulta no puede estar vacía."}), 400
    
    query_data = process_query(original_query, vocabulary)
    processed_query = query_data['processed_query']
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    top10_df = search_engine.search(processed_query, df, data_dir)
    
    if top10_df.empty:
        return jsonify({
            "original_query": original_query, 
            "processed_query": processed_query,
            "query_processing": query_data,
            "top10": []
        })
        
    rec_df = recommender.recommend(top10_df, data_dir)
    
    top10_list = []
    for _, row in top10_df.iterrows():
        paper_id = row['paper_id']
        paper_recs = rec_df[rec_df['source_paper_id'] == paper_id]
        
        rec_list = []
        for _, rec in paper_recs.iterrows():
            rec_list.append({
                "paper_id": int(rec['recommended_paper_id']),
                "title": rec['recommended_title'],
                "session": rec['recommended_session'],
                "similarity_score": round(float(rec['similarity_score']), 4)
            })
            
        top10_list.append({
            "rank": int(row['rank']),
            "paper_id": int(paper_id),
            "title": row['Title'],
            "keywords": row['Keywords'],
            "abstract": row['Abstract'],
            "session": str(row['Session']),
            "year": int(row['year']) if pd.notna(row['year']) else None,
            "score_title": round(float(row['score_title']), 4),
            "score_keywords": round(float(row['score_keywords']), 4),
            "score_abstract": round(float(row['score_abstract']), 4),
            "score_final": round(float(row['score_final']), 4),
            "recommendations": rec_list
        })
        
    return jsonify({
        "original_query": original_query,
        "processed_query": processed_query,
        "query_processing": {
            "was_translated": query_data['was_translated'],
            "was_corrected": query_data['was_corrected'],
            "used_wildcards": query_data['used_wildcards'],
            "corrections": query_data['corrections'],
            "wildcard_expansions": query_data['wildcard_expansions']
        },
        "recommendation_policy": "Recommendations are computed using document-to-document similarity and exclude the current Top 10 search results.",
        "weights": {
            "title": 0.10,
            "keywords": 0.20,
            "abstract": 0.70
        },
        "top10": top10_list
    })

@app.route('/api/search-embeddings', methods=['POST'])
def search_embeddings_api():
    if df is None:
        return jsonify({"error": "System not loaded."}), 500
        
    if embedding_model is None or abstract_embeddings is None or embedding_similarity_matrix is None:
        return jsonify({"error": "Los embeddings no han sido generados. Ejecuta primero: python scripts/generate_embeddings.py"}), 500
        
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter."}), 400
        
    original_query = data.get("query", "").strip()
    if not original_query:
        return jsonify({"error": "La consulta no puede estar vacía."}), 400
        
    # Intercept query to process intelligent translations/corrections
    query_data = process_query(original_query, vocabulary)
    processed_query = query_data['processed_query']
    
    # Embedding search
    top10_df = search_by_embeddings(processed_query, df, embedding_model, abstract_embeddings, top_n=10)
    
    if top10_df.empty:
        return jsonify({
            "query": original_query,
            "original_query": original_query, 
            "processed_query": processed_query,
            "query_processing": query_data,
            "top10": []
        })
        
    # Recommendations
    recs_df = recommend_by_embeddings(top10_df, df, embedding_similarity_matrix, top_k=3)
    
    top10_list = []
    for _, row in top10_df.iterrows():
        paper_id = row['paper_id']
        paper_recs = recs_df[recs_df['source_paper_id'] == paper_id]
        
        rec_list = []
        for _, rec in paper_recs.iterrows():
            rec_list.append({
                "paper_id": int(rec['recommended_paper_id']),
                "title": rec['recommended_title'],
                "session": rec['recommended_session'],
                "similarity_score": round(float(rec['similarity_score']), 4)
            })
            
        top10_list.append({
            "rank": int(row['rank']),
            "paper_id": int(paper_id),
            "title": row['Title'],
            "abstract": row['Abstract'],
            "session": str(row['Session']),
            "year": int(row['year']) if pd.notna(row['year']) else None,
            "embedding_score": round(float(row['embedding_score']), 4),
            "recommendations": rec_list
        })
        
    return jsonify({
        "query": original_query,
        "original_query": original_query,
        "processed_query": processed_query,
        "query_processing": {
            "was_translated": query_data['was_translated'],
            "was_corrected": query_data['was_corrected'],
            "used_wildcards": query_data['used_wildcards'],
            "corrections": query_data['corrections'],
            "wildcard_expansions": query_data['wildcard_expansions']
        },
        "method": "embeddings",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Search based only on Abstract embeddings using cosine similarity.",
        "top10": top10_list
    })

if __name__ == '__main__':
    load_system()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
