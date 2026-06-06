import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_engine import (
    load_embedding_model,
    load_articles_dataset,
    load_embeddings,
    search_by_embeddings,
    recommend_by_embeddings
)
import pandas as pd

def test_search():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'all-MiniLM-L6-v2')
    data_path = os.path.join(base_dir, 'data', 'DataSetArticulos.csv')
    emb_path = os.path.join(base_dir, 'data', 'abstract_embeddings.npy')
    matrix_path = os.path.join(base_dir, 'data', 'matriz_embeddings_coseno.csv')
    
    print("Loading system...")
    df = load_articles_dataset(data_path)
    model = load_embedding_model(model_path)
    embeddings = load_embeddings(emb_path)
    sim_matrix = pd.read_csv(matrix_path).values
    
    print("\n--- UPScholar Embedding Search Test ---")
    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            print("La consulta no puede estar vacía. Intenta nuevamente.")
            continue
            
        print("\nSearching...")
        top10 = search_by_embeddings(query, df, model, embeddings, top_n=10)
        
        print("\n--- TOP 10 RESULTS ---")
        for _, row in top10.iterrows():
            print(f"[{row['rank']}] ID:{row['paper_id']} Score:{row['embedding_score']:.4f} - {row['Title']}")
            
        print("\nCalculating recommendations...")
        recs = recommend_by_embeddings(top10, df, sim_matrix, top_k=3)
        
        print("\n--- RECOMMENDATIONS ---")
        for source_id in top10['paper_id']:
            source_recs = recs[recs['source_paper_id'] == source_id]
            print(f"\nFor Paper ID {source_id}:")
            for _, rec in source_recs.iterrows():
                print(f"  -> ID:{rec['recommended_paper_id']} Score:{rec['similarity_score']:.4f} - {rec['recommended_title']}")
                
        out_top10 = os.path.join(base_dir, 'data', 'top10_embeddings.csv')
        out_recs = os.path.join(base_dir, 'data', 'recomendaciones_embeddings.csv')
        
        top10.to_csv(out_top10, index=False)
        recs.to_csv(out_recs, index=False)
        print(f"\nResults saved to {out_top10} and {out_recs}")

if __name__ == "__main__":
    test_search()
