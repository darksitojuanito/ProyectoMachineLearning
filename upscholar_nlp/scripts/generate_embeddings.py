import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_engine import (
    load_embedding_model,
    load_articles_dataset,
    generate_abstract_embeddings,
    save_embeddings,
    save_embedding_metadata,
    build_embedding_similarity_matrix,
    save_similarity_matrix
)

def generate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'all-MiniLM-L6-v2')
    data_path = os.path.join(base_dir, 'data', 'DataSetArticulos.csv')
    
    print("1. Loading dataset...")
    df = load_articles_dataset(data_path)
    
    print("2. Loading model...")
    model = load_embedding_model(model_path)
    
    print("3. Generating embeddings...")
    embeddings = generate_abstract_embeddings(df, model)
    
    print("4. Saving embeddings...")
    emb_path = os.path.join(base_dir, 'data', 'abstract_embeddings.npy')
    save_embeddings(embeddings, emb_path)
    
    print("5. Saving metadata...")
    meta_path = os.path.join(base_dir, 'data', 'embedding_metadata.csv')
    save_embedding_metadata(df, meta_path)
    
    print("6. Building similarity matrix...")
    sim_matrix = build_embedding_similarity_matrix(embeddings)
    
    print("7. Saving similarity matrix...")
    matrix_path = os.path.join(base_dir, 'data', 'matriz_embeddings_coseno.csv')
    save_similarity_matrix(sim_matrix, matrix_path)
    
    print("8. Generating validation report...")
    report_path = os.path.join(base_dir, 'data', 'embedding_validation_report.txt')
    
    total_articles = len(df)
    empty_abstracts = sum(df['Abstract'] == "")
    dim = embeddings.shape[1]
    
    report = f"--- EMBEDDING VALIDATION REPORT ---\n"
    report += f"Total articles: {total_articles}\n"
    report += f"Empty abstracts: {empty_abstracts}\n"
    report += f"Embedding dimension: {dim}\n"
    report += f"Embedding matrix shape: {embeddings.shape}\n"
    report += f"Similarity matrix shape: {sim_matrix.shape}\n"
    report += f"Is square matrix: {sim_matrix.shape[0] == sim_matrix.shape[1] and sim_matrix.shape[0] == total_articles}\n"
    
    diag = np.diagonal(sim_matrix)
    is_diag_one = np.allclose(diag, 1.0, atol=1e-5)
    report += f"Is diagonal close to 1.0: {is_diag_one}\n"
    
    report += "\nFiles generated:\n"
    report += f"- {emb_path}\n"
    report += f"- {meta_path}\n"
    report += f"- {matrix_path}\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Process complete. Report saved to {report_path}")

if __name__ == "__main__":
    generate_all()
