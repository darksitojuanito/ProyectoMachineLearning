import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.exporter import export_to_csv

def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculates Jaccard similarity between two space-separated string texts."""
    if not text1 and not text2:
        return 0.0
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def build_jaccard_matrix(texts: pd.Series) -> np.ndarray:
    """Builds a document-document Jaccard similarity matrix."""
    n = len(texts)
    matrix = np.zeros((n, n))
    text_list = texts.tolist()
    for i in range(n):
        for j in range(i, n):
            sim = jaccard_similarity(text_list[i], text_list[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix

def build_tfidf_cosine_matrix(texts: pd.Series) -> np.ndarray:
    """Builds a document-document Cosine similarity matrix using TF-IDF."""
    # Replace empty with " " so vectorizer doesn't fail on all empty documents
    text_list = texts.replace("", " ").tolist()
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(text_list)
        cos_sim = cosine_similarity(tfidf_matrix)
        return cos_sim
    except ValueError:
        # In case all documents are completely empty and TFIDF fails
        return np.zeros((len(texts), len(texts)))

class SimilarityEngine:
    def __init__(self, w_title=0.10, w_keywords=0.20, w_abstract=0.70):
        self.w_title = w_title
        self.w_keywords = w_keywords
        self.w_abstract = w_abstract

    def build_and_save_matrices(self, df: pd.DataFrame, output_dir: str):
        """
        Builds Jaccard and Cosine matrices, combines them, and saves to CSV.
        Returns the final weighted matrix.
        """
        print("Building Jaccard matrix for Titles...")
        jaccard_titles = build_jaccard_matrix(df['Title_nlp'])
        
        print("Building Jaccard matrix for Keywords...")
        jaccard_keywords = build_jaccard_matrix(df['Keywords_nlp'])
        
        print("Building TF-IDF + Cosine matrix for Abstracts...")
        cos_abstracts = build_tfidf_cosine_matrix(df['Abstract_nlp'])
        
        # Combine
        final_matrix = (
            self.w_title * jaccard_titles +
            self.w_keywords * jaccard_keywords +
            self.w_abstract * cos_abstracts
        )
        
        # Force diagonal to 1.0
        np.fill_diagonal(final_matrix, 1.0)
        
        # Save matrices
        # To avoid saving huge matrices if not strictly needed, we convert them to dataframes
        # We use paper_id as index and columns
        paper_ids = df['paper_id'].astype(str).tolist()
        
        df_jaccard_titles = pd.DataFrame(jaccard_titles, index=paper_ids, columns=paper_ids)
        export_to_csv(df_jaccard_titles.reset_index().rename(columns={'index': 'paper_id'}), f"{output_dir}/matriz_jaccard_titles.csv")
        
        df_jaccard_keywords = pd.DataFrame(jaccard_keywords, index=paper_ids, columns=paper_ids)
        export_to_csv(df_jaccard_keywords.reset_index().rename(columns={'index': 'paper_id'}), f"{output_dir}/matriz_jaccard_keywords.csv")
        
        df_cos_abstracts = pd.DataFrame(cos_abstracts, index=paper_ids, columns=paper_ids)
        export_to_csv(df_cos_abstracts.reset_index().rename(columns={'index': 'paper_id'}), f"{output_dir}/matriz_coseno_abstracts.csv")
        
        df_final = pd.DataFrame(final_matrix, index=paper_ids, columns=paper_ids)
        export_to_csv(df_final.reset_index().rename(columns={'index': 'paper_id'}), f"{output_dir}/matriz_final_ponderada.csv")
        
        return final_matrix
