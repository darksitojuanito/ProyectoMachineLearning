import os
from src.filter_dataset import filter_articles
from src.data_loader import load_and_clean_data
from src.preprocessor import TextPreprocessor
from src.similarity import SimilarityEngine
from src.search_engine import SearchEngine
from src.recommender import Recommender
from src.validator import Validator

def main():
    print("=== UPScholar Phase 2 ===")
    
    data_dir = "data"
    raw_input_file = f"{data_dir}/DataSetCompleto.csv"
    articles_file = f"{data_dir}/DataSetArticulos.csv"
    
    if not os.path.exists(raw_input_file):
        print(f"Error: {raw_input_file} not found.")
        return

    # 0. Filtering
    print("\n0. Filtering non-papers...")
    filter_stats = filter_articles(raw_input_file, articles_file)
    print(f"Original: {filter_stats['total_original']}, Filtered: {filter_stats['total_filtered']}, Removed: {filter_stats['removed_count']}")

    # 1. Load Data
    print("\n1. Loading and cleaning data...")
    df = load_and_clean_data(articles_file)
    print(f"Loaded {len(df)} papers.")

    # 2. NLP Preprocessing
    print("\n2. Processing Text (NLP)...")
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df, data_dir)
    print("Text processing complete.")

    # 3. Similarity Matrices
    print("\n3. Building Similarity Matrices...")
    sim_engine = SimilarityEngine(w_title=0.10, w_keywords=0.20, w_abstract=0.70)
    final_matrix = sim_engine.build_and_save_matrices(df, data_dir)
    print("Matrices built and saved.")

    # 4. Search Query
    print("\n4. Interactive Search")
    query = input("Enter search query (e.g., 'deep learning medical image segmentation'): ")
    if not query.strip():
        print("Empty query. Exiting.")
        return

    # 5. Search Top 10
    print("\n5. Searching for Top 10 similar papers...")
    search_engine = SearchEngine(preprocessor, w_title=0.10, w_keywords=0.20, w_abstract=0.70)
    top10_df = search_engine.search(query, df, data_dir)
    
    print("\n=== TOP 10 RESULTS ===")
    for _, row in top10_df.iterrows():
        print(f"{row['rank']}. [Score: {row['score_final']:.4f}] {row['Title']}")

    # 6. Generate Recommendations
    print("\n6. Generating Recommendations (Top 3 for each Top 10 result)...")
    recommender = Recommender(final_matrix, df)
    rec_df = recommender.recommend(top10_df, data_dir)
    
    print("\n=== RECOMMENDATIONS ===")
    for paper_id in top10_df['paper_id']:
        paper_title = top10_df[top10_df['paper_id'] == paper_id]['Title'].values[0]
        recs = rec_df[rec_df['source_paper_id'] == paper_id]
        print(f"\nFor: {paper_title}")
        for _, rec in recs.iterrows():
            print(f"   -> [Sim: {rec['similarity_score']:.4f}] {rec['recommended_title']}")

    # 7. Validation
    print("\n7. Validating Results...")
    validator = Validator(df, final_matrix, top10_df, rec_df, filter_stats)
    validator.validate_and_report(f"{data_dir}/validation_report.txt")

if __name__ == "__main__":
    main()
