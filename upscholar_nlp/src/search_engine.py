import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.similarity import jaccard_similarity
from src.exporter import export_to_csv

class SearchEngine:
    def __init__(self, preprocessor, w_title=0.10, w_keywords=0.20, w_abstract=0.70):
        self.preprocessor = preprocessor
        self.w_title = w_title
        self.w_keywords = w_keywords
        self.w_abstract = w_abstract

    def search(self, query: str, df: pd.DataFrame, output_dir: str):
        """
        Executes a search query against the dataframe.
        """
        # Preprocess query
        processed_query = self.preprocessor.clean_text(query)
        if not processed_query:
            return pd.DataFrame()

        # Vectorize abstracts for cosine similarity
        vectorizer = TfidfVectorizer()
        abstracts_list = df['Abstract_nlp'].replace("", " ").tolist()
        try:
            tfidf_matrix = vectorizer.fit_transform(abstracts_list)
            query_vec = vectorizer.transform([processed_query])
            cos_sim_abstract = cosine_similarity(query_vec, tfidf_matrix)[0]
        except ValueError:
            cos_sim_abstract = [0.0] * len(df)

        results = []
        for i, row in df.iterrows():
            score_title = jaccard_similarity(processed_query, row['Title_nlp'])
            score_keywords = jaccard_similarity(processed_query, row['Keywords_nlp'])
            score_abstract = cos_sim_abstract[i]
            
            score_final = (self.w_title * score_title) + \
                          (self.w_keywords * score_keywords) + \
                          (self.w_abstract * score_abstract)
            
            results.append({
                'paper_id': row['paper_id'],
                'Title': row['Title'],
                'Keywords': row['Keywords'],
                'Abstract': row['Abstract'],
                'Session': row['Session'],
                'year': row['year'],
                'score_title': score_title,
                'score_keywords': score_keywords,
                'score_abstract': score_abstract,
                'score_final': score_final
            })
            
        results_df = pd.DataFrame(results)
        # Sort and get top 10
        top10_df = results_df.sort_values(by='score_final', ascending=False).head(10).copy()
        
        # Add rank
        top10_df.insert(0, 'rank', range(1, len(top10_df) + 1))
        
        # Export
        export_to_csv(top10_df, f"{output_dir}/top10_resultados.csv")
        
        return top10_df
