import pandas as pd
import numpy as np
from src.exporter import export_to_csv

class Recommender:
    def __init__(self, final_matrix: np.ndarray, df: pd.DataFrame):
        self.final_matrix = final_matrix
        self.df = df
        
        # Create a mapping from paper_id to matrix index and vice versa
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(df['paper_id'])}
        self.idx_to_paper_id = {idx: pid for idx, pid in enumerate(df['paper_id'])}

    def recommend(self, top10_df: pd.DataFrame, output_dir: str):
        """
        Recommends 3 similar articles for each article in top10_df.
        Excludes the article itself and all articles in the top 10.
        """
        if top10_df.empty:
            return pd.DataFrame()
            
        top10_paper_ids = set(top10_df['paper_id'].tolist())
        recommendations = []
        
        for _, row in top10_df.iterrows():
            source_paper_id = row['paper_id']
            source_title = row['Title']
            
            if source_paper_id not in self.paper_id_to_idx:
                continue
                
            idx = self.paper_id_to_idx[source_paper_id]
            scores = self.final_matrix[idx]
            
            # Create a list of tuples (target_idx, target_score)
            scored_items = [(i, score) for i, score in enumerate(scores)]
            
            # Sort descending by score
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            found_recs = 0
            for target_idx, score in scored_items:
                target_paper_id = self.idx_to_paper_id[target_idx]
                
                # Exclude the paper itself and any paper in the Top 10
                if target_paper_id == source_paper_id or target_paper_id in top10_paper_ids:
                    continue
                    
                target_row = self.df.iloc[target_idx]
                
                recommendations.append({
                    'source_paper_id': source_paper_id,
                    'source_title': source_title,
                    'recommended_paper_id': target_paper_id,
                    'recommended_title': target_row['Title'],
                    'recommended_session': target_row['Session'],
                    'similarity_score': score
                })
                
                found_recs += 1
                if found_recs >= 3:
                    break
                    
        rec_df = pd.DataFrame(recommendations)
        
        if not rec_df.empty:
            export_to_csv(rec_df, f"{output_dir}/recomendaciones_top3.csv")
            
        return rec_df
