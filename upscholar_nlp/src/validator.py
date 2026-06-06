import pandas as pd
import numpy as np

class Validator:
    def __init__(self, df: pd.DataFrame, final_matrix: np.ndarray, top10_df: pd.DataFrame, rec_df: pd.DataFrame, filter_stats: dict = None):
        self.df = df
        self.final_matrix = final_matrix
        self.top10_df = top10_df
        self.rec_df = rec_df
        self.filter_stats = filter_stats

    def validate_and_report(self, output_path: str):
        report = []
        report.append("=== Validation Report ===")
        
        if self.filter_stats:
            report.append(f"Total original records: {self.filter_stats['total_original']}")
            report.append(f"Removed non-papers (keynotes, etc.): {self.filter_stats['removed_count']}")
            report.append(f"Total valid papers (after filter): {self.filter_stats['total_filtered']}")
            report.append("-" * 20)
            
        # Dataset validation
        total_papers = len(self.df)
        report.append(f"Total papers: {total_papers}")
        
        base_columns = [col for col in self.df.columns if not col.endswith('_nlp') and not col.startswith('Unnamed')]
        report.append(f"Columns found: {', '.join(base_columns)}")
        
        empty_titles = (self.df['Title'] == '').sum()
        empty_keywords = (self.df['Keywords'] == '').sum()
        empty_abstracts = (self.df['Abstract'] == '').sum()
        empty_sessions = self.df['Session'].isna().sum() + (self.df['Session'] == '').sum()
        
        report.append(f"Empty Titles: {empty_titles}")
        report.append(f"Empty Keywords: {empty_keywords}")
        report.append(f"Empty Abstracts: {empty_abstracts}")
        report.append(f"Empty Sessions: {empty_sessions}")
        
        # Matrix validation
        shape = self.final_matrix.shape
        report.append(f"Matrix shape: {shape}")
        
        is_square = shape[0] == shape[1] and shape[0] == total_papers
        report.append(f"Is matrix square and matches dataset size? {'Yes' if is_square else 'No'}")
        
        # Check diagonal
        diagonal = np.diag(self.final_matrix)
        is_diag_one = np.allclose(diagonal, 1.0, atol=1e-5)
        report.append(f"Is diagonal 1 (or close to 1)? {'Yes' if is_diag_one else 'No'}")
        
        # Top 10 validation
        expected_top10 = min(10, total_papers)
        actual_top10 = len(self.top10_df)
        report.append(f"Top 10 count: {actual_top10} (Expected: {expected_top10})")
        
        # Recommendations validation
        top10_ids = set(self.top10_df['paper_id'].tolist())
        valid_recs_count = True
        no_overlap = True
        
        for paper_id in top10_ids:
            paper_recs = self.rec_df[self.rec_df['source_paper_id'] == paper_id]
            if len(paper_recs) != 3 and total_papers > 13: # at least 1 + 10 + 3 papers to be able to recommend 3 outside top 10
                valid_recs_count = False
            
            rec_ids = set(paper_recs['recommended_paper_id'].tolist())
            if rec_ids.intersection(top10_ids):
                no_overlap = False
                
        report.append(f"Does each Top 10 paper have 3 recommendations? {'Yes' if valid_recs_count else 'No'}")
        report.append(f"Are recommendations strictly outside the Top 10? {'Yes' if no_overlap else 'No'}")
        
        # Write report
        report_text = "\n".join(report)
        print(report_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print(f"Validation report saved to: {output_path}")
