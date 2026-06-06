import pandas as pd
import os

def filter_articles(input_file: str, output_file: str):
    """
    Reads the original dataset, filters out non-papers (keynotes, etc.),
    reassigns paper_id sequentially, and exports the clean dataset.
    Returns a dictionary with filtering statistics.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Detect separator (from data_loader logic)
    with open(input_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line and first_line.count(';') >= 5 else ','

    df = pd.read_csv(input_file, sep=sep, encoding='utf-8')
    df.columns = df.columns.str.strip()

    total_original = len(df)
    
    # Define exclusion rules
    # Exclude if Session contains keynote, invited, opening, etc.
    # Exclude if Title contains specific known non-paper titles
    
    def is_paper(row):
        session = str(row.get('Session', '')).lower()
        title = str(row.get('Title', '')).lower()
        
        exclusion_keywords = ['keynote', 'invited talk', 'opening remarks', 'best paper', 'tutorial']
        for keyword in exclusion_keywords:
            if keyword in session or keyword in title:
                return False
                
        specific_titles = [
            'learning with unpaired data',
            'a cognitive architecture for object recognition in video'
        ]
        for specific in specific_titles:
            if specific in title:
                return False
                
        return True

    # Apply filter
    is_paper_mask = df.apply(is_paper, axis=1)
    df_filtered = df[is_paper_mask].copy()
    
    removed_count = total_original - len(df_filtered)
    
    # Reassign paper_id sequentially 1 to N
    df_filtered['paper_id'] = range(1, len(df_filtered) + 1)
    
    # Save the new dataset
    df_filtered.to_csv(output_file, index=False, encoding='utf-8', sep=sep)
    
    stats = {
        'total_original': total_original,
        'total_filtered': len(df_filtered),
        'removed_count': removed_count
    }
    
    return stats
