import pandas as pd
import os

def detect_separator(file_path):
    """Detects whether the file uses comma or semicolon as a separator."""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if ';' in first_line and first_line.count(';') >= 5: # Assuming at least 6 columns
            return ';'
        return ','

def load_and_clean_data(file_path):
    """
    Loads the dataset, cleans column names, handles missing values,
    and returns a clean pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    sep = detect_separator(file_path)
    
    # Read CSV
    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
    
    # Clean column names: remove leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # Drop columns that start with 'Unnamed' or are empty strings
    cols_to_drop = [col for col in df.columns if col.startswith('Unnamed') or col == '']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Drop previous NLP columns if they exist in the loaded file
    nlp_cols = ['Title_nlp', 'Keywords_nlp', 'Abstract_nlp']
    df = df.drop(columns=[col for col in nlp_cols if col in df.columns])
    
    # Required columns
    required_cols = ['paper_id', 'Title', 'Keywords', 'Abstract', 'year', 'Session']
    
    # Validate missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    # Ensure empty string instead of NaN for text fields
    text_cols = ['Title', 'Keywords', 'Abstract']
    for col in text_cols:
        df[col] = df[col].fillna('')
        df[col] = df[col].astype(str).str.strip()
        
    return df
