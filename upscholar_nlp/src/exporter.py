import pandas as pd
import os

def export_to_csv(df: pd.DataFrame, file_path: str):
    """
    Exports a pandas DataFrame to a CSV file.
    Rounds numerical columns to 4 decimal places where applicable.
    Uses UTF-8 encoding.
    """
    # Round float columns to 4 decimal places
    df_rounded = df.copy()
    float_cols = df_rounded.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        df_rounded[col] = df_rounded[col].round(4)
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    df_rounded.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Exported successfully to: {file_path}")
