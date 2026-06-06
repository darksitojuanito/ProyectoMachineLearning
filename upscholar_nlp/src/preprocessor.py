import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.exporter import export_to_csv

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text: str) -> str:
        """
        Cleans text: lowercase, remove special characters and numbers, 
        remove stopwords, apply stemming.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenize (by splitting on whitespace)
        words = text.split()
        
        # Remove stopwords and stem
        processed_words = [
            self.stemmer.stem(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return " ".join(processed_words)

    def process_dataframe(self, df: pd.DataFrame, output_dir: str):
        """
        Processes Title, Keywords, and Abstract columns and saves the results.
        Returns the processed DataFrame.
        """
        df_processed = df.copy()
        
        columns_to_process = {
            'Title': 'processed_titles.csv',
            'Keywords': 'processed_keywords.csv',
            'Abstract': 'processed_abstracts.csv'
        }
        
        for col, filename in columns_to_process.items():
            # Apply cleaning
            df_processed[f'{col}_nlp'] = df_processed[col].apply(self.clean_text)
            
            # Save intermediate CSV
            temp_df = pd.DataFrame({
                'paper_id': df_processed['paper_id'],
                'texto_original': df_processed[col],
                'texto_nlp': df_processed[f'{col}_nlp']
            })
            export_to_csv(temp_df, f"{output_dir}/{filename}")
            
        return df_processed
