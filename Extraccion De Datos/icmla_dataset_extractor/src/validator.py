import pandas as pd
import os
from .utils import get_logger

logger = get_logger("Validator")

class Validator:
    def __init__(self, extended_file="data/DatasetExtendido.csv"):
        self.extended_file = extended_file
        self.report_file = "data/validation_report.txt"

    def validate(self):
        logger.info("Generando reporte de validación...")
        if not os.path.exists(self.extended_file):
            logger.error("No se puede validar porque no existe el dataset extendido.")
            return
            
        df = pd.read_csv(self.extended_file)
        
        total_papers = len(df)
        with_ieee_url = df['ieee_url'].notna().sum()
        with_abstract = df['abstract'].notna().sum()
        # pandas considera vacíos los strings vacíos usando .replace('', pd.NA) o verificando len
        empty_abstracts = (df['abstract'].fillna('') == '').sum()
        with_abstract = total_papers - empty_abstracts
        
        empty_keywords = (df['keywords'].fillna('') == '').sum()
        with_keywords = total_papers - empty_keywords
        
        # Duplicados
        df['final_title_lower'] = df['dblp_title'].str.lower().fillna('')
        duplicates = df.duplicated(subset=['final_title_lower'], keep=False).sum()
        
        report_lines = [
            "=========================================",
            "REPORTE DE VALIDACIÓN - ICMLA 2020",
            "=========================================",
            f"Total de papers extraídos desde DBLP: {total_papers}",
            f"Papers con URL de IEEE: {with_ieee_url}",
            f"Papers con Abstract: {with_abstract}",
            f"Papers sin Abstract: {empty_abstracts}",
            f"Papers con Author Keywords: {with_keywords}",
            f"Papers sin Keywords: {empty_keywords}",
            f"Papers con títulos duplicados (posibles errores en DBLP): {duplicates}",
            "========================================="
        ]
        
        report_text = "\n".join(report_lines)
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        logger.info(f"Reporte generado en {self.report_file}")
        print("\n" + report_text + "\n")

if __name__ == "__main__":
    validator = Validator()
    validator.validate()
