import pandas as pd
import os
from .utils import get_logger

logger = get_logger("DatasetBuilder")

class DatasetBuilder:
    def __init__(self, dblp_file="data/dblp_icmla2020_raw.csv", ieee_file="data/ieee_icmla2020_extended.csv"):
        self.dblp_file = dblp_file
        self.ieee_file = ieee_file
        self.output_extended = "data/DatasetExtendido.csv"
        self.output_final = "data/DatasetBasico.csv"

    def build(self):
        logger.info("Iniciando construcción de datasets finales...")
        
        if not os.path.exists(self.ieee_file):
            logger.error(f"No se encontró el archivo de IEEE: {self.ieee_file}")
            return

        df_ieee = pd.read_csv(self.ieee_file)
        
        # 1. Asignar paper_id
        df_ieee.insert(0, 'paper_id', range(1, 1 + len(df_ieee)))
        
        # 2. Agregar session (vacía por defecto)
        df_ieee['session'] = ''
        
        # 3. Fijar year en 2020
        df_ieee['year'] = 2020
        
        # 4. Consolidar autores (Usar los de IEEE si existen, sino DBLP)
        df_ieee['final_authors'] = df_ieee.apply(
            lambda row: row['authors'] if pd.notna(row.get('authors')) and row.get('authors') != '' else row.get('dblp_authors'),
            axis=1
        )
        
        # 5. Consolidar título (Usar IEEE si existe, sino DBLP)
        df_ieee['final_title'] = df_ieee.apply(
            lambda row: row['ieee_title'] if pd.notna(row.get('ieee_title')) and row.get('ieee_title') != '' else row.get('dblp_title'),
            axis=1
        )
        
        # 6. Llenar nulos en keywords y abstract con string vacío
        df_ieee['keywords'] = df_ieee['keywords'].fillna('')
        df_ieee['abstract'] = df_ieee['abstract'].fillna('')

        # Guardar Extended Dataset
        logger.info(f"Guardando Extended Dataset en {self.output_extended}")
        # Renombramos para ajustarnos a lo pedido en extended: dblp_title, ieee_title, authors, keywords, abstract, session, year, doi, pages, dblp_url(n/a), ieee_url, extraction_status, error_message
        df_extended = df_ieee.copy()
        df_extended.to_csv(self.output_extended, index=False)

        # Construir Final Dataset: paper_id, Title, Keywords, Abstract, year, Session
        df_final = df_ieee[['paper_id', 'final_title', 'keywords', 'abstract', 'year', 'session']].copy()
        df_final.rename(columns={'final_title': 'Title', 'keywords': 'Keywords', 'abstract': 'Abstract', 'session': 'Session'}, inplace=True)
        df_final = df_final[['paper_id', 'Title', 'Keywords', 'Abstract', 'year', 'Session']]
        
        logger.info(f"Guardando Dataset Final Obligatorio en {self.output_final}")
        df_final.to_csv(self.output_final, index=False)
        
        logger.info("Datasets construidos exitosamente.")

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
