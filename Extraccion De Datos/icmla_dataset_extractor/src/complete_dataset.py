import pandas as pd
import docx
import re
import os

def normalize_title(title):
    if not isinstance(title, str):
        return ""
    # Convert to lowercase
    title = title.lower()
    # If the docx title has numbers at the start, remove them (e.g. "70.Writer...")
    title = re.sub(r'^\d+\.\s*', '', title)
    # Remove standard paper types from docx title
    title = re.sub(r'\(regular paper\)', '', title)
    title = re.sub(r'\(short paper\)', '', title)
    title = re.sub(r'\(poster\)', '', title)
    # Remove all non-alphanumeric characters for a robust match
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def parse_sessions_from_docx(docx_path):
    doc = docx.Document(docx_path)
    sessions_map = {}
    
    for table in doc.tables:
        for row_idx, row in enumerate(table.rows):
            has_session = False
            for cell in row.cells:
                if 'Session Chair' in cell.text:
                    has_session = True
                    break
            
            if has_session:
                header_row = row
                sessions_in_columns = []
                for cell in header_row.cells:
                    text = cell.text.strip().replace('\n', ' ')
                    if 'Session Chair' in text:
                        session_name = text.split('Session Chair')[0].strip()
                    else:
                        session_name = None
                    sessions_in_columns.append(session_name)
                    
                # Look ahead for papers in these columns
                for paper_row in table.rows[row_idx+1:]:
                    # If this row is another header, we should technically stop, 
                    # but usually headers are separated by tables or clear breaks.
                    # We'll just read cells
                    for col_idx, cell in enumerate(paper_row.cells):
                        if col_idx < len(sessions_in_columns) and sessions_in_columns[col_idx]:
                            text = cell.text.strip().replace('\n', ' ')
                            if not text: continue
                            
                            # Usually titles are "ID. Title (type) Authors"
                            # Let's extract everything up to the first parenthesis or hyphen
                            match = re.search(r'^\d+\.\s*(.*?)\s*\(', text)
                            if match:
                                title = match.group(1).strip()
                            else:
                                title = text
                            
                            norm_title = normalize_title(title)
                            if norm_title:
                                sessions_map[norm_title] = sessions_in_columns[col_idx]
                                
    return sessions_map

def main():
    print("Iniciando la corrección y cruce del dataset...")
    
    input_csv = "data/DatasetBasico(revision).csv"
    output_csv = "data/DataSetCompleto.csv"
    docx_path = "data/ICMLA 2020 Program.docx"
    
    try:
        # El archivo de revisión fue modificado en Excel y está separado por punto y coma (;)
        df = pd.read_csv(input_csv, sep=';', encoding='latin-1')
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
        
    print(f"Cargados {len(df)} papers desde el CSV.")
    
    # 2. Arreglar la columna year
    missing_years_before = df['year'].isnull().sum()
    df['year'] = df['year'].fillna(2020)
    print(f"Rellenados {missing_years_before} valores vacíos en la columna 'year'.")
    
    # 3. Extraer sesiones del Docx
    sessions_map = parse_sessions_from_docx(docx_path)
    print(f"Extraídas {len(sessions_map)} posibles sesiones desde el documento Word.")
    
    # 4. Cruzar las sesiones con los papers
    matched_count = 0
    
    for idx, row in df.iterrows():
        # Tomar el título del CSV
        csv_title = str(row['Title'])
        norm_csv_title = normalize_title(csv_title)
        
        # Buscar en el mapa
        found_session = None
        # Búsqueda exacta
        if norm_csv_title in sessions_map:
            found_session = sessions_map[norm_csv_title]
        else:
            # Búsqueda difusa/parcial (por si hay pequeñas discrepancias)
            for docx_norm_title, session in sessions_map.items():
                if docx_norm_title in norm_csv_title or norm_csv_title in docx_norm_title:
                    if len(docx_norm_title) > 10 and len(norm_csv_title) > 10:
                        found_session = session
                        break
        
        if found_session:
            df.at[idx, 'Session'] = found_session
            matched_count += 1
            
    print(f"Sesiones machetadas y asignadas con éxito: {matched_count} de {len(df)}")
    
    # Asegurar el formato final y guardar
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Proceso completado. Dataset guardado en {output_csv}")

if __name__ == '__main__':
    main()
