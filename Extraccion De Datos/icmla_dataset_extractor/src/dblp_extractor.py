import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from .utils import get_logger, random_delay, load_from_cache, save_to_cache

logger = get_logger("DBLPExtractor")

class DBLPExtractor:
    def __init__(self, url="https://dblp.org/db/conf/icmla/icmla2020.html"):
        self.url = url
        self.output_dir = "data"
        self.output_file = os.path.join(self.output_dir, "dblp_icmla2020_raw.csv")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self):
        logger.info(f"Iniciando extracción desde DBLP: {self.url}")
        
        html = load_from_cache(self.url, ext="html")
        if not html:
            logger.info("No se encontró caché, realizando petición web...")
            response = requests.get(self.url)
            response.raise_for_status()
            html = response.text
            save_to_cache(self.url, html, ext="html")
            random_delay(2, 5) # Pequeño delay de cortesía post-descarga
        else:
            logger.info("Cargado desde caché HTML.")

        soup = BeautifulSoup(html, 'html.parser')
        papers = soup.find_all('li', class_='entry inproceedings')
        logger.info(f"Se encontraron {len(papers)} papers en DBLP.")
        
        extracted_data = []

        for paper in papers:
            # Titulo
            title_tag = paper.find('span', class_='title')
            title = title_tag.text.strip() if title_tag else ""

            # Autores
            authors_tags = paper.find_all('span', itemprop='author')
            authors = [a.text.strip() for a in authors_tags]

            # Páginas
            pages_tag = paper.find('span', itemprop='pagination')
            pages = pages_tag.text.strip() if pages_tag else ""

            # Enlaces (IEEE / DOI)
            nav_ul = paper.find('nav', class_='publ')
            ieee_url = ""
            if nav_ul:
                links = nav_ul.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    # Priorizar DOI o enlace directo de IEEE
                    if 'doi.org/10.1109' in href or 'ieeexplore.ieee.org/document' in href:
                        ieee_url = href
                        break
            
            extracted_data.append({
                'dblp_title': title,
                'dblp_authors': ", ".join(authors),
                'dblp_pages': pages,
                'ieee_url': ieee_url
            })

        df = pd.DataFrame(extracted_data)
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        logger.info(f"Extracción finalizada. Datos guardados en {self.output_file}")
        return df

if __name__ == "__main__":
    extractor = DBLPExtractor()
    extractor.extract()
