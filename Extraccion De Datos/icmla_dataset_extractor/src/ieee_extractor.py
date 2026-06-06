import os
import re
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from .utils import get_logger, random_delay, load_from_cache, save_to_cache

logger = get_logger("IEEEExtractor")

class IEEEExtractor:
    def __init__(self, input_file="data/dblp_icmla2020_raw.csv"):
        self.input_file = input_file
        self.output_dir = "data"
        self.output_file = os.path.join(self.output_dir, "ieee_icmla2020_extended.csv")
        
        # Headers para requests simulando un navegador real
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _extract_from_json_data(self, metadata):
        """Extrae la información relevante del objeto JSON de IEEE."""
        extracted = {
            'ieee_title': metadata.get('title', ''),
            'abstract': metadata.get('abstract', ''),
            'doi': metadata.get('doi', ''),
            'authors': ', '.join([a.get('name', '') for a in metadata.get('authors', [])]),
            'keywords': '',
            'extraction_status': 'success_json'
        }
        
        # Filtrar solo Author Keywords
        keywords_list = metadata.get('keywords', [])
        author_keywords = []
        for kw_category in keywords_list:
            if kw_category.get('type') == 'Author Keywords':
                author_keywords = kw_category.get('kwd', [])
                break
                
        if author_keywords:
            extracted['keywords'] = ', '.join(author_keywords)
        else:
            extracted['extraction_status'] = 'missing_author_keywords'
            
        return extracted

    def _extract_with_requests(self, url):
        """Intenta extraer interceptando el JSON embebido en el HTML con requests."""
        html = load_from_cache(url, ext="html")
        from_cache = True
        
        if not html:
            from_cache = False
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                html = response.text
                save_to_cache(url, html, ext="html")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Petición requests falló: {e}")
                if e.response is not None and e.response.status_code in [429, 420, 403]:
                    logger.warning(f"IEEE detectó demasiadas peticiones ({e.response.status_code}). Enfriando por 60 segundos...")
                    import time
                    time.sleep(60)
                return None, False
            
        # Buscar el objeto metadata JSON en el HTML
        match = re.search(r'(?:xplore|xplGlobal)\.document\.metadata\s*=\s*({.*?});', html, re.DOTALL)
        if match:
            metadata_str = match.group(1)
            try:
                metadata = json.loads(metadata_str)
                save_to_cache(url + "_meta", metadata, ext="json")
                return self._extract_from_json_data(metadata), from_cache
            except json.JSONDecodeError:
                logger.warning(f"Error decodificando JSON en {url}")
        
        return None, from_cache

    def _extract_with_playwright(self, url):
        """Fallback usando Playwright si requests no encuentra el JSON."""
        logger.info(f"Usando fallback Playwright para: {url}. Si ves una ventana, por favor resuelve el Captcha.")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(user_agent=self.headers["User-Agent"])
            page = context.new_page()
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Esperar a que cargue el título al menos (60 segundos por si hay Captcha)
                page.wait_for_selector(".document-title", timeout=60000)
                
                # Intentar buscar el JSON directamente en el código fuente de la página cargada
                html_content = page.content()
                import re
                match = re.search(r'(?:xplore|xplGlobal)\.document\.metadata\s*=\s*({.*?});', html_content, re.DOTALL)
                
                if match:
                    metadata_str = match.group(1)
                    import json
                    try:
                        metadata = json.loads(metadata_str)
                        save_to_cache(url + "_meta", metadata, ext="json")
                        extracted = self._extract_from_json_data(metadata)
                        extracted['extraction_status'] = 'success_playwright_json'
                        if not extracted['keywords']:
                            extracted['extraction_status'] = 'missing_author_keywords'
                        return extracted
                    except json.JSONDecodeError:
                        logger.warning("No se pudo decodificar el JSON desde Playwright")

                # Si falla el JSON, extraer del DOM (incluyendo keywords)
                title = page.locator(".document-title").inner_text() if page.locator(".document-title").count() > 0 else ""
                abstract = page.locator(".abstract-text").inner_text() if page.locator(".abstract-text").count() > 0 else ""
                
                # Intentar buscar las Author Keywords en el DOM
                keywords_str = ""
                if page.locator("a[data-teal-bpos='Author Keywords']").count() > 0:
                    kws = page.locator("a[data-teal-bpos='Author Keywords']").all_inner_texts()
                    keywords_str = ", ".join([k.strip() for k in kws if k.strip()])
                
                return {
                    'ieee_title': title,
                    'abstract': abstract,
                    'doi': '',
                    'authors': '',
                    'keywords': keywords_str,
                    'extraction_status': 'success_playwright_dom_partial' if title else 'error'
                }
            except Exception as e:
                logger.error(f"Error con Playwright en {url}: {str(e)}")
                return {
                    'extraction_status': 'error',
                    'error_message': str(e)
                }
            finally:
                browser.close()

    def extract_all(self):
        if not os.path.exists(self.input_file):
            logger.error(f"El archivo {self.input_file} no existe. Ejecuta DBLPExtractor primero.")
            return

        df_dblp = pd.read_csv(self.input_file)
        results = []

        for index, row in df_dblp.iterrows():
            url = row.get('ieee_url', '')
            dblp_title = row.get('dblp_title', '')
            
            base_data = {
                'dblp_title': dblp_title,
                'dblp_authors': row.get('dblp_authors', ''),
                'dblp_pages': row.get('dblp_pages', ''),
                'ieee_url': url,
                'ieee_title': '',
                'authors': '',
                'abstract': '',
                'doi': '',
                'keywords': '',
                'extraction_status': 'no_url',
                'error_message': ''
            }

            if not url:
                logger.warning(f"No hay URL para el paper: {dblp_title}")
                results.append(base_data)
                continue

            # Revisar si ya extraimos el meta json en caché directamente
            cached_meta = load_from_cache(url + "_meta", ext="json")
            if cached_meta:
                extracted = self._extract_from_json_data(cached_meta)
                base_data.update(extracted)
                results.append(base_data)
                continue

            logger.info(f"Procesando {index+1}/{len(df_dblp)}: {url}")
            
            try:
                # Intento principal
                extracted, from_cache = self._extract_with_requests(url)
                
                if extracted:
                    base_data.update(extracted)
                else:
                    # Fallback Playwright
                    extracted_pw = self._extract_with_playwright(url)
                    base_data.update(extracted_pw)

                if not from_cache:
                    random_delay(5, 15) # Delay de 5 a 15 segundos solicitado
                    
            except Exception as e:
                logger.error(f"Error procesando {url}: {str(e)}")
                base_data['extraction_status'] = 'error'
                base_data['error_message'] = str(e)
                random_delay(5, 15)

            results.append(base_data)

            # Guardar avances parciales
            if (index + 1) % 10 == 0:
                pd.DataFrame(results).to_csv(self.output_file, index=False)
                logger.info(f"Guardado parcial de {index+1} registros.")

        # Guardado final
        df_final = pd.DataFrame(results)
        df_final.to_csv(self.output_file, index=False)
        logger.info(f"Extracción finalizada. Datos guardados en {self.output_file}")
        return df_final

if __name__ == "__main__":
    extractor = IEEEExtractor()
    extractor.extract_all()
