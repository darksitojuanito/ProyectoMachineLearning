# Extractor de Dataset ICMLA 2020

Este proyecto contiene un pipeline de extracción automatizada para construir un dataset de los papers presentados en la conferencia ICMLA 2020. Combina un scraping inicial de los metadatos desde DBLP con una extracción profunda desde IEEE Xplore.

## Características

* Extrae la lista completa de artículos desde DBLP.
* Extrae metadatos (título, abstract, autores, DOI, URL, y **Author Keywords**) desde IEEE Xplore.
* Usa estrategias combinadas (`requests` + `BeautifulSoup` para datos internos en JSON, con `Playwright` como mecanismo de fallback).
* Implementa un sistema de caché (HTML y JSON) para evitar el sobreuso de la red.
* Genera múltiples CSVs y un reporte de validación.

## Requisitos e Instalación

Requiere Python 3.10 o superior.

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Instalar los navegadores de Playwright:
   ```bash
   playwright install
   ```

## Archivos Generados (`data/`)

El programa genera los siguientes archivos:

* `dblp_icmla2020_raw.csv`: Datos crudos directamente de DBLP.
* `ieee_icmla2020_extended.csv`: Todos los metadatos recopilados por IEEE.
* `DatasetExtendido.csv`: Dataset fusionado con todas las columnas disponibles y estados de extracción.
* `DatasetBasico.csv`: Dataset final requerido con las columnas: `paper_id, Title, Keywords, Abstract, year, Session`.
* `validation_report.txt`: Resumen del proceso de scraping.
* `sessions_manual.csv`: Archivo en el que se podrán cargar manualmente las sesiones por ID de paper.

## Notas Importantes

* **Sesiones**: La columna `session` se completará en una etapa posterior. Este proyecto provee una plantilla vacía (`sessions_manual.csv`) pero no auto-genera sesiones, ya que requieren del programa oficial de ICMLA 2020.
* **Keywords**: El dataset restringe las keywords única y exclusivamente a **Author Keywords**. Otras categorías como *IEEE Terms* o *INSPEC Terms* son omitidas.

## Cómo Ejecutar

Para iniciar el pipeline completo, simplemente ejecuta:
```bash
python main.py
```
