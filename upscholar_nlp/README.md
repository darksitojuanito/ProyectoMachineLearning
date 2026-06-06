# UPScholar - Buscador Inteligente de Documentos Científicos

## Objetivo del Sistema
La Fase 2 de UPScholar es un motor tradicional de búsqueda y recomendación basado en Procesamiento de Lenguaje Natural (NLP). Permite ingresar una consulta de búsqueda y devolver los 10 artículos científicos más relevantes del dataset de ICMLA 2020. Además, para cada artículo del top 10, recomienda 3 artículos relacionados excluyendo los propios del top 10.

## Estructura del Proyecto
```
upscholar_nlp/
├── main.py                     # CLI principal del pipeline
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación
├── data/                       # Dataset de entrada y matrices/resultados procesados
├── src/                        # Código fuente modular
│   ├── data_loader.py          # Carga y limpieza
│   ├── preprocessor.py         # Procesamiento NLP (stemming, stopwords)
│   ├── similarity.py           # Cálculo de Jaccard y TF-IDF+Coseno
│   ├── search_engine.py        # Motor de búsqueda (Top 10)
│   ├── recommender.py          # Motor de recomendaciones (Top 3)
│   ├── exporter.py             # Utilidades de exportación CSV
│   └── validator.py            # Validación de integridad del modelo
└── app/
    └── streamlit_app.py        # Interfaz Web
```

## Metodología y Pesos
El modelo no utiliza embeddings ni LLM en esta fase. Se basa en una bolsa de palabras y comparaciones directas:
- **NLP**: Limpieza de texto (minúsculas, sin caracteres especiales), eliminación de stopwords en inglés, y stemming (PorterStemmer). Las keywords vacías se mantienen sin aportar similitud.
- **Title (Peso 0.10)**: Similitud de Jaccard.
- **Keywords (Peso 0.20)**: Similitud de Jaccard.
- **Abstract (Peso 0.70)**: Vectorización TF-IDF y Similitud Coseno.

Las recomendaciones se extraen de la matriz final ponderada (documento-documento) que suma las tres métricas bajo sus respectivos pesos.

## Instalación de Dependencias

Se recomienda utilizar un entorno virtual:
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

## Ejecución por Consola

Para correr todo el pipeline analítico por consola (carga, procesamiento, matrices, búsqueda y validación):
```bash
python main.py
```
El sistema te pedirá ingresar una consulta (ej: `deep learning medical image segmentation`).

## Ejecución Interfaz Web (HTML/JS + Flask API)

Para levantar el servidor backend de Python y la aplicación web:
```bash
python backend/app.py
```

Al ejecutarse, el servidor se iniciará en el puerto 5000.
Simplemente abre tu navegador y ve a:
`http://localhost:5000`

Desde ahí podrás usar el buscador de forma interactiva con un diseño moderno.

## Interpretación de resultados

* El Top 10 se calcula comparando la consulta del usuario contra todos los artículos.
* Las recomendaciones se calculan comparando cada artículo del Top 10 contra todos los demás artículos.
* Las recomendaciones excluyen los artículos que ya están en el Top 10 actual.
* Por eso, si se busca luego el título exacto de un artículo, los resultados pueden cambiar, porque el Top 10 actual cambia y también cambia el conjunto excluido.
* El Score Final se calcula así:
  `Score Final = 0.10 * score_title + 0.20 * score_keywords + 0.70 * score_abstract`

## Procesamiento Inteligente de Consultas

El sistema cuenta con una capa de pre-procesamiento antes de ejecutar el motor de similitud:
* **Traducción Automática**: Si escribes en español términos técnicos (ej. "aprendizaje profundo", "segmentación médica"), se traducen automáticamente al inglés de forma local.
* **Tolerancia a Errores Ortográficos (Fuzzy Matching)**: Si te equivocas al teclear (ej. `machin learne`), el sistema lo corrige buscando la palabra más cercana en el vocabulario del dataset. Palabras y siglas clave como CNN, GAN, BERT están protegidas de este cambio.
* **Comodines (Wildcards)**: Puedes usar el asterisco `*` o el signo de interrogación `?`. Por ejemplo, `learn*` expandirá internamente a todas las variantes (learning, learned, etc.) presentes en el dataset.
* La consulta procesada final es la que se envía al motor tradicional, mejorando la precisión de búsqueda sin alterar la metodología original de similitud Jaccard / Coseno.

## Fase 3: Buscador por Embeddings

Se ha incorporado una fase experimental e independiente para búsqueda semántica utilizando modelos de Deep Learning:
* Este método usa únicamente la columna **Abstracts** de los artículos.
* Cada abstract se convierte en un vector numérico (embedding) usando el modelo de lenguaje `sentence-transformers/all-MiniLM-L6-v2`.
* La consulta también se convierte en embedding previo paso por el **Procesamiento Inteligente** (traducción, correcciones).
* La comparación se realiza matemáticamente con los embeddings de los abstracts mediante **Similitud Coseno**.
* Las recomendaciones también se calculan usando la matriz documento-documento de embeddings.
* Las recomendaciones excluyen los 10 resultados principales, igual que en el método clásico.

Para probar la Fase 3 de manera aislada (por consola):
1. Descarga el modelo localmente:
   ```bash
   python scripts/download_embedding_model.py
   ```
2. Genera los embeddings y la matriz de similitud (esto creará archivos `.npy` y `.csv` en `data/`):
   ```bash
   python scripts/generate_embeddings.py
   ```
3. Ejecuta el buscador interactivo por consola:
   ```bash
   python scripts/test_embedding_search.py
   ```

**Integración Web:**
Si ejecutas `python backend/app.py`, el sistema cargará los embeddings a memoria RAM automáticamente y habilitará la pestaña **Buscador por Embeddings** en la aplicación web, separando visualmente ambos motores.
