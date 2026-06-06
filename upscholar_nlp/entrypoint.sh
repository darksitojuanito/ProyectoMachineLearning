#!/bin/bash
set -e

echo "Verificando dependencias de la Fase 3..."

# Verificar si el modelo de embeddings ya existe (por el volumen montado)
if [ ! -d "models/all-MiniLM-L6-v2" ]; then
    echo "Modelo local no encontrado. Descargando modelo de embeddings..."
    python scripts/download_embedding_model.py
else
    echo "Modelo de embeddings detectado."
fi

# Verificar si las matrices y el .npy existen
if [ ! -f "data/abstract_embeddings.npy" ]; then
    echo "Archivos de embeddings no encontrados. Generando matrices..."
    python scripts/generate_embeddings.py
else
    echo "Archivos de embeddings detectados."
fi

echo "Iniciando servidor backend UPScholar..."
exec python backend/app.py
