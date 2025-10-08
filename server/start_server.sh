#!/bin/bash
# Script para iniciar el servidor de la API de recomendación de películas

echo "🎬 Iniciando servidor de recomendación de películas..."
echo ""

# Verificar que el entorno virtual existe
if [ ! -d ".venv" ]; then
    echo "❌ Error: No se encontró el entorno virtual (.venv)"
    echo "Por favor ejecuta: python3 -m venv .venv"
    exit 1
fi

# Activar entorno virtual
echo "📦 Activando entorno virtual..."
source .venv/bin/activate

# Verificar que las dependencias están instaladas
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "❌ Error: Dependencias no instaladas"
    echo "Por favor ejecuta: pip install -r requirements.txt"
    exit 1
fi

# Verificar que existen los datos
if [ ! -d "data" ] || [ ! -d "data/modeltrain" ]; then
    echo "❌ Error: No se encontraron los datos en data/ o data/modeltrain/"
    exit 1
fi

echo "✅ Verificaciones completadas"
echo ""
echo "🚀 Iniciando servidor en http://localhost:8000"
echo "📚 Documentación: http://localhost:8000/docs"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

# Iniciar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
