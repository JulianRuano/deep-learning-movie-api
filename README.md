# Sistema de Recomendación de Películas con Deep Learning

Sistema híbrido de recomendación basado en filtrado colaborativo utilizando TensorFlow Recommenders. Combina embeddings neuronales con redes profundas para predecir ratings y recomendar películas personalizadas.

![Rating](/web/img/image-1.png)

## Arquitectura del Modelo
### Embeddings
- **Dimensión de embedding**: 64 dimensiones para usuarios y películas
- **Capa de Usuario**: Convierte 671 IDs de usuario en vectores densos de 64 elementos
- **Capa de Película**: Transforma 42,373 títulos únicos en representaciones de 64 dimensiones

Los embeddings aprenden automáticamente relaciones semánticas entre usuarios y películas durante el entrenamiento, capturando patrones de preferencias y similitudes.

### Red Neuronal de Ratings
Arquitectura secuencial que procesa embeddings concatenados:

Input: [user_embedding + movie_embedding] → 128 dimensiones
↓
Dense(256, activation='relu')
↓
Dense(128, activation='relu')
↓
Dense(1) → Rating predicho


## Función de Activación ReLU

**ReLU (Rectified Linear Unit)** se aplica en las capas ocultas de 256 y 128 neuronas.

**Fórmula**: `f(x) = max(0, x)`

- **Para x > 0**: La neurona pasa el valor sin cambios (f(x) = x)
- **Para x ≤ 0**: La neurona se desactiva (f(x) = 0)

**Ventajas**:
- Evita el problema del gradiente desvaneciente
- Computacionalmente eficiente (solo una operación de umbral)
- Promueve activación dispersa (sparsity)
- Acelera convergencia en redes profundas

## Estrategia de Entrenamiento

### Aprendizaje Multi-Tarea
El modelo optimiza dos objetivos simultáneamente:

1. **Tarea de Ranking** (rating_weight=1.0):
   - Predice valores de rating (1-5 estrellas)
   - Función de pérdida: Mean Squared Error (MSE)
   - Métrica: Root Mean Squared Error (RMSE)

2. **Tarea de Retrieval** (retrieval_weight=1.0):
   - Recupera películas relevantes por similitud de embeddings
   - Métrica: FactorizedTopK accuracy (top-1, top-5, top-10, top-50, top-100)

**Función de pérdida combinada**:

Total Loss = rating_weight × rating_loss + retrieval_weight × retrieval_loss

## Datos y Entrenamiento

### Dataset
- **Total de interacciones**: 43,188 ratings
- **Usuarios únicos**: 671
- **Películas únicas**: 42,373
- **División**:
  - Entrenamiento: 35,000 ejemplos
  - Prueba: 8,188 ejemplos

### Configuración de Entrenamiento
- **Optimizador**: Adagrad (learning rate=0.1)
- **Épocas**: 15
- **Batch size**: 1,000
- **Shuffle**: 100,000 (sin re-shuffle entre épocas)

### Resultados

| Época | RMSE | Top-100 Accuracy |
|-------|------|------------------|
| 1     | 1.5053 | 17.27% |
| 5     | 0.9536 | 47.20% |
| 10    | 0.8932 | 56.88% |
| 15    | 0.8537 | 62.67% |

**Interpretación**: Al finalizar el entrenamiento, el modelo logra un error promedio de 0.85 estrellas en las predicciones de rating y coloca películas relevantes dentro del top-100 recomendaciones en el 63% de los casos.

## API REST con FastAPI

El modelo entrenado se despliega mediante una API REST construida con FastAPI, ubicada en la carpeta `/server`. La API expone el modelo para realizar predicciones y generar recomendaciones en tiempo real.

### Iniciar el Servidor

Ejecuta el script de inicio incluido:
./start_server.sh


El script realiza las siguientes verificaciones automáticas:
- ✅ Existencia del entorno virtual (`.venv`)
- ✅ Instalación de dependencias (FastAPI, Uvicorn)
- ✅ Presencia de datos en `data/` y `data/modeltrain/`

**URLs disponibles**:
- Servidor: `http://localhost:8000`
- Documentación interactiva: `http://localhost:8000/docs`

### Endpoints Disponibles

#### 1. Predecir Rating
GET /predict/{user_id}/{movie_title}

text

**Ejemplo**:
GET http://localhost:8000/predict/1/Batman

text

**Respuesta**:
{
"user": "1",
"movie": "Batman",
"predicted_rating": 3.49
}

text

Retorna el rating predicho (1-5 estrellas) para la combinación usuario-película especificada.

#### 2. Obtener Recomendaciones
GET /recommend/{user_id}?top_n={cantidad}

text

**Ejemplo**:
GET http://localhost:8000/recommend/1?top_n=3

text

**Respuesta**:
{
"user": "1",
"top_n": 3,
"recommendations": [
"Vivement dimanche!",
"American Pie",
"Rocky III"
]
}

text

Genera las top-N películas recomendadas personalizadas para el usuario especificado basándose en sus preferencias aprendidas.

#### 3. Estado del Sistema

GET /health

text

**Ejemplo**:
GET http://localhost:8000/health

text

**Respuesta**:
{
"status": "ok",
"model_loaded": true,
"total_movies": 42373,
"total_users": 671
}

text

Verifica que el modelo esté cargado correctamente y muestra estadísticas del dataset.

#### 4. Listar Películas
GET /movies?limit={cantidad}

text

**Ejemplo**:
GET http://localhost:8000/movies?limit=5

text

**Respuesta**:
[
"!Women Art Revolution",
"#1 Cheerleader Camp",
"#Horror",
"#Pellichoopulu",
"#SELFIEPARTY"
]

text

Retorna una lista de títulos de películas disponibles en el catálogo (útil para autocompletado en el frontend).

## Frontend

La aplicación incluye una interfaz web que consume la API REST, permitiendo a los usuarios:
- Ingresar el nombre de una película para obtener predicciones de rating
- Explorar recomendaciones personalizadas por usuario
- Visualizar resultados del modelo entrenado en tiempo real

El frontend se comunica con el servidor FastAPI para procesar solicitudes utilizando el modelo de Deep Learning cargado en memoria.

![Rating](/web/img/image.png)