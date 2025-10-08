"""API FastAPI para el servicio de recomendación de películas.

Proporciona endpoints para:
- Predicción de ratings
- Recomendaciones de películas
- Health check del servicio
"""
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.movie import Movie

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommendation API",
    description="API para predicción de ratings y recomendaciones de películas",
    version="2.0.0"
)

# Inicializar el servicio de películas (se hará en el evento startup para evitar
# trabajo pesado en el momento de importar el módulo)
movie_service: Movie = None


# Modelos Pydantic para respuestas
class PredictionResponse(BaseModel):
    """Respuesta para predicción de rating."""
    user: str = Field(..., description="ID del usuario")
    movie: str = Field(..., description="Título de la película")
    predicted_rating: float = Field(..., description="Rating predicho")


class RecommendationResponse(BaseModel):
    """Respuesta para recomendaciones."""
    user: str = Field(..., description="ID del usuario")
    top_n: int = Field(..., description="Número de recomendaciones solicitadas")
    recommendations: List[str] = Field(..., description="Lista de títulos recomendados")


class HealthResponse(BaseModel):
    """Respuesta para health check."""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    total_movies: int = Field(0, description="Número total de películas")
    total_users: int = Field(0, description="Número total de usuarios")


class ErrorResponse(BaseModel):
    """Respuesta para errores."""
    detail: str = Field(..., description="Descripción del error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Inicializa el servicio al arrancar la aplicación."""
    global movie_service
    if movie_service is None:
        try:
            logger.info("Iniciando carga del servicio de películas...")
            movie_service = Movie()
            logger.info("Servicio de películas cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar el servicio: {e}")
            raise


@app.on_event("shutdown")
async def shutdown_event():
    """Limpia recursos al apagar la aplicación."""
    logger.info("Cerrando servicio de películas...")
    # Aquí podrías agregar limpieza de recursos si fuera necesario


@app.get(
    "/predict/{user}/{movie_title}",
    response_model=PredictionResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Error interno del servidor"},
        404: {"model": ErrorResponse, "description": "Usuario o película no encontrados"}
    },
    summary="Predice el rating de una película para un usuario",
    tags=["Predicción"]
)
async def predict(user: int, movie_title: str):
    """Predice el rating que un usuario daría a una película específica.
    
    Args:
        user: ID del usuario
        movie_title: Título de la película (debe coincidir exactamente)
        
    Returns:
        Objeto con el usuario, película y rating predicho
    """
    if movie_service is None:
        logger.error("Intento de usar el servicio antes de inicializarlo")
        raise HTTPException(
            status_code=503, 
            detail="Servicio no inicializado"
        )
    
    try:
        value = movie_service.predict_rating(user, movie_title)
        return PredictionResponse(
            user=str(user),
            movie=movie_title,
            predicted_rating=round(value, 2)
        )
    except ValueError as e:
        logger.warning(f"Error de validación en predict: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al predecir rating: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@app.get(
    "/recommend/{user}",
    response_model=RecommendationResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Error interno del servidor"},
        404: {"model": ErrorResponse, "description": "Usuario no encontrado"}
    },
    summary="Obtiene recomendaciones de películas para un usuario",
    tags=["Recomendación"]
)
async def recommend(
    user: int,
    top_n: int = Query(default=3, ge=1, le=50, description="Número de recomendaciones (1-50)")
):
    """Genera una lista de películas recomendadas para un usuario específico.
    
    Args:
        user: ID del usuario
        top_n: Número de recomendaciones a retornar (default: 3, máx: 50)
        
    Returns:
        Objeto con el usuario y lista de películas recomendadas
    """
    if movie_service is None:
        logger.error("Intento de usar el servicio antes de inicializarlo")
        raise HTTPException(
            status_code=503, 
            detail="Servicio no inicializado"
        )
    
    try:
        recs = movie_service.recommend(user, top_n=top_n)
        return RecommendationResponse(
            user=str(user),
            top_n=top_n,
            recommendations=recs
        )
    except ValueError as e:
        logger.warning(f"Error de validación en recommend: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al recomendar: {e}")
        raise HTTPException(status_code=500, detail=f"Error al recomendar: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verifica el estado del servicio",
    tags=["Sistema"]
)
async def health():
    """Endpoint de health check para verificar el estado del servicio.
    
    Returns:
        Información sobre el estado del servicio y estadísticas básicas
    """
    model_loaded = movie_service is not None
    total_movies = 0
    total_users = 0
    
    if model_loaded:
        try:
            total_movies = len(movie_service.get_movie_titles())
            total_users = len(movie_service.get_user_ids())
        except Exception as e:
            logger.warning(f"Error al obtener estadísticas: {e}")
    
    return HealthResponse(
        status="ok" if model_loaded else "service_unavailable",
        model_loaded=model_loaded,
        total_movies=total_movies,
        total_users=total_users
    )


@app.get(
    "/movies",
    response_model=List[str],
    summary="Lista todas las películas disponibles",
    tags=["Información"]
)
async def list_movies(
    limit: int = Query(default=100, ge=1, le=1000, description="Número máximo de películas a retornar")
):
    """Retorna una lista de títulos de películas disponibles.
    
    Args:
        limit: Número máximo de películas a retornar (default: 100, máx: 1000)
        
    Returns:
        Lista de títulos de películas
    """
    if movie_service is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    
    try:
        movies = movie_service.get_movie_titles()
        return movies[:limit]
    except Exception as e:
        logger.error(f"Error al listar películas: {e}")
        raise HTTPException(status_code=500, detail=str(e))