"""Servicio de recomendación de películas usando TensorFlow Recommenders.

Este módulo proporciona la clase Movie que encapsula la funcionalidad
de carga de datos, modelo y predicciones.
"""
import os
import logging
from pathlib import Path
from typing import List, Union, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the MovieModel class so Keras can deserialize the custom model.
from app.movieModel import MovieModel


class Movie:
    """Servicio para manejo de datos y predicciones de películas.

    Carga los datos desde el directorio `data` (por defecto) y el modelo
    guardado en `data/modeltrain`.
    
    Attributes:
        data_dir: Directorio donde se encuentran los archivos CSV
        model_path: Ruta al modelo entrenado guardado
        movies_ds: Dataset de TensorFlow con películas
        ratings_df: DataFrame de pandas con ratings
        unique_movie_titles: Array con títulos únicos de películas
        unique_user_ids: Array con IDs únicos de usuarios
        model: Modelo de TensorFlow cargado
    """

    def __init__(
        self, 
        data_dir: str = "data", 
        model_path: str = "data/modeltrain"
    ) -> None:
        """Inicializa el servicio de películas.
        
        Args:
            data_dir: Directorio con los archivos CSV de datos
            model_path: Ruta al directorio con el modelo entrenado
            
        Raises:
            FileNotFoundError: Si no se encuentran los archivos necesarios
            ValueError: Si los datos están corruptos o incompletos
        """
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)

        # datasets y metadatos
        self.movies_ds: Optional[tf.data.Dataset] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.unique_movie_titles: Optional[np.ndarray] = None
        self.unique_user_ids: Optional[np.ndarray] = None

        # modelo cargado
        self.model: Optional[tf.keras.Model] = None

        # Inicializar
        logger.info("Inicializando servicio de películas...")
        self._load_data()
        self._load_model()
        logger.info("Servicio de películas inicializado correctamente")

    def _load_data(self) -> None:
        """Carga y prepara los datasets (movies y ratings).
        
        Raises:
            FileNotFoundError: Si algún archivo CSV no existe
            pd.errors.EmptyDataError: Si algún CSV está vacío
        """
        logger.info("Cargando archivos CSV...")
        
        # Validar que existan los archivos
        required_files = ['credits.csv', 'keywords.csv', 'movies_metadata.csv', 'ratings_small.csv']
        for file in required_files:
            file_path = self.data_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo requerido no encontrado: {file_path}")
        
        try:
            credits = pd.read_csv(self.data_dir / 'credits.csv')
            keywords = pd.read_csv(self.data_dir / 'keywords.csv')
            movies = pd.read_csv(self.data_dir / 'movies_metadata.csv')
        except Exception as e:
            logger.error(f"Error al cargar archivos CSV: {e}")
            raise

        # Limpiar columnas problemáticas y filas con tipos incorrectos
        logger.info("Limpiando datos de películas...")
        drop_cols = [
            'belongs_to_collection', 'homepage', 'imdb_id', 
            'poster_path', 'status', 'title', 'video'
        ]
        movies = movies.drop(
            columns=[c for c in drop_cols if c in movies.columns], 
            errors='ignore'
        )

        # Algunos índices en este dataset son filas con tipos inválidos
        bad_indices = [19730, 29503, 35587]
        movies = movies.drop(
            index=[idx for idx in bad_indices if idx in movies.index], 
            errors='ignore'
        )

        # Asegurar tipo correcto para id
        if 'id' in movies.columns:
            movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
            movies = movies.dropna(subset=['id'])
            movies['id'] = movies['id'].astype('int64')

        # Merge de datasets
        logger.info("Combinando datasets...")
        df = movies.merge(keywords, on='id', how='left').merge(credits, on='id', how='left')

        # Rellenar valores nulos
        df['original_language'] = df.get('original_language', '').fillna('')
        df['runtime'] = df.get('runtime', 0).fillna(0)
        df['tagline'] = df.get('tagline', '').fillna('')

        df.dropna(inplace=True)

        # Cargar ratings
        logger.info("Cargando ratings...")
        ratings_df = pd.read_csv(self.data_dir / 'ratings_small.csv')
        # Procesar timestamps
        ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        ratings_df.drop('timestamp', axis=1, inplace=True)

        # Merge ratings con información de películas
        ratings_df = ratings_df.merge(
            df[['id', 'original_title', 'genres', 'overview']], 
            left_on='movieId', 
            right_on='id', 
            how='left'
        )
        ratings_df = ratings_df[~ratings_df['id'].isna()]
        ratings_df.drop('id', axis=1, inplace=True)
        ratings_df.reset_index(drop=True, inplace=True)
        
        # Preparar DataFrame de películas
        movies_df = df[['id', 'original_title']].copy()
        movies_df.rename(columns={'id': 'movieId'}, inplace=True)

        # Convertir userId a string para consistencia con el modelo
        ratings_df['userId'] = ratings_df['userId'].astype(str)

        # Conservar datos en memoria para uso en el servicio
        self.ratings_df = ratings_df

        # Crear TensorFlow Datasets
        logger.info("Creando TensorFlow Datasets...")
        ratings = tf.data.Dataset.from_tensor_slices(
            dict(ratings_df[['userId', 'original_title', 'rating']])
        )
        movies_ds = tf.data.Dataset.from_tensor_slices(
            dict(movies_df[['original_title']])
        )

        # Mapear datasets al formato correcto
        ratings = ratings.map(lambda x: {
            "original_title": x["original_title"],
            "userId": x["userId"],
            "rating": float(x["rating"])
        })

        movies_ds = movies_ds.map(lambda x: x["original_title"])

        # Guardar referencias
        self.movies_ds = movies_ds

        # Calcular metadatos (listas únicas)
        logger.info("Calculando metadatos...")
        movie_titles = movies_ds.batch(1_000)
        user_ids = ratings.batch(1_000).map(lambda x: x["userId"])

        # Convertir a numpy arrays para StringLookup vocabulary
        self.unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
        self.unique_user_ids = np.unique(np.concatenate(list(user_ids)))

        logger.info(f'Total ratings: {len(self.ratings_df):,}')
        logger.info(f'Películas únicas: {len(self.unique_movie_titles):,}')
        logger.info(f'Usuarios únicos: {len(self.unique_user_ids):,}')

    def _load_model(self) -> None:
        """Carga el modelo Keras/TFRS guardado.
        
        Soporta tanto modelos nuevos (.keras, .h5) como modelos legacy (SavedModel).
        Para modelos SavedModel (formato antiguo), usa tf.saved_model.load() que es
        compatible con Keras 3.
        
        Raises:
            FileNotFoundError: Si no se encuentra el directorio del modelo
            Exception: Si hay errores al cargar el modelo
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f'Ruta del modelo no encontrada: {self.model_path}')

        logger.info(f"Cargando modelo desde {self.model_path}...")
        
        # Detectar el tipo de modelo
        model_format = self._detect_model_format()
        logger.info(f"Formato de modelo detectado: {model_format}")
        
        if model_format == "savedmodel":
            # Modelo legacy en formato SavedModel (TensorFlow < 2.16)
            logger.info("Cargando modelo legacy con tf.saved_model.load()...")
            try:
                # Cargar el SavedModel completo
                loaded = tf.saved_model.load(str(self.model_path))
                
                # Crear un wrapper que expone los métodos que necesitamos
                class SavedModelWrapper:
                    """Wrapper para hacer el SavedModel compatible con nuestra API."""
                    def __init__(self, loaded_model):
                        self.loaded = loaded_model
                        # Extraer los sub-modelos si existen
                        try:
                            # Los sub-modelos son ConcreteFunction, necesitamos envolverlos
                            self._user_model_fn = loaded_model.user_model
                            self._movie_model_fn = loaded_model.movie_model
                            self._rating_model_fn = loaded_model.rating_model
                            
                            # Crear callables que TensorFlow puede usar
                            self.user_model = lambda x: self._user_model_fn(x)
                            self.movie_model = lambda x: self._movie_model_fn(x)
                            self.rating_model = lambda x: self._rating_model_fn(x)
                            
                            logger.info("Sub-modelos extraídos y envueltos exitosamente")
                        except AttributeError as e:
                            logger.warning(f"No se pudieron extraer sub-modelos: {e}")
                            # Intentar acceder de otra forma
                            self.user_model = None
                            self.movie_model = None
                            self.rating_model = None
                
                self.model = SavedModelWrapper(loaded)
                logger.info("Modelo SavedModel cargado exitosamente")
                
            except Exception as e:
                logger.error(f"Error al cargar SavedModel: {e}")
                raise RuntimeError(f"No se pudo cargar el modelo SavedModel: {e}") from e
                
        else:
            # Modelo en formato Keras 3 (.keras) o H5 (.h5)
            logger.info("Intentando cargar con keras.models.load_model()...")
            try:
                self.model = tf.keras.models.load_model(
                    str(self.model_path), 
                    compile=False, 
                    custom_objects={"MovieModel": MovieModel}
                )
                logger.info("Modelo cargado exitosamente con custom_objects")
            except Exception as e:
                logger.warning(f"Error al cargar con custom_objects: {e}")
                logger.info("Intentando cargar sin custom_objects...")
                try:
                    self.model = tf.keras.models.load_model(
                        str(self.model_path), 
                        compile=False
                    )
                    logger.info("Modelo cargado exitosamente sin custom_objects")
                except Exception as e2:
                    logger.error(f"Error al cargar el modelo: {e2}")
                    raise RuntimeError(f"No se pudo cargar el modelo: {e2}") from e2
    
    def _detect_model_format(self) -> str:
        """Detecta el formato del modelo guardado.
        
        Returns:
            'savedmodel' si es TensorFlow SavedModel
            'keras' si es formato .keras
            'h5' si es formato HDF5
            'unknown' si no se puede determinar
        """
        if self.model_path.is_file():
            # Es un archivo único
            if self.model_path.suffix == '.keras':
                return 'keras'
            elif self.model_path.suffix in ['.h5', '.hdf5']:
                return 'h5'
        elif self.model_path.is_dir():
            # Es un directorio, verificar si es SavedModel
            saved_model_pb = self.model_path / 'saved_model.pb'
            if saved_model_pb.exists():
                return 'savedmodel'
            
            # Verificar si contiene un archivo .keras o .h5
            keras_files = list(self.model_path.glob('*.keras'))
            if keras_files:
                return 'keras'
            
            h5_files = list(self.model_path.glob('*.h5'))
            if h5_files:
                return 'h5'
        
        return 'savedmodel'  # Asumir SavedModel por defecto para modelos legacy

    def recommend(self, user: Union[int, str], top_n: int = 3) -> List[str]:
        """Devuelve los títulos de películas recomendados para un usuario.
        
        Args:
            user: ID del usuario (puede ser int o str)
            top_n: Número de recomendaciones a devolver (default: 3)
            
        Returns:
            Lista de títulos de películas recomendadas
            
        Raises:
            RuntimeError: Si el modelo no está cargado
            ValueError: Si top_n es inválido
        """
        if self.model is None:
            raise RuntimeError('Modelo no cargado')
        
        if top_n < 1:
            raise ValueError(f'top_n debe ser mayor a 0, recibido: {top_n}')

        try:
            logger.debug(f"Generando recomendaciones para usuario {user}...")
            
            # Detectar si es un SavedModel (tiene el wrapper) o modelo Keras normal
            is_savedmodel = hasattr(self.model, 'loaded')
            
            if is_savedmodel:
                # Para SavedModel, calcular scores manualmente
                return self._recommend_savedmodel(user, top_n)
            else:
                # Para modelos Keras normales, usar BruteForce
                return self._recommend_keras(user, top_n)
                
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}")
            raise
    
    def _recommend_keras(self, user: Union[int, str], top_n: int) -> List[str]:
        """Recomendaciones usando BruteForce para modelos Keras."""
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)
        index.index_from_dataset(
            tf.data.Dataset.zip((
                self.movies_ds.batch(100), 
                self.movies_ds.batch(100).map(self.model.movie_model)
            ))
        )

        _, titles = index(tf.constant([str(user)]))

        decoded = [t.decode('utf-8') for t in titles[0, :top_n].numpy()]
        logger.debug(f"Recomendaciones generadas: {decoded}")
        return decoded
    
    def _recommend_savedmodel(self, user: Union[int, str], top_n: int) -> List[str]:
        """Recomendaciones calculando scores manualmente para SavedModel.
        
        Este método calcula el producto punto entre el embedding del usuario
        y los embeddings de todas las películas para rankear.
        """
        # Obtener embedding del usuario
        user_embedding = self.model.user_model(tf.constant([str(user)]))  # [1, 64]
        
        # Calcular embeddings para todas las películas en batches
        all_movie_titles = []
        all_movie_embeddings = []
        
        batch_size = 1000
        movies_batched = self.movies_ds.batch(batch_size)
        
        for batch in movies_batched:
            # batch es un tensor de strings con shape [batch_size]
            movie_emb = self.model.movie_model(batch)  # [batch_size, 64]
            all_movie_embeddings.append(movie_emb)
            
            # Guardar títulos
            for title in batch.numpy():
                all_movie_titles.append(title.decode('utf-8'))
        
        # Concatenar todos los embeddings
        all_movie_embeddings = tf.concat(all_movie_embeddings, axis=0)  # [num_movies, 64]
        
        # Calcular scores (producto punto)
        # user_embedding: [1, 64], all_movie_embeddings: [num_movies, 64]
        scores = tf.matmul(user_embedding, all_movie_embeddings, transpose_b=True)  # [1, num_movies]
        scores = tf.squeeze(scores, axis=0)  # [num_movies]
        
        # Obtener top-N
        top_indices = tf.argsort(scores, direction='DESCENDING')[:top_n]
        
        # Obtener títulos correspondientes
        recommendations = [all_movie_titles[i] for i in top_indices.numpy()]
        
        logger.debug(f"Recomendaciones generadas: {recommendations}")
        return recommendations

    def predict_rating(self, user: Union[int, str], movie_title: str) -> float:
        """Predice el rating numérico para un usuario y una película dada.
        
        Args:
            user: ID del usuario (puede ser int o str)
            movie_title: Título de la película
            
        Returns:
            Rating predicho (típicamente entre 0.5 y 5.0)
            
        Raises:
            RuntimeError: Si el modelo no está cargado
            ValueError: Si los parámetros son inválidos
        """
        if self.model is None:
            raise RuntimeError('Modelo no cargado')
        
        if not movie_title or not isinstance(movie_title, str):
            raise ValueError(f'Título de película inválido: {movie_title}')

        try:
            logger.debug(f"Prediciendo rating para usuario {user} y película '{movie_title}'...")
            
            # Preparar tensores de entrada
            movie_tensor = np.array([movie_title], dtype=object)
            user_tensor = np.array([str(user)], dtype=object)

            # Obtener embeddings entrenados
            trained_movie_embeddings = self.model.movie_model(movie_tensor)
            trained_user_embeddings = self.model.user_model(user_tensor)

            # Predecir rating
            predicted = self.model.rating_model(
                tf.concat([trained_user_embeddings, trained_movie_embeddings], axis=1)
            )

            value = float(predicted.numpy()[0][0])
            logger.debug(f"Rating predicho: {value:.2f}")
            return value
        except Exception as e:
            logger.error(f"Error al predecir rating: {e}")
            raise
    
    def get_movie_titles(self) -> List[str]:
        """Retorna la lista de todos los títulos de películas disponibles.
        
        Returns:
            Lista de títulos de películas
        """
        if self.unique_movie_titles is None:
            return []
        return [title.decode('utf-8') if isinstance(title, bytes) else title 
                for title in self.unique_movie_titles]
    
    def get_user_ids(self) -> List[str]:
        """Retorna la lista de todos los IDs de usuarios disponibles.
        
        Returns:
            Lista de IDs de usuario
        """
        if self.unique_user_ids is None:
            return []
        return [uid.decode('utf-8') if isinstance(uid, bytes) else uid 
                for uid in self.unique_user_ids]

