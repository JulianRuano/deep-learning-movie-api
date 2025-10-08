"""Modelo personalizado de TensorFlow Recommenders para películas.

Este módulo define el modelo MovieModel que combina tareas de ranking
y retrieval para generar recomendaciones y predicciones de ratings.
"""
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text, Tuple


class MovieModel(tfrs.models.Model):
    """Modelo híbrido de recomendación que combina ranking y retrieval.
    
    Este modelo usa embeddings para usuarios y películas, y combina dos tareas:
    - Ranking: Predice ratings explícitos
    - Retrieval: Aprende a emparejar usuarios con películas relevantes
    
    Attributes:
        movie_model: Sequential model para embeddings de películas
        user_model: Sequential model para embeddings de usuarios
        rating_model: Red neuronal para predecir ratings
        rating_task: Tarea de ranking con RMSE como métrica
        retrieval_task: Tarea de retrieval con FactorizedTopK
        rating_weight: Peso de la pérdida de rating
        retrieval_weight: Peso de la pérdida de retrieval
    """

    def __init__(
        self, 
        rating_weight: float, 
        retrieval_weight: float, 
        unique_movie_titles,
        unique_user_ids,
        movies
    ) -> None:
        """Inicializa el modelo con pesos de pérdida y vocabularios.
        
        Args:
            rating_weight: Peso para la pérdida de rating (típicamente 1.0)
            retrieval_weight: Peso para la pérdida de retrieval (típicamente 1.0)
            unique_movie_titles: Array de títulos únicos de películas
            unique_user_ids: Array de IDs únicos de usuarios
            movies: Dataset de TensorFlow con películas para métricas
        """
        super().__init__()

        embedding_dimension = 64

        # Modelos de embeddings para películas y usuarios
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, 
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                len(unique_movie_titles) + 1, 
                embedding_dimension,
                name="movie_embedding"
            )
        ], name="movie_model")
        
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, 
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                len(unique_user_ids) + 1, 
                embedding_dimension,
                name="user_embedding"
            )
        ], name="user_model")

        # Red neuronal para predecir ratings a partir de embeddings concatenados
        # Se puede hacer tan compleja como se necesite, siempre que devuelva un escalar
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_2"),
            tf.keras.layers.Dense(1, name="rating_output"),
        ], name="rating_model")

        # Tareas de entrenamiento
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # Pesos de pérdida
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass del modelo.
        
        Args:
            features: Diccionario con 'userId' y 'original_title'
            
        Returns:
            Tupla con (user_embeddings, movie_embeddings, rating_predictions)
        """
        # Extraer características de usuario y pasarlas por el modelo de usuario
        user_embeddings = self.user_model(features["userId"])
        
        # Extraer características de película y pasarlas por el modelo de película
        movie_embeddings = self.movie_model(features["original_title"])

        # Aplicar el modelo de rating a la concatenación de embeddings
        rating_predictions = self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        )

        return user_embeddings, movie_embeddings, rating_predictions

    def compute_loss(
        self, 
        features: Dict[Text, tf.Tensor], 
        training: bool = False
    ) -> tf.Tensor:
        """Calcula la pérdida combinada del modelo.
        
        Args:
            features: Diccionario con 'userId', 'original_title' y 'rating'
            training: Si está en modo entrenamiento
            
        Returns:
            Pérdida combinada ponderada (rating_loss + retrieval_loss)
        """
        # Extraer los ratings verdaderos
        ratings = features.pop("rating")

        # Obtener predicciones
        user_embeddings, movie_embeddings, rating_predictions = self(features)

        # Calcular pérdida para cada tarea
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        # Combinar usando los pesos de pérdida
        total_loss = (
            self.rating_weight * rating_loss + 
            self.retrieval_weight * retrieval_loss
        )
        
        return total_loss