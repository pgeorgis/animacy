"""Animacy classification of nouns."""
from .classifier import AnimacyClassifier
from .constants import (ANIMATE_TAG, DEFAULT_ARCHITECTURE,
                        DEFAULT_EMBEDDINGS_MODEL, INANIMATE_TAG)
from .embeddings import (EmbeddingsModel, FastTextEmbeddingsModel,
                         init_embeddings_model)
from .paths import find_classifier

__all__ = [
    "AnimacyClassifier",
    "EmbeddingsModel",
    "FastTextEmbeddingsModel",
    "init_embeddings_model",
    "find_classifier",
    "ANIMATE_TAG",
    "INANIMATE_TAG",
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_EMBEDDINGS_MODEL",
]
