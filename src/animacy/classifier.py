import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .constants import (ANIMATE_TAG, DEFAULT_ARCHITECTURE,
                        DEFAULT_EMBEDDINGS_MODEL)
from .embeddings import EmbeddingsModel, init_embeddings_model
from .paths import find_classifier

logger = logging.getLogger(__name__)


class AnimacyClassifier:
    def __init__(self,
                 architecture: str,
                 X: np.ndarray = None,
                 y: np.ndarray = None,
                 test_size: float = 0.25,
                 random_state: int = 42,
                 ):
        self.architecture = architecture
        self.full_name = self._get_model_architecture_name()
        self.test_size = test_size
        self.random_state = random_state
        self._is_fitted = False
        self.model = self._get_model()
        self.encoder: EmbeddingsModel = None
        if X is not None and y is not None:
            self.score = self.fit_and_evaluate(X=X, y=y, test_size=test_size, random_state=random_state)

    def _get_model_architecture_name(self) -> str:
        if self.architecture == "rf":
            return "Random Forest"
        elif self.architecture == "mlp":
            return "Multi-Layer Perceptron"
        elif self.architecture == "knn":
            return "K-Nearest Neighbors"
        else:
            raise ValueError(f"Invalid model architecture '{self.architecture}'. Valid architectures: 'rf', 'mlp', 'knn'")

    def _get_model(self):
        if self.architecture == "rf":
            return RandomForestClassifier(random_state=self.random_state)
        elif self.architecture == "mlp":
            return MLPClassifier(random_state=self.random_state)
        elif self.architecture == "knn":
            return KNeighborsClassifier()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Fitting {self.full_name} animacy classifier model...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 ) -> tuple[float, float, float]:
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred, pos_label=ANIMATE_TAG)
        recall = recall_score(y_test, y_pred, pos_label=ANIMATE_TAG)
        f1 = f1_score(y_test, y_pred, pos_label=ANIMATE_TAG)
        logger.info(f"{self.full_name} animacy classifier model evaluation: Precision: {round(precision, 3)} | Recall: {round(recall, 3)} | F1: {round(f1, 3)}")
        return precision, recall, f1

    def fit_and_evaluate(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         ) -> tuple[float, float, float]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.fit(X_train, y_train)
        return self.evaluate(X_test, y_test)
    
    def classify(self,
                 word: str,
                 encoder: EmbeddingsModel = None,
                 ) -> str:
        if encoder is None:
            # Use pre-loaded self.encoder if available and encoder is unspecified
            encoder = self.encoder
        if encoder is None:
            raise ValueError("`encoder` unspecified and no default encoder loaded into AnimacyClassifier!")
        embedding = encoder.get_word_embedding(word)
        embedding = embedding.reshape(1, -1)
        pred = self.model.predict(embedding)
        return pred[0]

    def save(self, model_path: str) -> None:
        model_path = os.path.abspath(model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved model to {model_path}")

    def load(self, model_path: str) -> None:
        model_path = os.path.abspath(model_path)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self._is_fitted = True
        logger.info(f"Loaded model from {model_path}")

    def load_encoder(self, embeddings_model_path: str) -> None:
        self.encoder = init_embeddings_model(embeddings_model_path)
        logger.info(f"Loaded encoder model from {embeddings_model_path}")

    @classmethod
    def from_pretrained(cls,
                        architecture: str = DEFAULT_ARCHITECTURE,
                        model_path: str = None,
                        encoder: str | EmbeddingsModel | None = DEFAULT_EMBEDDINGS_MODEL,
                        ) -> "AnimacyClassifier":
        """Load a trained classifier together with its word embeddings encoder.

        Args:
            architecture: 'rf', 'mlp', or 'knn'.
            model_path: Path to a classifier .pkl file. If None, the classifier is looked up
                in $ANIMACY_MODEL_DIR and then among the classifiers packaged with the library.
            encoder: Hugging Face ID of the embeddings model to load, an already-loaded
                embeddings model, or None to load no encoder (then `classify` must be
                passed an `encoder`).
        """
        classifier = cls(architecture=architecture)
        classifier.load(model_path if model_path is not None else find_classifier(architecture))
        if isinstance(encoder, str):
            classifier.load_encoder(encoder)
        else:
            classifier.encoder = encoder
        return classifier
