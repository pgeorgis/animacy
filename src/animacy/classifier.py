import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from constants import ANIMATE_TAG
from embeddings import EmbeddingsModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AnimacyClassifier:
    def __init__(self,
                 architecture: str,
                 X: np.ndarray,
                 y: np.ndarray,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 ):
        self.architecture = architecture
        self.full_name = self._get_model_architecture_name()
        self.test_size = test_size
        self.random_state = random_state
        self._is_fitted = False
        self.model = self._get_model()
        self.score = self.fit_and_evaluate(X=X, y=y, test_size=test_size, random_state=random_state)

    def _get_model_architecture_name(self):
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
    
    def classify(self, word: str, embedding_model: EmbeddingsModel) -> str:
        embedding = embedding_model.get_word_embedding(word)
        embedding = embedding.reshape(1, -1)
        pred = self.model.predict(embedding)
        return pred[0]

    def save(self, model_file: str):
        model_file = os.path.abspath(model_file)
        pickle.dump(self.model, open(model_file, 'wb'))
        logger.info(f"Saved model to {model_file}")

    def load(self, model_file: str):
        model_file = os.path.abspath(model_file)
        self.model = pickle.load(open(model_file, 'rb'))
        logger.info(f"Loaded model from {model_file}")
