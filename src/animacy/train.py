# Adapted from similar work by Tepei & Bloem (2024) using WordNet annotations for animacy classification in Romanian: https://github.com/mariatepei/RO-animacy/blob/main/animacy-code.ipynb

import logging
from typing import Generator, Iterable

import numpy as np
import torch
from germanetpy.germanet import Germanet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from constants import (ANIMATE_EMBEDDINGS_FILE, ANIMATE_SEED_WORD, ANIMATE_TAG,
                       EMBEDDINGS_MODEL, GERMANET_DATA,
                       INANIMATE_EMBEDDINGS_FILE, INANIMATE_SEED_WORD,
                       INANIMATE_TAG)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingsModel:
    def __init__(self, name: str):
        self.name = name
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def get_word_embedding(self, word: str) -> np.ndarray:
        inputs = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0,1].numpy()
    
    def get_batch_embedding(self, batch: Iterable[str]) -> Generator:
        return (self.get_word_embedding(word) for word in batch)
    
    def batch_save_embeddings(self, batch: Iterable[str], filename: str) -> np.ndarray:
        embeddings = np.vstack(list(tqdm(self.get_batch_embedding(batch), total=len(batch))))
        np.save(filename, embeddings)
        return embeddings
    

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


def get_all_hyponym_lemmas(germanet: Germanet, seed_word: str) -> set:
    lemmas = set()
    for synset in germanet.get_synsets_by_orthform(seed_word):
        hyponyms = synset.all_hyponyms()
        for item in hyponyms:
            for lexunit in item.lexunits:
                lemmas.update(lexunit.get_all_orthforms())
    return lemmas

# Extract all animate and inanimate German lemmas recursively from hyponyms of seed words
germanet = Germanet(GERMANET_DATA)
animates = get_all_hyponym_lemmas(germanet, seed_word=ANIMATE_SEED_WORD)
inanimates = get_all_hyponym_lemmas(germanet, seed_word=INANIMATE_SEED_WORD)

# Get word embeddings of animate and inanimate nouns
embeddings_model = EmbeddingsModel(name=EMBEDDINGS_MODEL)
# animate_embeddings = embeddings_model.batch_save_embeddings(animates, ANIMATE_EMBEDDINGS_FILE)
# inanimate_embeddings = embeddings_model.batch_save_embeddings(inanimates, INANIMATE_EMBEDDINGS_FILE)
animate_embeddings = np.load(ANIMATE_EMBEDDINGS_FILE)
inanimate_embeddings = np.load(INANIMATE_EMBEDDINGS_FILE)

# Get full train/test set of embeddings and corresponding labels
all_embeddings = np.vstack([animate_embeddings, inanimate_embeddings])
labels = np.concatenate([
    np.full(len(animates), ANIMATE_TAG),
    np.full(len(inanimates), INANIMATE_TAG)
])

# Train and evaluate Random Forest Classifier
rf_model = AnimacyClassifier(architecture="rf", X=all_embeddings, y=labels)
# Precision: 0.939 | Recall: 0.872 | F1: 0.905

# Train and evaluate Multi-Layer Perceptron Model
mlp_model = AnimacyClassifier(architecture="mlp", X=all_embeddings, y=labels)
# Precision: 0.948 | Recall: 0.928 | F1: 0.938

# Train and evaluate KNNeighbors model
knn_model = AnimacyClassifier(architecture="knn", X=all_embeddings, y=labels)
# Precision: 0.916 | Recall: 0.917 | F1: 0.917
