# Adapted from similar work by Tepei & Bloem (2024) using WordNet annotations for animacy classification in Romanian: https://github.com/mariatepei/RO-animacy/blob/main/animacy-code.ipynb

import logging
import os

import numpy as np
from germanetpy.germanet import Germanet

from constants import (ANIMATE_EMBEDDINGS_FILE, ANIMATE_SEED_WORD, ANIMATE_TAG,
                       EMBEDDINGS_MODEL, GERMANET_DATA,
                       INANIMATE_EMBEDDINGS_FILE, INANIMATE_SEED_WORD,
                       INANIMATE_TAG, PRETRAINED_ANIMACY_MODEL_DIR)
from models.embeddings import EmbeddingsModel

from .animacy_classifier import AnimacyClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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
# Uncomment lines below to regenerate embeddings from scratch rather than reloading
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
# Precision: 0.939 | Recall: 0.874 | F1: 0.906 [Animacy seed = <Mensch>]

# Train and evaluate Multi-Layer Perceptron Model
mlp_model = AnimacyClassifier(architecture="mlp", X=all_embeddings, y=labels)
# Precision: 0.932 | Recall: 0.943 | F1: 0.938 [Animacy seed = <Mensch>]

# Train and evaluate KNNeighbors model
knn_model = AnimacyClassifier(architecture="knn", X=all_embeddings, y=labels)
# Precision: 0.917 | Recall: 0.915 | F1: 0.916 [Animacy seed = <Mensch>]


# Save models to file
rf_model.save(os.path.join(PRETRAINED_ANIMACY_MODEL_DIR, "rf_animacy_classifier.pkl"))
mlp_model.save(os.path.join(PRETRAINED_ANIMACY_MODEL_DIR, "mlp_animacy_classifier.pkl"))
knn_model.save(os.path.join(PRETRAINED_ANIMACY_MODEL_DIR, "knn_animacy_classifier.pkl"))
