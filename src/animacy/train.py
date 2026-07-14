# Adapted from similar work by Tepei & Bloem (2024) using WordNet annotations for animacy classification in Romanian: https://github.com/mariatepei/RO-animacy/blob/main/animacy-code.ipynb

import argparse
import logging
import os

import numpy as np
from germanetpy.germanet import Germanet

from constants.models import (ANIMACY_MODEL_DIR, DEFAULT_EMBEDDINGS_MODEL,
                              GERMANET_DATA, PRETRAINED_ANIMACY_MODEL_DIR)
from constants.tags import ANIMATE_TAG, INANIMATE_TAG
from models.embeddings import EmbeddingsModel, init_embeddings_model
from utils.utils import load_config

from .animacy_classifier import AnimacyClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Path to saved embeddings of animate and inanimate nouns
ANIMACY_EMBEDDINGS = os.path.join(ANIMACY_MODEL_DIR, "embeddings")
DEFAULT_ANIMACY_CLASSIFIER_ARCHITECTURE = "mlp"
# Default seed words for animate and inanimate classes
DEFAULT_ANIMATE_SEED_WORD = "Mensch"  #  Alternative: <Lebewesen>, will include more non-human animals and plants as animate vs. only humans
DEFAULT_INANIMATE_SEED_WORD = "Artefakt"


def get_all_hyponym_lemmas(germanet_data: Germanet, seed_word: str) -> set:
    lemmas = set()
    for synset in germanet_data.get_synsets_by_orthform(seed_word):
        hyponyms = synset.all_hyponyms()
        for item in hyponyms:
            for lexunit in item.lexunits:
                lemmas.update(lexunit.get_all_orthforms())
    return lemmas


def get_animacy_lemmas(germanet_data: Germanet,
                       animate_seed_word: str = DEFAULT_ANIMATE_SEED_WORD,
                       inanimate_seed_word: str = DEFAULT_INANIMATE_SEED_WORD,
                       ) -> tuple[set, set]:
    """Get sets of animate and inanimate lemmas from seed words."""
    animates = get_all_hyponym_lemmas(germanet_data, seed_word=animate_seed_word)
    inanimates = get_all_hyponym_lemmas(germanet_data, seed_word=inanimate_seed_word)
    return animates, inanimates


def train_animacy_classifier(
        germanet_data_path: str = GERMANET_DATA,
        animate_seed_word: str = DEFAULT_ANIMATE_SEED_WORD,
        inanimate_seed_word: str = DEFAULT_INANIMATE_SEED_WORD,
        embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL,
        embeddings_outdir: str = ANIMACY_EMBEDDINGS,
        model_architecture: str = DEFAULT_ANIMACY_CLASSIFIER_ARCHITECTURE,
        classifier_outdir: str = PRETRAINED_ANIMACY_MODEL_DIR,
        ):
    # Load GermaNet data
    germanet_data = Germanet(os.path.abspath(germanet_data_path))

    # Extract all animate and inanimate German lemmas recursively from hyponyms of seed words
    animates, inanimates = get_animacy_lemmas(
        germanet_data=germanet_data,
        animate_seed_word=animate_seed_word,
        inanimate_seed_word=inanimate_seed_word,
    )
    # Remove lemmas that appear in both categories
    overlap = animates & inanimates
    animates = animates - overlap
    inanimates = inanimates - overlap

    logger.info(f"Animate seed word: <{animate_seed_word}> ({len(animates)} lemmas)")
    logger.info(f"Inanimate seed word: <{inanimate_seed_word}> ({len(inanimates)} lemmas)")

    # Load embedding model
    embeddings_model_name = embeddings_model
    embeddings_model: EmbeddingsModel = init_embeddings_model(embeddings_model_name)

    # Check whether embeddings were already generated for the specified embedding model and seed words
    # If so, load the preloaded embeddings; else generate from scratch
    embeddings_outdir = os.path.abspath(embeddings_outdir)
    animate_embeddings_file = os.path.join(embeddings_outdir, embeddings_model_name, "animate", animate_seed_word.lower(), "word_embeddings.npy")
    inanimate_embeddings_file = os.path.join(embeddings_outdir, embeddings_model_name, "inanimate", inanimate_seed_word.lower(), "word_embeddings.npy")
    if os.path.exists(animate_embeddings_file):
        animate_embeddings = np.load(animate_embeddings_file)
    else:
        os.makedirs(os.path.dirname(animate_embeddings_file), exist_ok=True)
        animate_embeddings = embeddings_model.batch_save_embeddings(animates, animate_embeddings_file)
    if os.path.exists(inanimate_embeddings_file):
        inanimate_embeddings = np.load(inanimate_embeddings_file)
    else:
        os.makedirs(os.path.dirname(inanimate_embeddings_file), exist_ok=True)
        inanimate_embeddings = embeddings_model.batch_save_embeddings(inanimates, inanimate_embeddings_file)

    # Get full train/test set of embeddings and corresponding labels
    all_embeddings = np.vstack([animate_embeddings, inanimate_embeddings])
    labels = np.concatenate([
        np.full(len(animates), ANIMATE_TAG),
        np.full(len(inanimates), INANIMATE_TAG)
    ])

    # Train and evaluate classifier model
    model = AnimacyClassifier(architecture=model_architecture, X=all_embeddings, y=labels)

    # Save model to file
    classifier_outdir = os.path.abspath(classifier_outdir)
    model.save(os.path.join(classifier_outdir, f"{model_architecture}_animacy_classifier.pkl"))


def main(config):
    config = load_config(config)
    train_animacy_classifier(
        germanet_data_path=config.get("data", {}).get("GermaNet", GERMANET_DATA),
        animate_seed_word=config.get("data", {}).get("animate_seed", DEFAULT_ANIMATE_SEED_WORD),
        inanimate_seed_word=config.get("data", {}).get("inanimate_seed", DEFAULT_INANIMATE_SEED_WORD),
        embeddings_model=config.get("embeddings", {}).get("model", DEFAULT_EMBEDDINGS_MODEL),
        embeddings_outdir=config.get("embeddings", {}).get("outdir", ANIMACY_EMBEDDINGS),
        model_architecture=config.get("classifier", {}).get("architecture", DEFAULT_ANIMACY_CLASSIFIER_ARCHITECTURE),
        classifier_outdir=config.get("classifier", {}).get("outdir", PRETRAINED_ANIMACY_MODEL_DIR),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trains and saves a binary animacy classifier.")
    parser.add_argument('--config', default=None, help='Path to config.yml file')
    args = parser.parse_args()
    main(args.config)
