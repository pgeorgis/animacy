# Adapted from similar work by Tepei & Bloem (2024) using WordNet annotations for animacy classification in Romanian: https://github.com/mariatepei/RO-animacy/blob/main/animacy-code.ipynb

import argparse
import logging
import os

import numpy as np
from germanetpy.germanet import Germanet

from .classifier import AnimacyClassifier
from .constants import (ANIMATE_TAG, DEFAULT_ANIMATE_SEED_WORD,
                        DEFAULT_ARCHITECTURE, DEFAULT_CLASSIFIER_OUTDIR,
                        DEFAULT_EMBEDDINGS_MODEL, DEFAULT_EMBEDDINGS_OUTDIR,
                        DEFAULT_GERMANET_DIR, DEFAULT_INANIMATE_SEED_WORD,
                        INANIMATE_TAG)
from .embeddings import EmbeddingsModel, init_embeddings_model
from .paths import classifier_filename

logger = logging.getLogger(__name__)

# Locations (relative to the working directory) of GermaNet data, saved embeddings of animate
# and inanimate nouns, and the trained classifiers
GERMANET_DATA = DEFAULT_GERMANET_DIR
ANIMACY_EMBEDDINGS = DEFAULT_EMBEDDINGS_OUTDIR
PRETRAINED_ANIMACY_MODEL_DIR = DEFAULT_CLASSIFIER_OUTDIR
DEFAULT_ANIMACY_CLASSIFIER_ARCHITECTURE = DEFAULT_ARCHITECTURE


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


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
        classifier_outfile: str = None,
        ):
    """Train an animacy classifier from GermaNet semantic sets and save it.

    classifier_outfile: path (including file name) where the trained classifier is saved.
        Defaults to <PRETRAINED_ANIMACY_MODEL_DIR>/<architecture>_animacy_classifier.pkl.
    """
    if classifier_outfile is None:
        classifier_outfile = os.path.join(PRETRAINED_ANIMACY_MODEL_DIR, classifier_filename(model_architecture))
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
    model.save(classifier_outfile)


def main(config=None, output=None):
    config = load_config(config) if config else {}
    train_animacy_classifier(
        germanet_data_path=config.get("data", {}).get("GermaNet", GERMANET_DATA),
        animate_seed_word=config.get("data", {}).get("animate_seed", DEFAULT_ANIMATE_SEED_WORD),
        inanimate_seed_word=config.get("data", {}).get("inanimate_seed", DEFAULT_INANIMATE_SEED_WORD),
        embeddings_model=config.get("embeddings", {}).get("model", DEFAULT_EMBEDDINGS_MODEL),
        embeddings_outdir=config.get("embeddings", {}).get("outdir", ANIMACY_EMBEDDINGS),
        model_architecture=config.get("classifier", {}).get("architecture", DEFAULT_ANIMACY_CLASSIFIER_ARCHITECTURE),
        classifier_outfile=output or config.get("classifier", {}).get("outfile"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser("Trains and saves a binary animacy classifier.")
    parser.add_argument('--config', default=None, help='Path to config.yml file')
    parser.add_argument('--output', default=None, help='Path to output trained classifier model; overrides `classifier.outfile` in the config file')
    args = parser.parse_args()
    main(args.config, output=args.output)
