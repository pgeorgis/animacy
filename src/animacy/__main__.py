"""Command-line interface: classify German words as animate (AN) or inanimate (INAN).

Usage: python -m animacy Hund Tisch Lehrerin
"""
import argparse
import logging

from .classifier import AnimacyClassifier
from .constants import DEFAULT_ARCHITECTURE, DEFAULT_EMBEDDINGS_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(prog="animacy", description="Classify German words as animate (AN) or inanimate (INAN).")
    parser.add_argument("words", nargs="+", help="Word(s) to classify")
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE, choices=["rf", "mlp", "knn"], help="Classifier architecture")
    parser.add_argument("--model", default=None, help="Path to a classifier .pkl file (default: packaged model)")
    parser.add_argument("--encoder", default=DEFAULT_EMBEDDINGS_MODEL, help="Hugging Face ID of the word embeddings model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    classifier = AnimacyClassifier.from_pretrained(args.architecture, model_path=args.model, encoder=args.encoder)
    for word in args.words:
        print(f"{word}\t{classifier.classify(word)}")


if __name__ == "__main__":
    main()
