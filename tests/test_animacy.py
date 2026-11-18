import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from animacy import (ANIMATE_TAG, DEFAULT_EMBEDDINGS_MODEL, INANIMATE_TAG,
                     AnimacyClassifier, find_classifier)
from animacy.constants import MODEL_DIR_ENV_VAR
from animacy.paths import PACKAGED_MODEL_DIR


class StubEncoder:
    """Stands in for a fastText encoder (300-dimensional vectors)."""
    def get_word_embedding(self, word: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(word)) % (2**32))
        return rng.normal(size=300)


def test_packaged_classifier_is_found(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    assert find_classifier() == PACKAGED_MODEL_DIR / "mlp_animacy_classifier.pkl"


def test_env_var_overrides_packaged(monkeypatch, tmp_path):
    custom = tmp_path / "mlp_animacy_classifier.pkl"
    custom.write_bytes(b"")
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    assert find_classifier("mlp") == custom


def test_missing_classifier_raises(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        find_classifier("knn")  # only the MLP is packaged


def test_classify_with_stub_encoder():
    classifier = AnimacyClassifier.from_pretrained(encoder=StubEncoder())
    assert classifier.classify("Wort") in {ANIMATE_TAG, INANIMATE_TAG}


def test_classify_without_encoder_raises():
    classifier = AnimacyClassifier.from_pretrained(encoder=None)
    with pytest.raises(ValueError):
        classifier.classify("Wort")


@pytest.mark.skipif(
    try_to_load_from_cache(DEFAULT_EMBEDDINGS_MODEL, "model.bin") is None,
    reason="fastText model not in the local Hugging Face cache",
)
@pytest.mark.fasttext
def test_classify_with_fasttext():
    classifier = AnimacyClassifier.from_pretrained()
    assert classifier.classify("Lehrerin") == ANIMATE_TAG
    assert classifier.classify("Hund") == ANIMATE_TAG
    assert classifier.classify("Tisch") == INANIMATE_TAG
    assert classifier.classify("Auto") == INANIMATE_TAG


def test_save_creates_parent_directories(tmp_path):
    classifier = AnimacyClassifier.from_pretrained(encoder=None)
    outfile = tmp_path / "nested" / "dir" / "custom_name.pkl"
    classifier.save(outfile)
    reloaded = AnimacyClassifier(architecture="mlp")
    reloaded.load(outfile)
    X = np.random.default_rng(0).normal(size=(5, 300))
    assert list(reloaded.model.predict(X)) == list(classifier.model.predict(X))


def test_training_output_path_from_config_and_cli(tmp_path, monkeypatch):
    pytest.importorskip("germanetpy")
    from animacy import train

    calls = []
    monkeypatch.setattr(train, "train_animacy_classifier", lambda **kw: calls.append(kw))
    config = tmp_path / "config.yml"

    config.write_text("classifier:\n  architecture: rf\n  outfile: ./out/custom.pkl\n")
    train.main(str(config))
    assert calls[-1]["classifier_outfile"] == "./out/custom.pkl"
    assert calls[-1]["model_architecture"] == "rf"

    train.main(str(config), output="cli/override.pkl")  # command line overrides config
    assert calls[-1]["classifier_outfile"] == "cli/override.pkl"

    config.write_text("classifier:\n  architecture: knn\n")  # unset -> default chosen by train_animacy_classifier
    train.main(str(config))
    assert calls[-1]["classifier_outfile"] is None
