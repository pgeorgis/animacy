"""Locating trained animacy classifier files."""
import os
from pathlib import Path

from .constants import DEFAULT_ARCHITECTURE, MODEL_DIR_ENV_VAR

PACKAGED_MODEL_DIR = Path(__file__).resolve().parent / "models"


def classifier_filename(architecture: str) -> str:
    return f"{architecture}_animacy_classifier.pkl"


def find_classifier(architecture: str = DEFAULT_ARCHITECTURE,
                    model_dir: str | os.PathLike = None,
                    ) -> Path:
    """Return the path of the trained classifier file for the given architecture.

    Search order: `model_dir` (if given), the directory in the environment variable
    ANIMACY_MODEL_DIR (if set), then the classifiers packaged with this library.
    """
    filename = classifier_filename(architecture)
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir))
    if os.environ.get(MODEL_DIR_ENV_VAR):
        candidates.append(Path(os.environ[MODEL_DIR_ENV_VAR]).expanduser())
    candidates.append(PACKAGED_MODEL_DIR)
    for directory in candidates:
        path = directory / filename
        if path.is_file():
            return path
    searched = "\n  ".join(str(d / filename) for d in candidates)
    raise FileNotFoundError(f"No trained '{architecture}' animacy classifier found. Searched:\n  {searched}")
