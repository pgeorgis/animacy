"""Constants and defaults for German noun animacy classification."""

# Animacy class labels
ANIMATE_TAG = "AN"
INANIMATE_TAG = "INAN"

# Word embeddings model (Hugging Face Hub ID) used to encode words for classification
DEFAULT_EMBEDDINGS_MODEL = "facebook/fasttext-de-vectors"

# Classifier architectures:
# rf = Random Forest
# mlp = Multi-Layer Perceptron
# knn = K-Nearest Neighbors
ARCHITECTURES = ("rf", "mlp", "knn")
DEFAULT_ARCHITECTURE = "mlp"

# Seed words whose GermaNet synonym/hyponym sets define the animate and inanimate classes
# (alternative animate seed: <Lebewesen>, which would also include non-human animals and plants)
DEFAULT_ANIMATE_SEED_WORD = "Mensch"
DEFAULT_INANIMATE_SEED_WORD = "Artefakt"

# Training defaults (relative to the working directory the training script is run from)
DEFAULT_GERMANET_DIR = "GermaNet/GN_V200/GN_V200_XML"
DEFAULT_EMBEDDINGS_OUTDIR = "embeddings"
DEFAULT_CLASSIFIER_OUTDIR = "pretrained"

# Environment variable that may point to a directory containing trained classifier .pkl files
MODEL_DIR_ENV_VAR = "ANIMACY_MODEL_DIR"
