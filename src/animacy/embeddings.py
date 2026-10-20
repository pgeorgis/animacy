import logging
from typing import Generator, Iterable

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingsModel:
    """Contextual (transformer) word embeddings. Requires the optional `bert` extra (torch, transformers)."""
    def __init__(self, name: str):
        from transformers import AutoModel, AutoTokenizer
        self.name = name
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def get_word_embedding(self, word: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0,1].numpy()

    def get_batch_embedding(self, batch: Iterable[str]) -> Generator:
        return (self.get_word_embedding(word) for word in batch)

    def batch_save_embeddings(self, batch: Iterable[str], filename: str) -> np.ndarray:
        embeddings = np.vstack(list(tqdm(self.get_batch_embedding(batch), total=len(batch))))
        np.save(filename, embeddings)
        logger.info(f"Saved embeddings to {filename}")
        return embeddings


class FastTextEmbeddingsModel(EmbeddingsModel):
    def __init__(self, name: str):
        self.name = name
        self.model = self.load_model()

    def load_model(self):
        import fasttext
        logger.info(f"Loading fastText embedding model <{self.name}>")
        model_path = hf_hub_download(repo_id=self.name, filename="model.bin")
        return fasttext.load_model(model_path)

    def get_word_embedding(self, word: str) -> np.ndarray:
        return self.model[word]


def is_fasttext(model_id: str) -> bool:
    """Returns True or False, according to whether the specified model ID is a fastText model."""
    if "fasttext" in model_id:
        return True
    if model_id.startswith("facebook"):  # e.g. facebook/fasttext-de-vectors
        return True
    return False


def init_embeddings_model(model_id: str) -> EmbeddingsModel | FastTextEmbeddingsModel:
    """Loads and initializes the appropriate embeddings model class by model ID."""
    if is_fasttext(model_id):
        return FastTextEmbeddingsModel(model_id)
    return EmbeddingsModel(model_id)
