import logging
from typing import Generator, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
        logger.info(f"Saved embeddings to {filename}")
        return embeddings
