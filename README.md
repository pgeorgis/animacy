# Animacy

Binary animacy classification (animate `AN` vs. inanimate `INAN`) of nouns from word embeddings. Currently implemented for German but could be adapted for other languages. Originally developed for the [D-pronoun corpus annotation project](https://github.com/SFB1252/c07-dpron-korpus), where it annotates the animacy of pronoun coreferents and clausemates.

```python
from animacy import AnimacyClassifier

classifier = AnimacyClassifier.from_pretrained()  # packaged MLP classifier + fastText embeddings
classifier.classify("Lehrerin")  # 'AN'
classifier.classify("Tisch")     # 'INAN'
```

or from the command line:
```bash
python -m animacy Hund Tisch Lehrerin
```

## Table of Contents
* [Method](#method)
* [Installation](#installation)
* [Usage](#usage)
    * [Locating trained classifiers](#locating-trained-classifiers)
* [Training](#training)
    * [GermaNet](#germanet)
* [Repository structure](#repository-structure)

## Method
The methodology largely follows that of [Klenner & Göhring (2022)](https://aclanthology.org/2022.lrec-1.145/) and [Tepei & Bloem (2024)](https://aclanthology.org/2024.lrec-main.163/) for automatic animacy classification in German and Romanian, respectively.

Each class is defined by a seed noun from which all lemmas in synonym and hyponym sets are extracted recursively using [`GermaNet`](#germanet). Following Klenner & Göhring, the seed words used here are `Mensch` for the animate and `Artefakt` for the inanimate class. Lemmas that occur in both classes are removed. This step results in sets of approximately 27000 animate and 31000 inanimate lemmas, which are then embedded with the [fastText German](https://huggingface.co/facebook/fasttext-de-vectors) model. These word embeddings are taken as the training and test data (75/25 split) for a binary classifier.

Classification performance (measured by F1 score) was compared with both `fastText` and a [German BERT](https://huggingface.co/dbmdz/bert-base-german-cased) embedding model as well as three different classifier model architectures on the test data:

| Model Architecture        | `fastText` | `bert-base-german-cased` |
|---------------------------|-----------:|-------------------------------:|
| Random Forest (`rf`)      | 0.952      | 0.922                          |
| Multi-Layer Perceptron (`mlp`) | **0.976**  | 0.953                     |
| K-Nearest Neighbors (`knn`) | 0.959    | 0.927                          |

`fastText` embeddings produced better classification results in all three model architectures. The overall best model was obtained using `fastText` embeddings and a multi-layer perceptron architecture, achieving an F1 score of 0.976. This model is the default, and it is the only classifier packaged with this library (`src/animacy/models/mlp_animacy_classifier.pkl`).

## Installation
This library requires Python 3.11 and is best managed with [uv](https://docs.astral.sh/uv/).

As a development environment:
```bash
uv sync                  # inference only
uv sync --extra train    # also the dependencies for training new classifiers
```

Optional extras: `bert` (torch and transformers, for transformer-based instead of fastText embeddings) and `train` (see [Training](#training)).

## Usage
```python
from animacy import AnimacyClassifier

classifier = AnimacyClassifier.from_pretrained(
    architecture="mlp",                  # 'rf', 'mlp', or 'knn'
    model_path=None,                     # path to a classifier .pkl file, otherwise see below
    encoder="facebook/fasttext-de-vectors",
)
classifier.classify("Lehrerin")  # 'AN'
classifier.classify("Tisch")  # 'INAN'
```

## Training
Install the training dependencies (`uv sync --extra train`), obtain the [GermaNet](#germanet) data, and run from the repository root:
```bash
uv run python -m animacy.train --config configs/animacy_config.yml
```
The config file specifies the GermaNet location, the seed words, the embeddings model, and the classifier architecture. Paths are relative to the working directory:
* embeddings of the extracted lemma sets are cached under `embeddings/<embeddings model>/<class>/<seed>/word_embeddings.npy` and re-used in later runs for the same model and seed words;
* the trained classifier is saved as `pretrained/<model_name>.pkl`.

#### GermaNet
Semantic sets including synonyms, hyponyms, and hypernyms are drawn from [`GermaNet`](https://uni-tuebingen.de/en/faculties/faculty-of-humanities/departments/modern-languages/department-of-linguistics/chairs/general-and-computational-linguistics/ressources/lexica/germanet/), a lexical-semantic resource for German similar to [`WordNet`](https://wordnet.princeton.edu/) developed at the University of Tübingen. 

GermaNet is free for academic users but requires a signed license agreement and therefore the dataset is NOT included as part of this repository. To acquire the data, please follow [instructions at the project page](https://uni-tuebingen.de/en/faculties/faculty-of-humanities/departments/modern-languages/department-of-linguistics/chairs/general-and-computational-linguistics/ressources/lexica/germanet/licenses/). Place the unpacked data at `GermaNet/GN_V200/GN_V200_XML` (or set the path in the config file).

GermaNet is only needed to train a German animacy classifier, not to use a pretrained model.

## Repository structure
* [`src/animacy/`](./src/animacy/)
    * [`classifier.py`](./src/animacy/classifier.py): `AnimacyClassifier`, wrapping the scikit-learn model and the embeddings encoder.
    * [`embeddings.py`](./src/animacy/embeddings.py): fastText and transformer-based word embeddings models.
    * [`train.py`](./src/animacy/train.py): extracts lemma sets from GermaNet, embeds them, and trains and saves a classifier.
    * [`paths.py`](./src/animacy/paths.py), [`constants.py`](./src/animacy/constants.py): classifier file lookup, class labels, and defaults.
    * [`models/`](./src/animacy/models/): the packaged default pretrained classifier for German.
* [`configs/animacy_config.yml`](./configs/animacy_config.yml): example training configuration.
