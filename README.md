# flex-nlp

The flex-nlp package consists of a set of tools and utilities to work with Natural Language Processing (NLP) datasets and models. It is designed to be used with the [FLEXible](https://github.com/FLEXible-FL/FLEXible/) framework, as it is an extension of it.

flex-nlp comes with some tools to work with NLP datasets, such as sampling and collating. It also provides tools to adapt external datasets to the FLEXible framework, like the Triplet Dataset tipically used in Sentence Similarity tasks.

## Tutorials

To get started with flex-nlp, you can check the [notebooks](https://github.com/FLEXible-FL/flex-nlp/tree/main/notebooks) available in the repository. They cover the following topics:

- [Sentiment Analysis using the IMDB dataset with a BiGRU model](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20IMDb%20PT%20using%20FLExible%20with%20a%20GRU.ipynb).
- [Question Answering using the SQuAD dataset with DistilBERT model](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20QA%20with%20Hugginface%20using%20FLEXIBLE.ipynb).
- [Semantic Textual Similarity using the QQP-Triplets dataset with a distilled version of Roberta](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20SS%20with%20SentenceTransformers%20using%20FLEXible.ipynb).

## Installation

We recommend Anaconda/Miniconda as the package manager. The following is the corresponding `flex-nlp` versions and supported Python versions.

| `flex`            | `flex-nlp`      | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11`   |

To install the package, you can use the following commands:

Using pip:
```
pip install flex-nlp
```

Download the repository and install it locally:
```
git clone https://github.com/FLEXible-FL/flex-nlp.git
cd flex-nlp
pip install -e .
```

## Citation

If you use this package, please cite the following paper:

``` TODO: Add citation ```