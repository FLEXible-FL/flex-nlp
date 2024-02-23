# flex-nlp

The flex-nlp package consists of a set of tools and utilities to work with Natural Language Processing (NLP) datasets and models. It is designed to be used with the [FLEXible](https://github.com/FLEXible-FL/FLEXible/) framework, as it is an extension of it.

flex-nlp comes with some tools to work with NLP datasets, that are the following ones:

- `ss_triplet_input_adapter` a Semantic Textual Similarity (STS) dataset adapter: It is a dataset adapter that allows to work with the TripletQQP dataset and other datasets that are similar to it.
- `default_data_collator_classification`: It is a data collator that allows to work with the classification task, and it is the default data collator for the classification task.
- `basic_collate_pad_sequence_classification`: It is a data collator that allows to work with the classification task, and it is a basic data collator for the classification task. This collator pads the sequences to the maximum length of the batch, and it puts the batch dimension in the first position.

We also provide an aggregator to work with neural networks, clip_avg. Alonside, we have used some aggregator available in the [FLEXible](https://github.com/FLEXible-FL/FLEXible/) framework.


| `Aggregator`            | `Description`      | `Citation`              |
| ------------------ | :------------------: | -------------------: |
| clip_avg | It is a federated aggregator that clips the weights recieved by the clients, averaging only those that surpass a selected threshold. | [Reviewing Federated Learning Aggregation Algorithms; Strategies, Contributions, Limitations and Future Perspectives](https://www.mdpi.com/2079-9292/12/10/2287) |
| fedavg | It is a federated aggregator that compute the mean of the weights recieved by the clients. | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |
| weighted_avg | Similar to fedavg, it is a federated aggregator that add weights to the clients in order of giving more importance to some clients than to another clients. | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |

##  Tutorials

To get started with flex-nlp, you can check the [notebooks](https://github.com/FLEXible-FL/flex-nlp/tree/main/notebooks) available in the repository. They cover the following topics:

- [Sentiment Analysis using the IMDB dataset with a BiGRU model](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20IMDb%20PT%20using%20FLExible%20with%20a%20GRU.ipynb).
- [Question Answering using the SQuAD dataset with DistilBERT model](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20QA%20with%20Hugginface%20using%20FLEXIBLE.ipynb).
- [Semantic Textual Similarity using the QQP-Triplets dataset with a distilled version of Roberta](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20SS%20with%20SentenceTransformers%20using%20FLEXible.ipynb).

In the following we detail the tasks, models, and the datasets used in the notebooks:

| `Task`            | `Model`      | `Dataset`              |
| ------------------ | :------------------: | -------------------: |
| Sentiment Analysis (SA) | [BiGRU](https://arxiv.org/abs/1412.3555) | [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/) |
| Question Answering (QA) | [DistilBERT](https://arxiv.org/abs/1910.01108) | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) |
| Semantic Textual Similarity (STS) | [DistilRoberta](https://arxiv.org/abs/1907.11692) | [QQP-Triplets](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) |

## Installation

We recommend Anaconda/Miniconda as the package manager. The following is the corresponding `flex-nlp` versions and supported Python versions.

| `flex`            | `flex-nlp`      | Python              |
| :------------------: | :------------------: | :-------------------: |
| `main` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11`   |
| `v0.6.0`           | `v0.1.0`           | `>=3.8`, `<=3.11`    |

To install the package, you can use the following commands:

Using pip:
```
pip install flex-nlp
```

Download the repository and install it locally:
```
git clone git@github.com:FLEXible-FL/flex-nlp.git
cd flex-nlp
pip install -e .
```

## Citation

If you use this package, please cite the following paper:

``` TODO: Add citation ```