from copy import deepcopy
import numpy as np

from datasets.load import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, FastText, vocab


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = "cpu"

# imdb_dataset = load_dataset('imdb', split=['train', 'test']) # Get the dataset from huggingface library
train_imdb_dataset, test_imdb_dataset = torchtext.datasets.IMDB() # Get the dataset from torchtext library

embeddings_dim = 100 # Dimension of the embeddings
glove = GloVe(name='6B', dim=embeddings_dim) # Load GloVe embeddings with 100 dimensions.
# fasttext = FastText(language='en') # To use FastText instead of GloVe
vocabulary = vocab(glove.stoi)
# vocabulary_fasttext = vocab(fasttext.stoi) # To use FastText instead of GloVe
vocab_size = len(vocabulary) # Get the vocabulary size
print(f"Total vocabulary size: {vocab_size}")
print(f"Shape of embeddings: {glove.vectors.shape}")

example = "This is an example sentence to test the tokenizer."
tokenizer = get_tokenizer("basic_english")
spacy_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
example_tokens = tokenizer(example)
example_tokens_spacy = spacy_tokenizer(example)

print(f"Padding token idx, pad: {vocabulary.get_stoi()['pad']}") # Get the index of the word 'pad' for padding
print(f"Unknown token idx, unk: {vocabulary.get_stoi()['unk']}") # Get the index of the word 'unk' for unknown words

pad_token = "<pad>"
pad_index = 1
unk_token = "<unk>"
unk_index = 0
vocabulary.insert_token(pad_token, 0)
vocabulary.insert_token(unk_token, unk_index)
vocabulary.set_default_index(unk_index)
# glove.vectors = torch.cat(1, (torch.zeros(1, embeddings_dim), glove.vectors))
pretrained_embeddings = glove.vectors
print(f"Len pretrained embeddings: {len(pretrained_embeddings)}")
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
print(f"Len pretrained embeddings: {len(pretrained_embeddings)}")
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
print(f"Len pretrained embeddings: {len(pretrained_embeddings)}")

print(f"Basic English Tokenizer: {example_tokens}")
print(f"Spacy Tokenizer: {example_tokens_spacy}")

# Remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

print(f"Example tokens tokenized: {[word.lower() for word in example_tokens_spacy]}")

print(f"Example tokens without stopwords: {[word.lower() for word in example_tokens_spacy if word not in stop_words]}")

print(f"Example tokens without stopwords and word in vocabulary: {[word.lower() for word in example_tokens_spacy if word not in stop_words and word.lower() in vocabulary]}")

print(f"Example tokens without quitting stopwords and word in vocabulary: {[word.lower() for word in example_tokens_spacy if word.lower() in vocabulary]}")

from flex.data import FedDatasetConfig, FedDataDistribution

config = FedDatasetConfig(seed=0)
config.n_clients = 2
config.replacement = False # ensure that clients do not share any data
config.client_names = ['client1', 'client2'] # Optional
flex_dataset = FedDataDistribution.from_config_with_torchtext_dataset(data=train_imdb_dataset, config=config)

from flex.data import Dataset

test_dataset = Dataset.from_torchtext_dataset(test_imdb_dataset)

from flex.model import FlexModel
from flex.pool import FlexPool

from flex.pool.decorators import init_server_model
from flex.pool.decorators import deploy_server_model


class LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_classes, emb_vectors) -> None:
        super().__init__()
        # self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dim)
        # self.emb.weight.data.copy_(emb_vectors) # Initialize the embedding layer with the pretrained embeddings
        # self.emb.requires_grad_(False) # Freeze the embedding layer
        # We can do the 3 steps above with the following line
        self.emb = nn.Embedding.from_pretrained(embeddings=emb_vectors, freeze=True)
        embeddings_dim = emb_vectors.shape[1]
        # self.lstm = nn.LSTM(input_size=embeddings_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.gru = nn.GRU(input_size=embeddings_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        # print(f"Showing the shape of the input: {x.shape}")
        x = self.emb(x)
        # print(f"Showing the shape of the input after the embedding layer: {x.shape}")
        _, x = self.gru(x)
        # x = F.relu(x[:, -1, :])
        # print(f"Showing the shape of the input after the LSTM layer: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"Showing the shape of the input after the first linear layer: {x.shape}")
        x = self.fc2(x)
        # print(f"Showing the shape of the output shape: {x.shape}")
        return x


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model['model'] = LSTMNet(hidden_size=128, num_classes=2, emb_vectors=pretrained_embeddings)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.AdamW
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model


flex_pool = FlexPool.client_server_architecture(fed_dataset=flex_dataset, init_func=build_server_model)

clients = flex_pool.clients
servers = flex_pool.servers
aggregators = flex_pool.aggregators

print(f"Number of nodes in the pool {len(flex_pool)}: {len(servers)} server plus {len(clients)} clients. The server is also an aggregator")

from flex.pool import deploy_server_model, deploy_server_model_pt

@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    return deepcopy(server_flex_model)

servers.map(copy_server_model_to_clients, clients) # Using the function created with the decorator
# servers.map(deploy_server_model_pt, clients) # Using the primitive function


import re
import random

from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

BATCH_SIZE = 64
NUM_EPOCHS = 10

def clean_str(string):
    """
    Tokenization/string cleaning.
    Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def preprocess_text(text):
    text_transform = lambda x: [pad_index] + [vocabulary[token] for token in spacy_tokenizer(x)] + [pad_index]
    return list(text_transform(clean_str(text)))

def preprocess_text_old(text):
    return list(spacy_tokenizer(clean_str(text)))

def convert_token_to_idx(text_tokenized):
    return [vocabulary[token] for token in text_tokenized]

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_transform = lambda x: 0 if x == 2 else x
        label_list.append(label_transform(_label))
        # preprocessed_text = preprocess_text(_text)
        # prepro_old = convert_token_to_idx(preprocess_text_old(_text))
        # print(f"Preprocessed text: {preprocessed_text}")
        # print(f"Preprocessed text old: {prepro_old}")
        # print(type(preprocessed_text))
        # processed_text = torch.tensor(preprocess_text(_text))
        # processed_text = torch.tensor(convert_token_to_idx(preprocess_text_old(_text)))
        # text_list.append(processed_text)
        text_list.append(torch.tensor(preprocess_text(_text)))
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=pad_index, batch_first=True)

def batch_sampler_v2(batch_size, indices):
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths 
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

def train(client_flex_model: FlexModel, client_data: Dataset):
    X_data, y_data = client_data.to_list()
    if 'train_indices' not in client_flex_model:
        train_indices = [(i, len(tokenizer(s[0]))) for i, s in enumerate(X_data)]
        client_flex_model['train_indices'] = train_indices
    else:
        train_indices = client_flex_model['train_indices']
    # batch_size=BATCH_SIZE, shuffle=True, # No es necesario usarlo porque usamos el batch_sampler
    # client_dataloader = DataLoader(client_data, collate_fn=collate_batch,
    #                             batch_sampler=batch_sampler_v2(BATCH_SIZE, train_indices))
    model = client_flex_model["model"]
    # lr = 0.001
    optimizer = client_flex_model['optimizer_func'](model.parameters(), **client_flex_model["optimizer_kwargs"])
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    # Al usar batch_sampler, hay que recargar el DataLoader en cada epoch.
    for _ in tqdm(range(NUM_EPOCHS)):
        client_dataloader = DataLoader(client_data, collate_fn=collate_batch,
                                    batch_sampler=batch_sampler_v2(BATCH_SIZE, train_indices))
        for labels, texts in client_dataloader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(texts)
            pred = pred.squeeze(dim=0)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()


clients.map(train)


from flex.pool import collect_clients_weights_pt

aggregators.map(collect_clients_weights_pt, clients)

from flex.pool import fed_avg

aggregators.map(fed_avg)

from flex.pool import set_aggregated_weights_pt

aggregators.map(set_aggregated_weights_pt, servers)

from flex.pool import evaluate_server_model

@evaluate_server_model
def evaluate_global_model(server_flex_model: FlexModel, test_data=None):
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion=server_flex_model['criterion']
    # get test data as a torchvision object
    # test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=False, collate_fn=collate_batch)
    X_data, _ = test_dataset.to_list()
    test_indices = [(i, len(tokenizer(s[0]))) for i, s in enumerate(X_data)]
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_batch,
                                    batch_sampler=batch_sampler_v2(BATCH_SIZE, test_indices))
    losses = []
    with torch.no_grad():
        for target, data in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.squeeze(dim=0)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

metrics = servers.map(evaluate_global_model, test_data=test_dataset)

print(f"Metrics: {metrics}")

