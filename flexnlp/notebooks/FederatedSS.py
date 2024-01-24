from copy import deepcopy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models # , util
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset

# FLEXible imports
from flex.data import FedDatasetConfig, FedDataDistribution
from flex.data import Dataset
from flex.model import FlexModel
from flex.pool import FlexPool
from flex.pool.decorators import init_server_model
from flex.pool.decorators import deploy_server_model
from flex.pool import deploy_server_model, deploy_server_model_pt
from torch.utils.data import Dataset as TorchDataset
from flex.pool import collect_clients_weights_pt
from flex.pool import fed_avg
from flex.pool import set_aggregated_weights_pt
from flex.pool import evaluate_server_model
from flexnlp.utils.adapters import ss_triplet_input_adapter

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)


# Load the dataset
dataset_id = "embedding-data/QQP_triplets"
# dataset_id = "embedding-data/sentence-compression"

data = load_dataset(dataset_id, split=['train[:1%]'])[0].train_test_split(test_size=0.1)
dataset, test_dataset = data['train'], data['test']
print(f"- The {dataset_id} dataset has {dataset.num_rows} examples.")
print(f"- Each example is a {type(dataset[0])} with a {type(dataset[0]['set'])} as value.")
print(f"- Examples look like this: {dataset[0]}")


# Load the model
## Step 1: use an existing language model
# word_embedding_model = models.Transformer('distilroberta-base')

## Step 2: use a pool function over the token embeddings
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# From centralized to federated data

config = FedDatasetConfig(seed=0)
config.n_clients = 2
config.replacement = False # ensure that clients do not share any data
config.client_names = ['client1', 'client2'] # Optional
flex_dataset = FedDataDistribution.from_config_with_huggingface_dataset(data=dataset, config=config,
                                                                        X_columns=['set'], # 'title', 'context', 'question'],
                                                                        label_columns=['set']
                                                                        # X_columns=['input_ids', 'attention_mask'],
                                                                        # label_columns=['start_positions', 'end_positions']
                                                                        )

@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    word_embedding_model = models.Transformer('distilroberta-base')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    server_flex_model['model'] = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return server_flex_model

flex_pool = FlexPool.client_server_pool(fed_dataset=flex_dataset, init_func=build_server_model)

clients = flex_pool.clients
servers = flex_pool.servers
aggregators = flex_pool.aggregators

print(f"Number of nodes in the pool {len(flex_pool)}: {len(servers)} server plus {len(clients)} clients. The server is also an aggregator")

# Deploy the model
@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    return deepcopy(server_flex_model)

servers.map(copy_server_model_to_clients, clients) # Using the function created with the decorator
# servers.map(deploy_server_model_pt, clients) # Using the primitive function

# Prepare data for training phase

def create_input_examples_for_training(X_data_as_list, X_test_as_list):
    """Function to create a DataLoader to train/finetune the model at client level

    Args:
        X_data_as_list (list): List containing the examples. Each example is a dict
        with the following keys: query, pos, neg.
    """
    train_examples = [InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]) for example in X_data_as_list]
    dev_examples = [InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]) for example in X_test_as_list]
    return DataLoader(train_examples, shuffle=True, batch_size=16), dev_examples

def train(client_flex_model: FlexModel, client_data: Dataset):
    print("Training client")
    model = client_flex_model['model']
    sentences = ['This is an example sentence', 'Each sentence is converted']
    encodings = model.encode(sentences)
    print(f"Old encodings: {encodings}")
    X_data = client_data.X_data.tolist()
    tam_train = int(len(X_data) * 0.75)
    X_data, X_test = X_data[:tam_train], X_data[tam_train:]
    train_dataloader, dev_examples = ss_triplet_input_adapter(X_train_as_list=X_data, X_test_as_list=X_test)
    train_loss = losses.TripletLoss(model=model)
    evaluator = TripletEvaluator.from_input_examples(dev_examples)
    warmup_steps = int(len(train_dataloader) * 1 * 0.1) #10% of train data
    model.fit(train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=1000,
    )
    # model.evaluate(evaluator, 'model_evaluation')
    sentences = ['This is an example sentence', 'Each sentence is converted']
    encodings = model.encode(sentences)
    print(f"New encodings: {encodings}")

clients.map(train)

aggregators.map(collect_clients_weights_pt, clients)

aggregators.map(fed_avg)

aggregators.map(set_aggregated_weights_pt, servers)

def create_input_examples_for_testing(X_test_as_list):
    """Function to create a DataLoader to train/finetune the model at client level

    Args:
        X_test_as_list (list): List containing the examples. Each example is a dict
        with the following keys: query, pos, neg.
    """
    return [InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]) for example in X_test_as_list]

test_dataset = Dataset.from_huggingface_dataset(test_dataset, X_columns=['set'])

@evaluate_server_model
def evaluate_global_model(server_flex_model: FlexModel, test_data=None):
    _, X_test = ss_triplet_input_adapter(X_test_as_list=test_dataset.X_data.tolist(), train=False)
    model = server_flex_model["model"]
    evaluator = TripletEvaluator.from_input_examples(X_test)
    model.evaluate(evaluator, 'server_evaluation')
    print("Model evaluation saved to file.")

servers.map(evaluate_global_model, test_data=test_dataset)

def train_n_rounds(n_rounds, clients_per_round=2):  
    pool = FlexPool.client_server_pool(fed_dataset=flex_dataset, init_func=build_server_model)
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients
        print(f"Selected clients for this round: {len(selected_clients)}")
        # Deploy the server model to the selected clients
        pool.servers.map(deploy_server_model_pt, selected_clients)
        # Each selected client trains her model
        selected_clients.map(train)
        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_clients_weights_pt, selected_clients)
        pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_weights_pt, pool.servers)
        servers.map(evaluate_global_model, test_data=test_dataset)

# Train the model for n_rounds
# train_n_rounds(5)

# End
