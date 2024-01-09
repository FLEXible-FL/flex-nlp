from copy import deepcopy
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import collections
import numpy as np
import evaluate

# FLEXible imports
from flex.data import FedDatasetConfig, FedDataDistribution
from flex.data import Dataset
from flex.model import FlexModel
from flex.pool import FlexPool
from flex.pool.decorators import init_server_model
from flex.pool.decorators import deploy_server_model
from flex.pool import deploy_server_model, deploy_server_model_pt
from torch.utils.data import Dataset as TorchDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

# Load the dataset
# Load a percentage of squal
squad = load_dataset("squad", split="train[:1%]")
# Split 80% train, 20% test
squad  = squad.train_test_split(test_size=0.2)
print(squad)

# Preprocess functions

model_checkpoint = "distilbert-base-uncased"
#model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

train_dataset = squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=squad["train"].column_names,
)

test_dataset = squad["test"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["test"].column_names,
)

# From centralized to federated data

config = FedDatasetConfig(seed=0)
config.n_clients = 2
config.replacement = False # ensure that clients do not share any data
config.client_names = ['client1', 'client2'] # Optional
flex_dataset = FedDataDistribution.from_config_with_huggingface_dataset(data=train_dataset, config=config, 
                                                                        X_columns=['input_ids', 'attention_mask'],
                                                                        label_columns=['start_positions', 'end_positions'])

# Adapt the test dataset to a FlexDataset
test_dataset = Dataset.from_huggingface_dataset(test_dataset,
                                                X_columns=['input_ids', 'attention_mask'])
                                                # label_columns=['start_positions', 'end_positions'])

# Init the server model and deploy it
@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model['model'] = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    # Required to store this for later stages of the FL training process
    server_flex_model['training_args'] = TrainingArguments(
        output_dir="my_awesome_qa_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # server_flex_model['trainer'] = trainer

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

# Create a custom TorchDataset to use it with the Trainer
class QADataset(TorchDataset):
    def __init__(self, encodings, labels) -> None:
        self.encodings = encodings
        self.labels = labels

# Train each client's model
def train(client_flex_model: FlexModel, client_data: Dataset):
    model = client_flex_model['model']
    training_args = client_flex_model['training_args']
    client_train_dataset = client_data.to_numpy()
    print(client_data)
    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=client_train_dataset,
        # eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    # data_collator=data_collator,
    )
    trainer.train()

clients.map(train)
