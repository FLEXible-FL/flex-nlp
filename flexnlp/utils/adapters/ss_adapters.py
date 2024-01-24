def ss_triplet_input_adapter(X_train_as_list: list = None, X_test_as_list: list = None,
                    batch_size=16, shuffle=True, train=True, test=True):
    """Function that adapt the input from a Triplet Dataset to use within a
    SentenceTransformer's model. 

    The method ensures that the data is provived in order to give an output.
    Args:
        X_train_as_list (list, optional): _description_. Defaults to None.
        X_test_as_list (list, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to 16.
        shuffle (bool, optional): _description_. Defaults to True.
        train (bool, optional): _description_. Defaults to True.
        test (bool, optional): _description_. Defaults to True.
    Returns:
        tuple: 
    """
    if not X_train_as_list and not X_test_as_list:
        raise ValueError("No data given. Please provide data for train or test.")
    if not train and not test:
        raise ValueError("train or test parameters must be true in order to give an output.")

    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader

    train_examples = None
    dev_examples = None
    if train and len(X_train_as_list) > 1:
        train_examples = [InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]) for example in X_data_as_list]
        train_examples = DataLoader(train_examples, shuffle=shuffle, batch_size=batch_size)
    if test and len(X_test_as_list) > 1:
        dev_examples = [InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]) for example in X_test_as_list]

    return train_examples, dev_examples        