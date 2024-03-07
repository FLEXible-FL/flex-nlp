"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch


def default_data_collator_classification(batch):
    """Default data collator for classification whether
    it expects just a batch that contains labels and text.

    This function does not apply any preprocessing to the text,
    nor the labels, so they must be preprocessed before via the
    torch.data.Dataset. This means, that the functions should recieve
    the text already tokenized and converted to ids, and the labels
    should be already transformed to the interval 0,..,n, in case
    of a classification problem, or as the desired expected type.

    Args:
        batch (Batch): Batch with the elements to process.
    Returns:
        tuple: Tuple the text and the labels of the batch.
    """
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        text_list.append(torch.tensor(_text)) if not isinstance(_text, torch.tensor) else text_list.append(_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list

def basic_collate_pad_sequence_classification(batch):
    """Basic collate function that convert the batches into torch
    tensors and returns the text padded.

    This function does not apply any preprocessing to the text,
    nor the labels, so they must be preprocessed before via the
    torch.data.Dataset. This means, that the functions should recieve
    the text already tokenized and converted to ids, and the labels
    should be already transformed to the interval 0,..,n, in case
    of a classification problem, or as the desired expected type.

    Args:
        batch (Batch): Batch with the elements to process.
    Returns:
        tuple: Tuple containing the text padded with the pad_sequence
        function and the labels for the batch.
    """
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text) if torch.is_tensor(_text) else text_list.append(torch.tensor(_text))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True), label_list
