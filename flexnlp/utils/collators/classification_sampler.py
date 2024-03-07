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
import random
from torch.utils.data.sampler import Sampler

class ClassificationCollator(Sampler):
    """Class the acts as a sampler in a classification problem.
    This class will create the indices of the dataset, and will
    create the batches with instances of similar length, improving
    the performance of the model trained.
    """
    def __init__(self, data_source, tokenizer, text_idx, batch_size) -> None:
        self._ind_n_len = [(i, len(tokenizer(s[text_idx]))) for i, s in enumerate(data_source)]
        self._batch_size = batch_size

    def __iter__(self):
        random.shuffle(self._ind_n_len)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(self._ind_n_len), self._batch_size * 100):
            pooled_indices.extend(sorted(self._ind_n_len[i:i + self._batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self._batch_size):
            yield pooled_indices[i:i + self._batch_size]
