# import from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers

import collections
from regex import T

import torch
from torch.utils.data.sampler import Sampler
import numpy as np


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices


def get_labels_to_ngram_ids(labels, ngram_ids):
    labels_to_ngram_ids = collections.defaultdict(list)

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(ngram_ids):
        ngram_ids = ngram_ids.cpu().numpy()

    for i, label in enumerate(labels):
        labels_to_ngram_ids[label].append(ngram_ids[i])
        labels_to_ngram_ids[label] = list(set(labels_to_ngram_ids[label]))
    for k, v in labels_to_ngram_ids.items():
        labels_to_ngram_ids[k] = np.array(v, dtype=int)
    return labels_to_ngram_ids


def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)


# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class

        idx_list = [idx.item() for idx in idx_list]
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1


class MPerClassSamplerWithoutEasyPostives(Sampler):
    def __init__(self, labels, ngram_ids, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if isinstance(ngram_ids, torch.Tensor):
            ngram_ids = ngram_ids.numpy()

        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size

        self.labels_to_ngram_ids = get_labels_to_ngram_ids(labels, ngram_ids)
        self.ngram_ids_to_indices = get_labels_to_indices(ngram_ids)

        self.labels = list(self.labels_to_ngram_ids.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_ngram_ids[label]

                random_ngram_ids = safe_random_choice(
                    t, size=self.m_per_class
                )
                for j in range(self.m_per_class):
                    idx_list[i+j] = np.random.choice(self.ngram_ids_to_indices[random_ngram_ids[j]])
                i += self.m_per_class

        idx_list = [idx.item() for idx in idx_list]
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1