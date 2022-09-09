# import from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/

import faiss
import numpy as np
import torch


def small_val(dtype):
    return torch.finfo(dtype).tiny


def torch_arange_from_size(input, size_dim=0):
    return torch.arange(input.size(size_dim), device=input.device)


def check_shapes(embeddings, labels):
    if embeddings.size(0) != labels.size(0):
        raise ValueError("Number of embeddings must equal number of labels")
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
        )
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def maybe_get_avg_of_avgs(accuracy_per_sample, sample_labels, avg_of_avgs):
    if avg_of_avgs:
        unique_labels = torch.unique(sample_labels, dim=0)
        mask = torch_all_from_dim_to_end(
            sample_labels == unique_labels.unsqueeze(1), 2
        )
        mask = torch.t(mask)
        acc_sum_per_class = torch.sum(accuracy_per_sample.unsqueeze(1) * mask, dim=0)
        mask_sum_per_class = torch.sum(mask, dim=0)
        average_per_class = acc_sum_per_class / mask_sum_per_class
        return torch.mean(average_per_class).item()
    return torch.mean(accuracy_per_sample).item()


def get_relevance_mask(
    shape,
    gt_labels,
    embeddings_come_from_same_source,
    label_counts,
    label_comparison_fn,
):
    relevance_mask = torch.zeros(size=shape, dtype=torch.bool, device=gt_labels.device)

    for label, count in zip(*label_counts):
        matching_rows = torch.where(
            torch_all_from_dim_to_end(gt_labels == label, 1)
        )[0]
        max_column = count - 1 if embeddings_come_from_same_source else count
        relevance_mask[matching_rows, :max_column] = True
    return relevance_mask


def r_precision(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    label_counts,
    avg_of_avgs,
    label_comparison_fn,
):
    relevance_mask = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
    )
    same_label = label_comparison_fn(gt_labels, knn_labels)
    matches_per_row = torch.sum(same_label * relevance_mask, dim=1)
    max_possible_matches_per_row = torch.sum(relevance_mask, dim=1)
    accuracy_per_sample = (
        to_dtype(matches_per_row, dtype=torch.float64)
        / max_possible_matches_per_row
    )
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)


def get_label_match_counts(query_labels, reference_labels, label_comparison_fn):
    unique_query_labels = torch.unique(query_labels, dim=0)
    comparison = unique_query_labels[:, None] == reference_labels
    match_counts = torch.sum(torch_all_from_dim_to_end(comparison, 2), dim=1)

    return (unique_query_labels, match_counts)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def to_numpy(v):
    if is_list_or_tuple(v):
        return np.stack([to_numpy(sub_v) for sub_v in v], axis=1)
    try:
        return v.cpu().numpy()
    except AttributeError:
        return v


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x


def add_to_index_and_search(index, reference_embeddings, test_embeddings, k):
    index.add(reference_embeddings)
    return index.search(test_embeddings, k)


def try_gpu(cpu_index, reference_embeddings, test_embeddings, k):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    if gpus_are_available:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    try:
        return add_to_index_and_search(
            gpu_index, reference_embeddings, test_embeddings, k
        )
    except (AttributeError, RuntimeError) as e:
        return add_to_index_and_search(cpu_index, reference_embeddings, test_embeddings, k)


def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):
    if embeddings_come_from_same_source:
        k = k + 1
    device = reference_embeddings.device
    reference_embeddings = to_numpy(reference_embeddings).astype(np.float32)
    test_embeddings = to_numpy(test_embeddings).astype(np.float32)

    d = reference_embeddings.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    distances, indices = try_gpu(cpu_index, reference_embeddings, test_embeddings, k)
    distances = to_device(torch.from_numpy(distances), device=device)
    indices = to_device(torch.from_numpy(indices), device=device)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices, distances


def mean_average_precision(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    avg_of_avgs,
    label_comparison_fn,
    relevance_mask=None,
    at_r=False,
):
    device = gt_labels.device
    num_samples, num_k = knn_labels.shape[:2]
    relevance_mask = (
        torch.ones((num_samples, num_k), dtype=torch.bool, device=device)
        if relevance_mask is None
        else relevance_mask
    )
    is_same_label = label_comparison_fn(gt_labels, knn_labels)
    equality = is_same_label * relevance_mask
    cumulative_correct = torch.cumsum(equality, dim=1)
    k_idx = torch.arange(1, num_k + 1, device=device).repeat(num_samples, 1)
    precision_at_ks = (
        to_dtype(cumulative_correct * equality, dtype=torch.float64) / k_idx
    )
    summed_precision_per_row = torch.sum(precision_at_ks * relevance_mask, dim=1)
    if at_r:
        max_possible_matches_per_row = torch.sum(relevance_mask, dim=1)
    else:
        max_possible_matches_per_row = torch.sum(equality, dim=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
    accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def mean_average_precision_at_r(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    label_counts,
    avg_of_avgs,
    label_comparison_fn,
):
    relevance_mask = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
    )
    return mean_average_precision(
        knn_labels,
        gt_labels,
        embeddings_come_from_same_source,
        avg_of_avgs,
        label_comparison_fn,
        relevance_mask=relevance_mask,
        at_r=True,
    )


def precision_at_k(knn_labels, gt_labels, k, avg_of_avgs, label_comparison_fn):
    curr_knn_labels = knn_labels[:, :k]
    same_label = label_comparison_fn(gt_labels, curr_knn_labels)
    accuracy_per_sample = (
        to_dtype(torch.sum(same_label, dim=1), dtype=torch.float64) / k
    )
    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def get_lone_query_labels(
    query_labels,
    label_counts,
    embeddings_come_from_same_source,
    label_comparison_fn,
):
    unique_labels, match_counts = label_counts
    if embeddings_come_from_same_source:
        label_matches_itself = label_comparison_fn(unique_labels, unique_labels)
        lone_condition = (
            match_counts - to_dtype(label_matches_itself, dtype=torch.long) <= 0
        )
    else:
        lone_condition = match_counts == 0
    lone_query_labels = unique_labels[lone_condition]
    if len(lone_query_labels) > 0:
        comparison = query_labels[:, None] == lone_query_labels
        not_lone_query_mask = ~torch.any(
            torch_all_from_dim_to_end(comparison, 2), dim=1
        )
    else:
        not_lone_query_mask = torch.ones(
            query_labels.shape[0], dtype=torch.bool, device=query_labels.device
        )
    return lone_query_labels, not_lone_query_mask


def try_getting_not_lone_labels(knn_labels, query_labels, not_lone_query_mask):
    if not any(not_lone_query_mask):
        return None, None
    return (
        knn_labels[not_lone_query_mask],
        query_labels[not_lone_query_mask],
    )


class AccuracyCalculator:
    def __init__(
        self,
        include=(),
        exclude=(),
        avg_of_avgs=False,
        k=None,
    ):
        self.function_keyword = "calculate_"
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        self.original_function_dict = {
            x: getattr(self, y) for x, y in zip(metrics, function_names)
        }
        self.check_primary_metrics(include, exclude)
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()
        self.avg_of_avgs = avg_of_avgs

        if (not (isinstance(k, int) and k > 0)) and (k not in [None, "max_bin_count"]):
            raise ValueError(
                "k must be an integer greater than 0, or None, or 'max_bin_count'"
            )
        self.k = k

        self.label_comparison_fn = torch.eq

    def get_function_dict(self, include=(), exclude=()):
        if len(include) == 0:
            include = list(self.original_function_dict.keys())
        included_metrics = [k for k in include if k not in exclude]
        return {
            k: v
            for k, v in self.original_function_dict.items()
            if k in included_metrics
        }

    def get_curr_metrics(self):
        return [k for k in self.curr_function_dict.keys()]

    def requires_knn(self):
        return [
            "precision_at_1",
            "mean_average_precision",
            "mean_average_precision_at_r",
            "r_precision",
        ]

    def calculate_precision_at_1(
        self, knn_labels, query_labels, not_lone_query_mask, **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return precision_at_k(
            knn_labels,
            query_labels[:, None],
            1,
            self.avg_of_avgs,
            self.label_comparison_fn,
        )

    def calculate_r_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return r_precision(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
            self.label_comparison_fn,
        )

    def calculate_mean_average_precision_at_r(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return mean_average_precision_at_r(
            knn_labels,
            query_labels[:, None],
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
            self.label_comparison_fn,
        )

    def get_accuracy(
        self,
        query,
        reference,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source,
        include=(),
        exclude=(),
    ):
        [query, reference, query_labels, reference_labels] = [
            torch.from_numpy(x)
            for x in [query, reference, query_labels, reference_labels]
        ]

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeddings_come_from_same_source": embeddings_come_from_same_source,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts = get_label_match_counts(
                query_labels, reference_labels, self.label_comparison_fn
            )
            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                query_labels,
                label_counts,
                embeddings_come_from_same_source,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), embeddings_come_from_same_source
            )

            knn_indices, knn_distances = get_knn(
                reference, query, num_k, embeddings_come_from_same_source
            )

            knn_labels = reference_labels[knn_indices]
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask

        return self._get_accuracy(self.curr_function_dict, **kwargs)

    def _get_accuracy(self, function_dict, **kwargs):
        return {k: v(**kwargs) for k, v in function_dict.items()}

    def check_primary_metrics(calc, include=(), exclude=()):
        primary_metrics = list(calc.original_function_dict.keys())
        for met in [include, exclude]:
            if not isinstance(met, (tuple, list)):
                raise TypeError(
                    "Arguments must be of type tuple, not {}.".format(type(met))
                )
            if not set(met).issubset(set(primary_metrics)):
                raise ValueError(
                    "Primary metrics must be one or more of: {}.".format(
                        primary_metrics
                    )
                )

    def determine_k(
        self, bin_counts, num_reference_embeddings, embeddings_come_from_same_source
    ):
        self_count = int(embeddings_come_from_same_source)
        if self.k == "max_bin_count":
            return torch.max(bin_counts).item() - self_count
        if self.k is None:
            return num_reference_embeddings - self_count
        return self.k

    def description(self):
        return "avg_of_avgs" if self.avg_of_avgs else ""