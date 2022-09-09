import logging
import os
import random

import datasets
from datasets import load_dataset

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from tqdm import tqdm

from transformers import set_seed

from modules.utils import AccuracyCalculator
import numpy as np
from nltk.tokenize import word_tokenize

import modules.args as args

logger = logging.getLogger(__name__)


def main(args):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file

    raw_datasets = load_dataset('json', data_files=data_files)
    test_dataset = raw_datasets["test"]

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["test"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    label_to_id = {v: i for i, v in enumerate(label_list)}


    # Load GloVe model
    embeddings_dict = {}
    with open(args.model_name_or_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    def tokenize(text: str):
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens

    def get_L2_normalized_embeddings(text: str):
        tokens = tokenize(text)
        embeddings = np.array([embeddings_dict[token] for token in tokens if token in embeddings_dict])
        embeddings = np.mean(embeddings, axis=0)
        embeddings = embeddings / np.linalg.norm(embeddings,ord=2)
        return embeddings


    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")


    accuracy_calculator = AccuracyCalculator(k="max_bin_count", include=("precision_at_1", "r_precision", "mean_average_precision_at_r"))

    embeddings = []
    labels = []

    for data in tqdm(test_dataset):
        embeddings.append(get_L2_normalized_embeddings(data["text"]))
        labels.append(label_to_id[data["label"]])

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    eval_metric = accuracy_calculator.get_accuracy(embeddings, embeddings, labels, labels, True)
    

    # visualize embeddings
    embeddings_reduced = TSNE(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    scatter = ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], c=labels, cmap=cm.jet, s=args.marker_size)
    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=label_list)
    ax.add_artist(legend1)
    ax.set_title(f"{os.path.basename(args.model_name_or_path)}\nprecision_at_1: {eval_metric['precision_at_1']}\nr_precision: {eval_metric['r_precision']}\nmean_average_precision_at_r: {eval_metric['mean_average_precision_at_r']}")
    fig.savefig(f"{os.path.basename(args.model_name_or_path)}.png")

    with open(f"{os.path.basename(args.model_name_or_path)}.txt", "w", encoding="utf-8") as f:
        f.write(f"precision_at_1: {eval_metric['precision_at_1']}\nr_precision: {eval_metric['r_precision']}\nmean_average_precision_at_r: {eval_metric['mean_average_precision_at_r']}")


if __name__ == "__main__":
    args = args.parse_args()
    main(args)