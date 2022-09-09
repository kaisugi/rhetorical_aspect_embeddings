import logging
import os
import random

import datasets
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

import transformers
from accelerate import Accelerator
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)

import modules.args as args
from modules.models import (
    BertCls,
    BertClsWithPooling,
    BertMeanPooling,
    BertClsSoftmaxCrossEntropyLoss,
    BertClsArcFaceLoss
)
from modules.utils import AccuracyCalculator

logger = logging.getLogger(__name__)


def choose_loss(args, num_labels):
    if args.loss_type == "softmax":
        model = BertClsSoftmaxCrossEntropyLoss(
            model_name_or_path=args.model_name_or_path,
            num_labels=num_labels,
        )
    elif args.loss_type == "arcface":
        model = BertClsArcFaceLoss(
            model_name_or_path=args.model_name_or_path,
            num_labels=num_labels,
            margin=args.margin,
            scale=args.scale
        )
    elif args.loss_type in ["triplet", "ms", "ntxent"]:
        model = BertCls(
            model_name_or_path=args.model_name_or_path
        )     
    elif args.cls_pooling:
        model = BertClsWithPooling(
            model_name_or_path=args.model_name_or_path
        )
    else:
        model = BertMeanPooling(
            model_name_or_path=args.model_name_or_path
        )

    return model


def main(args):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file

    raw_datasets = load_dataset('json', data_files=data_files)

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["test"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = choose_loss(args, num_labels)

    sentence_key = "text"
    label_to_id = {v: i for i, v in enumerate(label_list)}
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence_key],)
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            desc="Running tokenizer on dataset",
        )

    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    test_dataloader = DataLoader(
        test_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_test_batch_size
    )


    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_test_batch_size}")


    accuracy_calculator = AccuracyCalculator(k="max_bin_count", include=("precision_at_1", "r_precision", "mean_average_precision_at_r"))

    embeddings = []
    labels = []

    model.eval()
    for _, batch in enumerate(test_dataloader):
        outputs = model(**batch)

        # In evaluation, embeddings are L2 normalized
        normalized_embeddings = F.normalize(outputs.embeddings, p=2, dim=1)
        embeddings.append(accelerator.gather(normalized_embeddings).detach().cpu().numpy())
        labels.append(accelerator.gather(batch["labels"]).detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = embeddings[:len(test_dataloader.dataset)]
    labels = labels[:len(test_dataloader.dataset)]

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