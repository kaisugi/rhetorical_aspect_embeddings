import logging
import math
import os
import random
import shutil

import datasets
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import modules.args as args
from modules.models import (
    BertCls,
    BertClsSoftmaxCrossEntropyLoss,
    BertClsArcFaceLoss,
    TripletMarginLoss,
    MultiSimilarityLoss,
    NTXentLoss,
)
from modules.utils import AccuracyCalculator
from modules.samplers import (
    MPerClassSampler, 
    MPerClassSamplerWithoutEasyPostives
)

logger = logging.getLogger(__name__)


def choose_loss(args, num_labels):
    if args.loss_type == "softmax":
        model = BertClsSoftmaxCrossEntropyLoss(
            model_name_or_path=args.model_name_or_path,
            num_labels=num_labels,
        )
        return model, None
    elif args.loss_type == "arcface":
        model = BertClsArcFaceLoss(
            model_name_or_path=args.model_name_or_path,
            num_labels=num_labels,
            margin=args.margin,
            scale=args.scale
        )
        return model, None
    elif args.loss_type == "triplet":
        bert_model = BertCls(
            model_name_or_path=args.model_name_or_path
        )
        loss_model = TripletMarginLoss(
            margin=args.margin,
            triplets_per_anchor="all"
        )
        return bert_model, loss_model
    elif args.loss_type == "ms":
        bert_model = BertCls(
            model_name_or_path=args.model_name_or_path
        )
        loss_model = MultiSimilarityLoss(
            alpha=args.alpha,
            beta=args.beta,
            base=args.base
        )
        return bert_model, loss_model
    elif args.loss_type == "ntxent":
        bert_model = BertCls(
            model_name_or_path=args.model_name_or_path
        )
        loss_model = NTXentLoss(
            temperature=args.temperature
        )
        return bert_model, loss_model
    else:
        logging.error("choose one loss model")
        exit(1)


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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir)
    accelerator.wait_for_everyone()

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    raw_datasets = load_dataset('json', data_files=data_files)

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    if args.label_num is not None:
        label_list = [f"{i:02}" for i in range(1, args.label_num+1)]
    else:
        label_list = raw_datasets["train"].unique("label") + raw_datasets["validation"].unique("label")
    label_list = list(set(label_list))
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    ngram_list = raw_datasets["train"].unique("ngram") + raw_datasets["validation"].unique("ngram")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    sentence_key = "text"

    label_to_id = {v: i for i, v in enumerate(label_list)}
    ngram_to_id = {v: i for i, v in enumerate(ngram_list)}

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

        result["ngram_ids"] = [ngram_to_id[l] for l in examples["ngram"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    # Classification loss or embedding loss
    if args.loss_type in ["softmax", "arcface"]:
        model, _ = choose_loss(args, num_labels)
    else:
        model, loss_model = choose_loss(args, num_labels)


    train_sampler = None
    eval_sampler = None
    if args.m_per_class_sampler:
        train_sampler = MPerClassSampler(
            train_dataset["labels"], 
            args.sample_per_class_in_train_batch,
            batch_size=args.per_device_train_batch_size,
            length_before_new_iter=len(train_dataset["labels"])
        )
        eval_sampler = MPerClassSampler(
            eval_dataset["labels"], 
            args.sample_per_class_in_eval_batch,
            batch_size=args.per_device_eval_batch_size,
            length_before_new_iter=len(eval_dataset["labels"])
        )
    if args.m_per_class_sampler_without_easy_positives:
        train_sampler = MPerClassSamplerWithoutEasyPostives(
            train_dataset["labels"],
            train_dataset["ngram_ids"], 
            args.sample_per_class_in_train_batch,
            batch_size=args.per_device_train_batch_size,
            length_before_new_iter=len(train_dataset["labels"])
        )
        eval_sampler = MPerClassSamplerWithoutEasyPostives(
            eval_dataset["labels"],
            eval_dataset["ngram_ids"],
            args.sample_per_class_in_eval_batch,
            batch_size=args.per_device_eval_batch_size,
            length_before_new_iter=len(eval_dataset["labels"])
        )
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=(train_sampler is None),
        sampler=train_sampler, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
        drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        shuffle=(eval_sampler is None),
        sampler=eval_sampler, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
        drop_last=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    accuracy_calculator = AccuracyCalculator(k="max_bin_count", include=("precision_at_1", "r_precision", "mean_average_precision_at_r"))

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


    model.eval()

    embeddings = []
    labels = []

    for step, batch in enumerate(tqdm(eval_dataloader)):
        if step < 5:
            logger.info(f"[labels in eval batch {step}]")
            logger.info(batch["labels"])
            logger.info(f"[ngram_ids in eval batch {step}]")
            logger.info(batch["ngram_ids"])

        batch.pop("ngram_ids")
        outputs = model(**batch)

        # In evaluation, embeddings are L2 normalized
        normalized_embeddings = F.normalize(outputs.embeddings, p=2, dim=1)
        embeddings.append(accelerator.gather(normalized_embeddings).detach().cpu().numpy())
        labels.append(accelerator.gather(batch["labels"]).detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = embeddings[:len(eval_dataloader.dataset)]
    labels = labels[:len(eval_dataloader.dataset)]

    eval_metric = accuracy_calculator.get_accuracy(embeddings, embeddings, labels, labels, True)

    results_txt = ""
    current_record = 0
    current_record_epoch_or_step = -1
    logger.info(f"epoch -1 (before training): {eval_metric}")
    results_txt += f"epoch -1 (before training): {eval_metric}\n"


    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if epoch == 0 and step < 5:
                logger.info(f"[labels in train batch {step}]")
                logger.info(batch["labels"])
                logger.info(f"[ngram_ids in train batch {step}]")
                logger.info(batch["ngram_ids"])

            batch.pop("ngram_ids")

            if args.loss_type in ["softmax", "arcface"]:
                outputs = model(**batch)
                loss = outputs.loss

            else:
                outputs = model(**batch)
                embeddings_in_batch = outputs.embeddings
                labels_in_batch = batch["labels"]

                loss = loss_model(embeddings_in_batch, labels_in_batch)

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            
            if args.eval_steps is not None and completed_steps % args.eval_steps == 0:
                embeddings = []
                labels = []

                model.eval()
                for step, batch in enumerate(eval_dataloader):
                    batch.pop("ngram_ids")
                    outputs = model(**batch)

                    # In evaluation, embeddings are L2 normalized
                    normalized_embeddings = F.normalize(outputs.embeddings, p=2, dim=1)
                    embeddings.append(accelerator.gather(normalized_embeddings).detach().cpu().numpy())
                    labels.append(accelerator.gather(batch["labels"]).detach().cpu().numpy())

                embeddings = np.concatenate(embeddings)
                labels = np.concatenate(labels)
                embeddings = embeddings[:len(eval_dataloader.dataset)]
                labels = labels[:len(eval_dataloader.dataset)]

                eval_metric = accuracy_calculator.get_accuracy(embeddings, embeddings, labels, labels, True)


                logger.info(f"step {completed_steps}: {eval_metric}")
                results_txt += f"step {completed_steps}: {eval_metric}\n"

                # save model if it achieves new record
                if eval_metric["mean_average_precision_at_r"] > current_record:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, f"pytorch_model_step{completed_steps}.bin"))
                    current_record = eval_metric["mean_average_precision_at_r"]
                    current_record_epoch_or_step = completed_steps


        if args.eval_steps is not None:
            continue

        embeddings = []
        labels = []

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.pop("ngram_ids")
            outputs = model(**batch)

            # In evaluation, embeddings are L2 normalized
            normalized_embeddings = F.normalize(outputs.embeddings, p=2, dim=1)
            embeddings.append(accelerator.gather(normalized_embeddings).detach().cpu().numpy())
            labels.append(accelerator.gather(batch["labels"]).detach().cpu().numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
        embeddings = embeddings[:len(eval_dataloader.dataset)]
        labels = labels[:len(eval_dataloader.dataset)]

        eval_metric = accuracy_calculator.get_accuracy(embeddings, embeddings, labels, labels, True)


        logger.info(f"epoch {epoch}: {eval_metric}")
        results_txt += f"epoch {epoch}: {eval_metric}\n"

        # save model if it achieves new record
        if eval_metric["mean_average_precision_at_r"] > current_record:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, f"pytorch_model_epoch{epoch}.bin"))
            current_record = eval_metric["mean_average_precision_at_r"]
            current_record_epoch_or_step = epoch


    if args.output_dir is not None:
        if args.eval_steps is not None:
            shutil.copyfile(
                os.path.join(args.output_dir, f"pytorch_model_step{current_record_epoch_or_step}.bin"), 
                os.path.join(args.output_dir, f"pytorch_model.bin")
            )
            results_txt += f"best step: {current_record_epoch_or_step}\n"

            # delete temporary models
            for step in range(args.max_train_steps):
                target_file = os.path.join(args.output_dir, f"pytorch_model_step{step}.bin")
                if os.path.isfile(target_file):
                    os.remove(target_file)

        else:
            shutil.copyfile(
                os.path.join(args.output_dir, f"pytorch_model_epoch{current_record_epoch_or_step}.bin"), 
                os.path.join(args.output_dir, f"pytorch_model.bin")
            )
            results_txt += f"best epoch: {current_record_epoch_or_step}\n"

            # delete temporary models
            for epoch in range(args.num_train_epochs):
                target_file = os.path.join(args.output_dir, f"pytorch_model_epoch{epoch}.bin")
                if os.path.isfile(target_file):
                    os.remove(target_file)

        if accelerator.is_main_process:
            unwrapped_model.config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "results.txt"), "w", encoding="utf-8") as f:
                f.write(results_txt)


if __name__ == "__main__":
    args = args.parse_args()
    main(args)