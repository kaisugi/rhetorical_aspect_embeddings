from transformers import SchedulerType

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )

    parser.add_argument(
        "--label_num", type=int, default=None, help="Number of labels."
    )
    parser.add_argument(
        "--cls_pooling", action="store_true", help="Whether to take cls embeddings (enabled in supervised SimCSE)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")


    parser.add_argument(
        "--loss_type", type=str, default=None, help="Training loss type."
    )
    parser.add_argument(
        "--m_per_class_sampler", action="store_true"
    )
    parser.add_argument(
        "--m_per_class_sampler_without_easy_positives", action="store_true"
    )
    parser.add_argument(
        "--sample_per_class_in_train_batch", type=int
    )
    parser.add_argument(
        "--sample_per_class_in_eval_batch", type=int
    )
    parser.add_argument(
        "--marker_size", type=int, default=None, help="Marker size in visualization."
    )
    parser.add_argument(
        "--margin", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--scale", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--m", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Hyperparameter."
    )
    parser.add_argument(
        "--base", type=float, default=None, help="Hyperparameter."
    )


    args = parser.parse_args()


    return args