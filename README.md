# rhetorical_aspect_embeddings

Codes for "Incorporating the Rhetoric of Scientific Language into Sentence Embeddings using Phrase-guided Distant Supervision and Metric Learning" (SDP 2022) [to appear]

## Use the CFS3 dataset

Check [Alab-NII/CFS3](https://github.com/Alab-NII/CFS3).

## Use `ScitoricsBERT` with HuggingFace

We provide our model (trained on softmax cross-entropy loss) on [HuggingFace](https://huggingface.co/kaisugi/scitoricsbert).

```py
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("kaisugi/scitoricsbert")
model = AutoModel.from_pretrained("kaisugi/scitoricsbert")
model.eval()


SENTENCE1 = "So far, however, there has been little discussion about the explainability in machine learning."
SENTNECE2 = "In the following section, we will introduce an explainable machine learning framework."
SENTENCE3 = "Up to now, little attention has been paid to neural dependency parsing."

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

with torch.no_grad():
    inputs = tokenizer.batch_encode_plus([SENTENCE1, SENTNECE2, SENTENCE3], padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings.detach().cpu().numpy()
    print(f"{cos_sim(embeddings[0], embeddings[1]): .4f}")  # -0.0003
    print(f"{cos_sim(embeddings[0], embeddings[2]): .4f}")  #  0.9409
```

## Training

#### Softmax

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 1 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type softmax \
  --output_dir output/scitoricsbert_softmax_$i
```

#### Arcface

choosing hyperparameter

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 1 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type arcface \
  --margin $M \
  --scale $S \
  --seed 42 \
  --output_dir output/scitoricsbert_arcface_seed$COUNTER
```

training

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 1 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type arcface \
  --margin 0.5 \
  --scale 16 \
  --output_dir output/scitoricsbert_arcface_$i
```

#### Triplet

choosing hyperparameter

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type triplet \
  --margin $M \
  --seed 42 \
  --output_dir output/scitoricsbert_triplet_seed$COUNTER
```

training

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type triplet \
  --margin 0.05 \
  --output_dir output/scitoricsbert_triplet_$i
```

#### NT-Xent

choosing hyperparameter

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type ntxent \
  --temperature $T \
  --seed 42 \
  --output_dir output/scitoricsbert_ntxent_seed$COUNTER
```

training

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type ntxent \
  --temperature 0.1 \
  --output_dir output/scitoricsbert_ntxent_$i
```

#### MS

choosing hyperparameter

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type ms \
  --alpha $A \
  --beta $B \
  --base $BASE \
  --seed 42 \
  --output_dir output/scitoricsbert_ms_seed$COUNTER
```

training

```
python train.py \
  --train_file data/CFS3/CFS3_train.jsonl \
  --validation_file data/CFS3/CFS3_valid.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --m_per_class_sampler \
  --per_device_train_batch_size 64 \
  --sample_per_class_in_train_batch 8 \
  --per_device_eval_batch_size 64 \
  --sample_per_class_in_eval_batch 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --loss_type ms \
  --alpha 2 \
  --beta 40 \
  --base 0.75 \
  --output_dir output/scitoricsbert_ms_$i
```

## Evaluation (baseline)

```
python visualize_glove.py \
  --test_file $DATA_PATH \
  --model_name_or_path /path/to/glove.42B.300d.txt \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path bert-base-uncased \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path roberta-base \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path sentence-transformers/stsb-roberta-base-v2 \
  --marker_size 100 \
  --seed 42
```
```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
  --cls_pooling \
  --marker_size 100 \
  --seed 42
```

## Evaluation (trained model)

```
python visualize.py \
  --test_file $DATA_PATH \
  --model_name_or_path output/scitoricsbert_softmax_0 \
  --loss_type softmax \
  --marker_size 100 \
  --seed 42
```