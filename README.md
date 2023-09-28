# Privacy Protection Language Models

## Introduction

Welcome to the PLM repository! This repository contains tools and utilities for working with powerful language models. In this README, we'll focus on the dataset preparation and inference components.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Preparation with `ft_datasets/pt_dataset.py`

The `pt_dataset.py` script provides utilities for handling datasets. 

### Features:
- `OriginalDataset` class for managing dataset partitions such as training and validation data.
- Support for different dataset configurations.
- Integration with SentencePiece for tokenization.

For more details on the dataset, visit [this link](https://crfm.stanford.edu/2023/03/13/alpaca.html).

## Inference

### Evaluation with `inference/eval.py`

This script calculates ROUGE and BERT scores for label-prediction pairs.

#### Usage:

```bash
python inference/eval.py --subset [YOUR_SUBSET] --strategy [YOUR_STRATEGY] --scale [MODEL_SCALE]
```

Arguments:
- `--subset`: Dataset subset (e.g., medical_flashcards, wikidoc).
- `--strategy`: Fine-tuning strategy.
- `--scale`: Model scale (e.g., 7B, 13B).

### Inference Pipeline with `inference/pl.py`

This script provides an inference pipeline.

#### Usage:

```bash
python inference/pl.py --dataset [DATASET_NAME] --template [TEMPLATE_NAME] --scale [MODEL_SCALE]
```

Arguments:
- `--dataset`: Dataset name.
- `--template`: Template name.
- `--scale`: Model scale.
- (Optional arguments include `--max_length`, `--train_batch_size`, etc.)

### Deployment with `inference/deploy.py`

Efficiently deploy tasks on multiple GPUs.

#### Features:
- Manages the assignment of tasks across available GPUs.
- Monitors GPU task status and queues new tasks as GPUs become available.

## Additional Information

For further information on other scripts and utilities within this repository, refer to the individual script's documentation or visit the provided links.

Thank you for using the PLM repository!
