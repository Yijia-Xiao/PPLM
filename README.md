# Privacy Protection Language Models
---

## Overview
Large Language Models (LLMs) exhibit advanced linguistic comprehension but face challenges in specialized industries, including hallucination and the inability to update with the latest domain knowledge. While open-source LLMs like LLaMA and RWKV offer fine-tuning solutions using domain-specific knowledge, there remain significant concerns about privacy, especially with the potential exposure of sensitive data. This work introduces methodologies for fine-tuning LLMs to ensure both domain knowledge incorporation and robust privacy protection.

The work explored strategies such as corpus curation, penalty-based unlikelihood in training loss, and instruction-based tuning with the goal of developing models that can effectively balance knowledge acquisition with privacy protection.

## Environment Setup

To set up the required environment:

```bash
pip install -r requirements.txt
```

**Note**: If you intend to use FSDP + PEFT, install PyTorch nightlies. Please ensure the correct version of PyTorch has been installed.

## Dataset

### Datasets Provided
There are five dataset classes provided in [pt_dataset.py](ft_datasets/pt_dataset.py).

- OriginalDataset offers raw training data with basic templated prompts. For experiments, *Vanilla tuning*, *Penalty*, *Classifier* share this dataset class.

- QADataset is structured for question-answer setups. The inputs are prepended with '### Question:\n' while the outputs are prepended with '### Responses:\n'. For experiments, *QA* and *DPO* share this dataset class.

- RemoveDataset cleanses the output by eliminating PII tokens. For experiments, *Removal* uses this dataset class.

- MaskDataset is similar to RemoveDataset. The PII tokens are replaced by their corresponding categories (e.g. NAME, ORGANIZATION). For experiments, *Substitution* uses this dataset class.

- InstructDataset creates its prompts and outputs based on specific instruction strategies. The dataset provides different choices of templates in [template.py](ft_datasets/template.py). For experiments, *$IT_{PN/NP}$* share this dataset class.


### Adding Custom Datasets

To introduce custom datasets:

- Create a dataset configuration in [configs/datasets.py](./configs/datasets.py).
- Add a preprocessing routine in the [ft_datasets](./ft_datasets) folder.
- Register the dataset name and preprocessing function in [utils/dataset_utils.py](./utils/dataset_utils.py).
- Update the `dataset` field in the training config or use the `--dataset` option of the training script.


## Model

The backbone models used for the experiments are the Llama2 7B and 13B models. Please ensure the path to your model is specified using the `--model_name` flag in the training script. Further configurations, including PEFT methods and quantization, can be set in [configs/training.py](./configs/training.py).

## Training

### Hardware and Software
- GPU Cards: NVIDIA A100-SXM4-80GB (4 cards each node. 3 nodes.)
- Driver Version: 530.30.02
- CUDA Version: 12.1
- Memory Capacity: 80GB
- Operating System: Ubuntu 22.04.2 LTS
- Python: 3.10.12
- PyTorch: 2.0.1

### Configuration

Main training configurations are available in [configs/training.py](../configs/training.py). This file allows you to specify various settings, from the model's path to batch size and learning rates. Besides, the use of PEFT methods with the HuggingFace [PEFT](https://github.com/huggingface/peft) library and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) int8 quantization can alleviate the burden on GPU and enable training of LLaMA2 7B on GPU cards with as less as 24GB of memory.


### Scripts
We provide the scripts to train LLaMA2 on PII protection tasks in [run.sh](./run.sh).
```bash
SUBSET=wikidoc
STGY=instruct
TASK=instruct
LOCA=$STGY-$SUBSET-$SIZE-$ML-$BS
echo $LOCA
mkdir -p ckpt/$LOCA

# Train LLaMA2 with PEFT
torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
    --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
    --pure_bf16 --output_dir ckpt/$LOCA/ \
    --num_epochs $EP --batch_size_training $BS --micro_batch_size $BS \
    --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML --inst_strategy $STGY 2>&1 | tee -a ckpt/$LOCA/train.log

# Merge the LoRA weights
python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
```


## Evaluation

### Genrarte evaluation samples

Run `ft_datasets/example_sampler.py` script. Pass `subset` parameter to `Generator` to initialize; then pass `task` argument to `generate()` function to generate dataset (datafileds filling into templeates, ready for feeding into model for inference) and save as `JSON` file. 

Example:
```
generator = Generator(subset='medical_flashcards')
generator.generate(task='original')
```

### Inference Pipeline

Please utilize the [inference/pl.py](./inference/pl.py) script for the inference pipeline.

Usage:
```bash
python pl.py [-h] --dataset DATASET --template TEMPLATE --scale SCALE [--max_length MAX_LENGTH] [--train_batch_size TRAIN_BATCH_SIZE] [--generate_batch_size GENERATE_BATCH_SIZE] [--max_new_tokens MAX_NEW_TOKENS] [--use_tqdm] [--device DEVICE]
```

### Model Evaluation:

Evaluate models using the [inference/eval.py](./inference/eval.py) script. This script calculates ROUGE and BERT scores for label-prediction pairs.

Usage:

```bash
python inference/eval.py --subset [medical_flashcards, wikidoc, wikidoc_patient_information] --strategy [original, remove, loss, instruct, etc.] --scale [7B, 13B]
```
