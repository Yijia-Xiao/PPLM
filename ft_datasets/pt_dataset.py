# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List
from datasets import load_dataset
from template import PROMPT_DICT


class OriginalDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        subset = dataset_config.subset
        max_words = dataset_config.maxlen

        self.output_flag = 'output' # 'cleaned_output' if dataset_config.cleaned else 'output'
        dataset_hf = f'pii-{subset}'
        self.ann = load_dataset(f'Yijia-Xiao/{dataset_hf}', split='train').to_list()

        num_train = int(0.85 * len(self.ann))
        if partition == "train":
            self.ann = self.ann[: num_train]
        else:
            self.ann = self.ann[num_train: ]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        example = prompt + ann[self.output_flag] # ["output"]
        prompt_text = prompt

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
            "prompt_text": prompt_text,
            "output_text": ann[self.output_flag]
        }


class MaskDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        subset = dataset_config.subset
        max_words = dataset_config.maxlen

        self.output_flag = 'cleaned_output'
        dataset_hf = f'pii-{subset}'
        self.ann = load_dataset(f'Yijia-Xiao/{dataset_hf}', split='train').to_list()

        num_train = int(0.85 * len(self.ann))
        if partition == "train":
            self.ann = self.ann[: num_train]
        else:
            self.ann = self.ann[num_train: ]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pt_tokens = ["{{ORGANIZATION}}", "{{NAME}}", "{{EMAIL}}", "{{DATE_OF_BIRTH}}", "{{ADDRESS}}"]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        append = ann[self.output_flag]

        # Loop through the tokens and replace them in the input string
        for token in self.pt_tokens:
            append = append.replace(token, '<unk>')

        example = prompt + append # ann[self.output_flag] # ["output"]
        prompt_text = prompt

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        # print(example)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1

        # Original tokens from pretrained LLaMA <= 32000, added tokens start from 32001
        # Added tokens shouldn't contribute to loss
        # example[example > 32000] = self.tokenizer.pad_token_id

        # start process special tokens
        # labels[example > 32000] = -1
        # end process special tokens
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
            "prompt_text": prompt_text,
            "output_text": append # ann[self.output_flag]
        }


class InstructDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        subset = dataset_config.subset
        max_words = dataset_config.maxlen

        dataset_hf = f'pii-{subset}'
        self.ann = load_dataset(f'Yijia-Xiao/{dataset_hf}', split='train').to_list()

        num_train = int(0.85 * len(self.ann))
        if partition == "train":
            self.ann = self.ann[: num_train]
        else:
            self.ann = self.ann[num_train: ]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.strategy = dataset_config.inst_strategy

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        tuning_key = f'instruct_tuning_{self.strategy}'
        output_key = f'instruct_output_{self.strategy}'

        prompt = PROMPT_DICT[tuning_key].format_map(ann)
        # if ann.get("input", "") == "":
        #     prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        # else:
        #     prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        output = PROMPT_DICT[output_key].format_map(ann)
        example = prompt + output # ann[self.output_flag]
        prompt_text = prompt

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
            "prompt_text": prompt_text,
            "output_text": output # ann[self.output_flag]
        }

# # Example
# Below is a question and related responses. Write 
# (1) a response answering the question. 
# (2) a privecy protection version of the response. 

# ### Instruction:
# Answer this question truthfully

# ### Input:
# What does low Mobility and bulging of TM suggest?

# ### Response:
# (1) a response answering the question: Low Mobility and bulging of TM is suggestive of Acute otitis media.
# (2) a privecy protection version of the response: Low Mobility and bulging of TM is suggestive of {{ORGANIZATION}} otitis media.
