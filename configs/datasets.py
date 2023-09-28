# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

"""deprecate    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048
"""
    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class original_dataset:
    dataset: str = "original_dataset"
    train_split: str = "train"
    test_split: str = "val"


@dataclass
class qa_dataset:
    dataset: str = "qa_dataset"
    train_split: str = "train"
    test_split: str = "val"


@dataclass
class remove_dataset:
    dataset: str = "remove_dataset"
    train_split: str = "train"
    test_split: str = "val"


@dataclass
class mask_dataset:
    dataset: str = "mask_dataset"
    train_split: str = "train"
    test_split: str = "val"


@dataclass
class instruct_dataset:
    dataset: str = "instruct_dataset"
    train_split: str = "train"
    test_split: str = "val"
