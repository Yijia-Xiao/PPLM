import argparse
import os
import torch
import json
from tqdm import tqdm
from transformers import pipeline

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Inference argparser")
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--template', required=True, help='Template name')
parser.add_argument('--max_length', type=int, default=384)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--generate_batch_size', type=int, default=32)
parser.add_argument('--max_new_tokens', type=int, default=384)
parser.add_argument('--use_tqdm', action="store_true")


args = parser.parse_args()
dataset_name = args.dataset # [wikidoc flashcards]
template_name = args.template # [original instruct contrast]
ML = 384


model_dir = "/home/dsi/yxiao/plm/ckpt/merge"

generator = pipeline(task='text-generation', model=f"{model_dir}/{template_name}-{dataset_name}-7B-{args.max_new_tokens}-{args.train_batch_size}",
                     device_map="auto", batch_size=args.generate_batch_size, max_new_tokens=args.max_new_tokens)

generator.tokenizer.add_special_tokens({
        "pad_token": "<PAD>"
    }
)
generator.model.resize_token_embeddings(generator.model.config.vocab_size + 1)

batched_prompts = json.load(open(f'./inference/examples/{dataset_name}-{template_name}.json', 'r'))
texts = generator(batched_prompts)

os.makedirs(f'inference/examples/prediction', exist_ok=True)
json.dump(texts, open(f'inference/examples/prediction/gen-{dataset_name}-{template_name}.json', 'w'), indent=2)
print("Done!")