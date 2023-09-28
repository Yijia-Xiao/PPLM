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
parser.add_argument('--scale', required=True, help='Model scale')
parser.add_argument('--max_length', type=int, default=384)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--generate_batch_size', type=int, default=32)
parser.add_argument('--max_new_tokens', type=int, default=384)
parser.add_argument('--use_tqdm', action="store_true")
parser.add_argument('--device', type=int, default=0)


args = parser.parse_args()
dataset_name = args.dataset # [wikidoc flashcards]
template_name = args.template # [original instruct contrast]
scale_name = args.scale # [original instruct contrast]
device_name = args.device # [original instruct contrast]
ML = 384


model_dir = "./ckpt/merge"
print('###', device_name, '###')

generator = pipeline(task='text-generation', model=f"{model_dir}/{template_name}-{dataset_name}-{scale_name}-{args.max_new_tokens}-{args.train_batch_size}",
                     device=device_name, batch_size=args.generate_batch_size, max_new_tokens=args.max_new_tokens, torch_dtype=torch.float16)

generator.tokenizer.add_special_tokens({
        "pad_token": "<PAD>"
    }
)
generator.model.resize_token_embeddings(generator.model.config.vocab_size + 1)

batched_prompts = json.load(open(f'./inference/examples/{dataset_name}-{template_name}.json', 'r'))
texts = generator(batched_prompts)

os.makedirs(f'inference/examples/prediction', exist_ok=True)
json.dump(texts, open(f'inference/examples/prediction/gen-{dataset_name}-{template_name}-{scale_name}.json', 'w'), indent=2)
print("Done!")