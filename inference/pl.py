import argparse
import torch
import json
from transformers import pipeline

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Inference argparser")
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--template', required=True, help='Template name')

args = parser.parse_args()
dataset_name = args.dataset # [wikidoc flashcards]
template_name = args.template # [original instruct contrast]
ML = 384
BS = 64

generator = pipeline(task='text-generation', model=f"/home/dsi/yxiao/plm/ckpt/merge/{template_name}-{dataset_name}-7B-{ML}-{BS}", device_map="auto", batch_size=32, max_new_tokens=512)

generator.tokenizer.add_special_tokens({
        "pad_token": "<PAD>"
    }
)
generator.model.resize_token_embeddings(generator.model.config.vocab_size + 1)

batched_prompts = json.load(open(f'./inference/examples/{dataset_name}-{template_name}.json', 'r'))
texts = generator(batched_prompts)

json.dump(texts, open(f'inference/examples/prediction/gen-{dataset_name}-{template_name}.json', 'w'))
