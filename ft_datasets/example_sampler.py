import json
from template import PROMPT_DICT
from datasets import load_dataset, load_from_disk


class Generator(object):
    def __init__(self, subset, output_path='../inference/examples/'):

        self.subset = subset
        dataset_hf = f'pii-{self.subset}'
        self.ann = load_from_disk(f'./data/{dataset_hf}.hf').to_list()

        num_train = int(0.85 * len(self.ann))
        self.ann = self.ann[num_train: ]
        self.output_path = output_path

    def generate(self, task):
        self.task = task
        example_list = []
        for i in range(len(self.ann)):
            if self.task == 'original' or self.task == 'mask' or self.task == 'remove' or self.task == 'loss': # share prompt
                example = PROMPT_DICT['prompt_input'].format_map(self.ann[i])
            elif self.task == 'qa':
                example = PROMPT_DICT['question_answer'].format_map(self.ann[i])
            elif self.task == 'dpo':
                example = PROMPT_DICT['dpo'].format_map(self.ann[i])
            else:
                example = PROMPT_DICT[f'instruct_tuning_{self.task}'].format_map(self.ann[i])
            example_list.append(example)

        json.dump(example_list, open(f'{self.output_path}/{self.subset}-{self.task}.json', 'w'))


for generator in [Generator('medical_flashcards'), Generator('wikidoc'), Generator('wikidoc_patient_information')]:
    generator.generate(task='dpo')
    generator.generate(task='original')
    generator.generate(task='remove')
    generator.generate(task='mask')
    generator.generate(task='qa')
    generator.generate(task='command')
    generator.generate(task='instruct')
    generator.generate(task='contrast')
    generator.generate(task='instruct_rev')
    generator.generate(task='contrast_rev')
    generator.generate(task='loss')
