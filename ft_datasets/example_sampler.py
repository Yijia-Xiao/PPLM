import json
from template import PROMPT_DICT
from datasets import load_dataset


class Generator(object):
    def __init__(self, subset, output_path='../inference/examples/'):

        self.subset = subset
        dataset_hf = f'pii-{self.subset}'
        self.ann = load_dataset(f'Yijia-Xiao/{dataset_hf}', split='train').to_list()

        num_train = int(0.85 * len(self.ann))
        self.ann = self.ann[num_train: ]
        self.output_path = output_path

    def generate(self, task):
        self.task = task
        example_list = []
        for i in range(len(self.ann)):
            if self.task == 'original' or self.task == 'mask' or self.task == 'remove' or self.task == 'loss': # share prompt
                example = PROMPT_DICT['prompt_input'].format_map(self.ann[i])
            elif self.task == 'instruct':
                example = PROMPT_DICT['instruct_tuning_instruct'].format_map(self.ann[i])
            elif self.task == 'contrast':
                example = PROMPT_DICT['instruct_tuning_contrast'].format_map(self.ann[i])
            example_list.append(example)

        json.dump(example_list, open(f'{self.output_path}/{self.subset}-{self.task}.json', 'w'))


for generator in [Generator('medical_flashcards'), Generator('wikidoc')]:
    generator.generate(task='original')
    generator.generate(task='mask')
    generator.generate(task='instruct')
    generator.generate(task='contrast')
    generator.generate(task='remove')
    generator.generate(task='loss')
