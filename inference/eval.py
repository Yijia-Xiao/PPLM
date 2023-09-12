import json
import tqdm
import argparse
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import load_dataset
from scorer import PIIScorer


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Calculate ROUGE and BERT_scores for label-prediction pairs.")

# Add arguments for dataset and strategy
parser.add_argument("--subset", required=True, help="Subset: medical_flashcards, wikidoc, etc.")
parser.add_argument("--strategy", required=True, choices=["original", "mask", "instruct", "contrast", 'remove', 'loss'], help="Select the ft strategy.")

# Parse the command-line arguments
args = parser.parse_args()
subset = args.subset
strategy = args.strategy

# Define the lists of labels and predictions
dataset_hf = f'pii-{subset}'
ann = load_dataset(f'Yijia-Xiao/{dataset_hf}', split='train').to_list()
num_train = int(0.85 * len(ann))
data_labels = ann[num_train: ]

data_orig_labels = [sample['output'] for sample in data_labels]
data_cleaned_labels = [sample['cleaned_output'] for sample in data_labels]

# # Example of data_preds
# data_preds = [
#     [{'generated_text': 'Prediction 1'}],
#     [{'generated_text': 'Prediction 2'}],
# ]
data_preds = json.load(open(f'./examples/prediction/gen-{subset}-{strategy}.json', 'r'))

# def calc(prediction, label):
#     # Calculate ROUGE scores
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(label, prediction)
#     # Calculate BERT_score
#     bert_scores_pair = bert_score([prediction], [label], lang='en')
#     return [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure, bert_scores_pair[2].mean().item()]


from bert_score import BERTScorer
from rouge_score import rouge_scorer

class TextScorer:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang='en')

    def calculate_scores(self, prediction, label):
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(label, prediction)

        # Calculate BERT_score
        bert_scores_pair = self.bert_scorer.score([prediction], [label])

        return [
            rouge_scores['rouge1'].fmeasure,
            rouge_scores['rouge2'].fmeasure,
            rouge_scores['rougeL'].fmeasure,
            bert_scores_pair[2].mean().item()
        ]


pii_scorer = PIIScorer()
results = list()
text_scorer = TextScorer()

# Iterate through the label-prediction pairs
for orig_label, cleaned_label, prediction_dict in tqdm.tqdm(zip(data_orig_labels, data_cleaned_labels, data_preds)):
    pred_text = prediction_dict[0]['generated_text']

    if strategy == 'instruct' or strategy == 'contrast':
        if strategy == 'instruct':
            hint1 = "(1) a response answering the question: "
            hint2 = "(2) a privacy protection version of the response: "
            plain = pred_text.split(hint1)[-1].split(hint2)[0].strip()
            protected = pred_text.split(hint2)[-1].strip()
        elif strategy == 'contrast':
            hint1 = "(1) desired answer to the question: "
            hint2 = "(2) undesired answer to the question: "
            protected = pred_text.split(hint1)[-1].split(hint2)[0].strip()
            plain = pred_text.split(hint2)[-1].strip()
        if plain == "" or protected == "" or orig_label == "" or cleaned_label == "":
            print(plain, protected, orig_label, cleaned_label)

        # plain_scores = calc(plain, orig_label)
        # protected_scores = calc(protected, orig_label)
        plain_orig_scores = text_scorer.calculate_scores(plain, orig_label)
        protected_orig_scores = text_scorer.calculate_scores(protected, orig_label)

        plain_cleaned_scores = text_scorer.calculate_scores(plain, cleaned_label)
        protected_cleaned_scores = text_scorer.calculate_scores(protected, cleaned_label)
        
        priv_plain_score = score = pii_scorer.score_text(plain)
        priv_protected_score = pii_scorer.score_text(protected)
        priv_pred_score = pii_scorer.score_text(pred_text)

        results.append(
            [{'plain_orig': plain_orig_scores},
            {'protect_orig': protected_orig_scores},
            {'plain_cleaned': plain_cleaned_scores},
            {'protect_cleaned': protected_cleaned_scores},
            {'priv_plain_score': priv_plain_score},
            {'priv_protected_score': priv_protected_score},
            {'priv_pred_score': priv_pred_score}]
        )

    elif strategy == 'original' or strategy == 'mask' or strategy == 'remove' or strategy == 'loss':
        # Extract the response part from predictions
        prediction = prediction_dict[0]['generated_text'].split("### Response:\n")[1].strip()

        orig_scores = text_scorer.calculate_scores(prediction, orig_label)
        cleaned_scores = text_scorer.calculate_scores(prediction, cleaned_label)
        priv_score = score = pii_scorer.score_text(prediction)

        results.append(
            [{'plain_orig': orig_scores},
            {'plain_cleaned': cleaned_scores},
            {'priv_score': priv_score}]
        )
    else:
        raise NotImplementedError


json.dump(results, open(f'examples/results/{subset}-{strategy}-res.json', 'w'))
