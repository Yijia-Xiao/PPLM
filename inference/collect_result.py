import json


def analyze(data):
    example_result = data[0]
    results = dict()

    for schema_scores_dict in example_result:
        for k, v in schema_scores_dict.items():
            results[k] = [0.0] * len(v) if isinstance(v, list) else [0.0]


    for sample in data:
        for schema_scores_dict in sample:
            # print('dict', schema_scores_dict)
            for schema, scores in schema_scores_dict.items():
                if isinstance(scores, list):
                    for i in range(len(scores)):
                        results[schema][i] += scores[i]
                else:
                    results[schema][0] += scores

    # print(results)
    num_samples = len(data)
    for schema_key in results.keys():
        results[schema_key] = [round(v / num_samples, 4) for v in results[schema_key]]

    res = dict()
    if 'protect_orig' in results:
        for k, v in results.items():
            if k == 'protect_orig' or k == 'protect_cleaned':
                # if k == 'plain_orig':
                if k == 'protect_orig':
                    res['orig'] = v
                else:
                    res['cleaned'] = v
            else:
                if k == 'priv_protected_score':
                # if k == 'priv_plain_score':
                    res['priv_score'] = v

    else:
        for k, v in results.items():
            if k == 'plain_orig' or k == 'plain_cleaned':
                if k == 'plain_orig':
                    res['orig'] = v
                else:
                    res['cleaned'] = v
            else:
                if k == 'priv_score':
                    res['priv_score'] = v

    print(res)
    return res


data_curve = []
for subset in ["wikidoc_patient_information"]:
    for scale in ['7B']:
        # for stgy in ['mask']: # ['mask']: # ["original", "mask", "remove", "qa", "command", "instruct", "contrast", "instruct_rev", "contrast_rev"]: # ["dpo"]:
        for stgy in ['contrast_rev']: # ['mask']: # ["original", "mask", "remove", "qa", "command", "instruct", "contrast", "instruct_rev", "contrast_rev"]: # ["dpo"]:
            for ep in range(30):
                f = f"./examples/results/plot/{subset}-{stgy}-{scale}-{ep}.json"
                data = json.load(open(f, 'r'))
                print(f'{ep}')
                res = analyze(data)
                data_curve.append(res)
            print()

# print(len(data_curve))
json.dump(data_curve, open(f'contrast_rev.json', 'w'))


for subset in ["medical_flashcards", "wikidoc", "wikidoc_patient_information"]:
    for scale in ['7B', '13B']:
        for stgy in ['command', 'dpo']: # ["original", "mask", "remove", "qa", "command", "instruct", "contrast", "instruct_rev", "contrast_rev"]: # ["dpo"]:
            f = f"./examples/results/{subset}-{stgy}-{scale}.json"
            data = json.load(open(f, 'r'))
            print(f'SUBSET={subset}. STRATEGY={stgy}. SCALE={scale}.')
            analyze(data)
        print()
