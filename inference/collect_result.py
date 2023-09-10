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

    print(results)



files = "medical_flashcards-contrast-res.json  medical_flashcards-mask-res.json      wikidoc-contrast-res.json  wikidoc-mask-res.json  medical_flashcards-instruct-res.json  medical_flashcards-original-res.json  wikidoc-instruct-res.json  wikidoc-original-res.json"
for subset in ["medical_flashcards", "wikidoc"]:
    for stgy in ["original", "mask", "remove", "loss", "instruct", "contrast"]:
        f = f"./examples/results/{subset}-{stgy}-res.json"
        data = json.load(open(f, 'r'))
        print(f'SUBSET={subset}. STRATEGY={stgy}.')
        analyze(data)



# SUBSET=medical_flashcards. STRATEGY=original.
# {'plain_orig': [0.4634, 0.3109, 0.394, 0.9002], 'plain_cleaned': [0.4488, 0.2934, 0.3806, 0.8903], 'priv_score': [0.0224]}
# SUBSET=medical_flashcards. STRATEGY=mask.
# {'plain_orig': [0.4509, 0.2926, 0.3781, 0.8949], 'plain_cleaned': [0.4409, 0.2816, 0.3689, 0.8867], 'priv_score': [0.0138]}
# SUBSET=medical_flashcards. STRATEGY=remove.
# {'plain_orig': [0.4554, 0.2997, 0.3864, 0.8967], 'plain_cleaned': [0.4445, 0.2871, 0.3764, 0.8884], 'priv_score': [0.015]}
# SUBSET=medical_flashcards. STRATEGY=loss.
# {'plain_orig': [0.4561, 0.2961, 0.3825, 0.895], 'plain_cleaned': [0.4457, 0.2843, 0.3728, 0.8867], 'priv_score': [0.0147]}
# SUBSET=medical_flashcards. STRATEGY=instruct.
# {'plain_orig': [0.4595, 0.3043, 0.3894, 0.8991], 'protect_orig': [0.4537, 0.2967, 0.3844, 0.8947], 'plain_cleaned': [0.4452, 0.2869, 0.3761, 0.8892], 'protect_cleaned': [0.4481, 0.2897, 0.3782, 0.8911], 'priv_plain_score': [0.0219], 'priv_protected_score': [0.0137], 'priv_pred_score': [0.0089]}
# SUBSET=medical_flashcards. STRATEGY=contrast.
# {'plain_orig': [0.4502, 0.2915, 0.3783, 0.8931], 'protect_orig': [0.4571, 0.3002, 0.3844, 0.8983], 'plain_cleaned': [0.4462, 0.286, 0.3737, 0.8905], 'protect_cleaned': [0.4433, 0.2836, 0.3717, 0.8885], 'priv_plain_score': [0.0123], 'priv_protected_score': [0.0216], 'priv_pred_score': [0.0094]}
# SUBSET=wikidoc. STRATEGY=original.
# {'plain_orig': [0.1694, 0.06, 0.136, 0.8189], 'plain_cleaned': [0.166, 0.057, 0.133, 0.814], 'priv_score': [0.0246]}
# SUBSET=wikidoc. STRATEGY=mask.
# {'plain_orig': [0.1592, 0.0502, 0.1261, 0.8092], 'plain_cleaned': [0.1566, 0.0481, 0.1237, 0.8048], 'priv_score': [0.0143]}
# SUBSET=wikidoc. STRATEGY=remove.
# {'plain_orig': [0.1533, 0.0523, 0.1244, 0.8062], 'plain_cleaned': [0.1511, 0.0507, 0.1223, 0.8021], 'priv_score': [0.0125]}
# SUBSET=wikidoc. STRATEGY=loss.
# {'plain_orig': [0.1506, 0.044, 0.1175, 0.8103], 'plain_cleaned': [0.148, 0.0421, 0.1151, 0.8057], 'priv_score': [0.0141]}
# SUBSET=wikidoc. STRATEGY=instruct.
# {'plain_orig': [0.1885, 0.0687, 0.1492, 0.8325], 'protect_orig': [0.1875, 0.0662, 0.147, 0.8294], 'plain_cleaned': [0.1852, 0.0656, 0.1463, 0.8273], 'protect_cleaned': [0.1893, 0.0666, 0.1479, 0.8288], 'priv_plain_score': [0.0234], 'priv_protected_score': [0.0128], 'priv_pred_score': [0.0119]}
# SUBSET=wikidoc. STRATEGY=contrast.
# {'plain_orig': [0.1851, 0.0662, 0.1454, 0.8268], 'protect_orig': [0.1855, 0.0687, 0.1475, 0.8299], 'plain_cleaned': [0.1861, 0.0662, 0.1458, 0.8258], 'protect_cleaned': [0.1821, 0.0657, 0.1444, 0.8246], 'priv_plain_score': [0.0149], 'priv_protected_score': [0.0249], 'priv_pred_score': [0.0146]}


# calculate improvement
ps = [[0.0224, 0.0138, 0.015, 0.0147, 0.0137, 0.0216], [0.0246, 0.0143, 0.0125, 0.0141, 0.0128, 0.0249]]

def calc(priv_list):
    base = priv_list[0]
    res = []
    for v in priv_list:
        res.append((base - v) / base)
    print(res)


# # Raw results (no rounding)
# medical_flashcards-original
# {'plain_orig': [0.46338614064762523, 0.31094237010662557, 0.39395140420611247, 0.900248847641475], 'plain_cleaned': [0.4488317325403389, 0.2933741993640899, 0.3805676148355229, 0.8903357689994712], 'priv_score': [0.02239662977113084]}
# medical_flashcards-mask
# {'plain_orig': [0.45089431500162735, 0.292632493650616, 0.37806424778760184, 0.894873458331924], 'plain_cleaned': [0.4409428552455009, 0.28155216537119276, 0.3688883195073261, 0.8867220868304238], 'priv_score': [0.013795115485043937]}
# medical_flashcards-instruct
# {'plain_orig': [0.4594537811385262, 0.30429631404718627, 0.38941691494574576, 0.8990810696850302], 'protect_orig': [0.4537402129174479, 0.29668162611169674, 0.38438149589210924, 0.8947490159090145], 'plain_cleaned': [0.44518798587626945, 0.28688295244742007, 0.3760711524004603, 0.8892044239082568], 'protect_cleaned': [0.44810895606417145, 0.2897490820360272, 0.378214731211342, 0.8910918475765315], 'priv_score': [0.008912418527863869]}
# medical_flashcards-contrast
# {'plain_orig': [0.4502292861010036, 0.2914564545481899, 0.37833422837332054, 0.893135126111177], 'protect_orig': [0.4570905483043596, 0.3002190793552675, 0.3844432182709263, 0.8982696838597948], 'plain_cleaned': [0.44620506707051116, 0.2860327215683317, 0.3737309739355444, 0.8905436214883046], 'protect_cleaned': [0.4433262609607586, 0.28364621502810206, 0.3716824301191086, 0.888509163668823], 'priv_score': [0.009394893273500097]}
# wikidoc-original
# {'plain_orig': [0.16940261993012606, 0.05995692304774608, 0.13597892357718616, 0.8189341444969177], 'plain_cleaned': [0.16597117589674545, 0.05703352908768083, 0.13301001437662055, 0.8139707122246425], 'priv_score': [0.02462893149240902]}
# wikidoc-mask
# {'plain_orig': [0.15924341405974005, 0.05020522821410434, 0.12607896398500917, 0.8091843843460083], 'plain_cleaned': [0.1566331331932689, 0.048112036530771236, 0.12374964219183984, 0.8047962417999903], 'priv_score': [0.014294761656674749]}
# wikidoc-instruct
# {'plain_orig': [0.18850286683630418, 0.0687431334582267, 0.14919772132953033, 0.8325495663483937], 'protect_orig': [0.18751134481757523, 0.0662204873251195, 0.14697563666913, 0.8294088853597641], 'plain_cleaned': [0.18524453118700346, 0.0656177619141829, 0.1462504792481038, 0.8272934358914693], 'protect_cleaned': [0.1892736451286147, 0.06660268189452784, 0.1478960905535141, 0.8287874371608098], 'priv_score': [0.01187747477352425]}
# wikidoc-contrast
# {'plain_orig': [0.18513464524250275, 0.06616439757492919, 0.145377941586065, 0.8267700806856155], 'protect_orig': [0.1855237820290122, 0.0687174324267564, 0.14748579259779876, 0.8299243499437968], 'plain_cleaned': [0.18613736294155095, 0.06623880553490058, 0.14579092199074195, 0.8257965823809306], 'protect_cleaned': [0.18208704820723673, 0.06573212776812627, 0.1444423405835759, 0.824634367664655], 'priv_score': [0.014557351725357118]}

