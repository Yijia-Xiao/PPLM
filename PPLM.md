# PPLM README

## Fintetuning the model
See `run.sh` script.

## Genrarte evaluation set

Run `ft_datasets/example_sampler.py` script. Pass `subset` parameter to `Generator` to initialize; then pass `task` argument to `generate()` function to generate dataset (datafileds filling into templeates, ready for feeding into model for inference) and save as `JSON` file. 

Example:
```
generator = Generator(subset='medical_flashcards')
generator.generate(task='original')
```

## Run inference
See `infer.sh` and `inference/pl.py` scripts.

## Run evaluation
See `inference/eval.sh` and `inference/eval.py`.

## Collect results
Run `inference/collect_result.py` to collect results output by `eval.py`.
