#!/bin/bash

# Enable debugging mode to echo all commands
set -x

SIZE=7B
EP=10
P=64
ML=512

# mediqa usmle_self_assessment mmmlu medical_flashcards wikidoc_patient_information wikidoc pubmed_causal medqa health_advice cord19;
for SUBSET in medical_flashcards; # wikidoc;
do
    for STGY in contrast instruct;
    do
        TASK=instruct
        LOCA=$SIZE-$STGY-$SUBSET
        echo $LOCA
        mkdir -p ckpt/$LOCA

        (torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $P --micro_batch_size $P \
            --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML --inst_strategy $STGY) 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done


for SUBSET in medical_flashcards; # wikidoc;
do
    for TASK in original; # mask
    do
        LOCA=$SIZE-$TASK-$SUBSET
        echo $LOCA
        mkdir -p ckpt/$LOCA

        (torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $P --micro_batch_size $P \
            --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML) 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done

