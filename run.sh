#!/bin/bash

# Enable debugging mode to echo all commands
set -x

SIZE=7B
EP=5
BS=64
ML=384

for SUBSET in wikidoc_patient_information medical_flashcards wikidoc;
do
    if [ "$SUBSET" = "cord19" ]; then
        EP=3
    fi

    for STGY in contrast instruct;
    do
        TASK=instruct
        LOCA=$STGY-$SUBSET-$SIZE-$ML-$BS
        echo $LOCA
        mkdir -p ckpt/$LOCA

        torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $BS --micro_batch_size $BS \
            --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML --inst_strategy $STGY 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done

for SUBSET in wikidoc_patient_information medical_flashcards wikidoc;
do
    if [ "$SUBSET" = "cord19" ]; then
        EP=3
    fi

    for TASK in original mask;
    do
        LOCA=$TASK-$SUBSET-$SIZE-$ML-$BS
        echo $LOCA
        mkdir -p ckpt/$LOCA

        torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $BS --micro_batch_size $BS \
            --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done
experiment_1


for SUBSET in medical_flashcards; # wikidoc;
do
    for TASK in remove;
    do
        LOCA=$TASK-$SUBSET-$SIZE-$ML-$BS
        echo $LOCA
        mkdir -p ckpt/$LOCA

        torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $BS --micro_batch_size $BS \
            --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET --maxlen $ML 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done


<< loss
for SUBSET in medical_flashcards wikidoc;
do
    for TASK in loss;
    do
        LOCA=$TASK-$SUBSET-$SIZE-$ML-$BS
        echo $LOCA
        mkdir -p ckpt/$LOCA

        torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
            --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
            --pure_bf16 --output_dir ckpt/$LOCA/ \
            --num_epochs $EP --batch_size_training $BS --micro_batch_size $BS \
            --num_workers_dataloader 64 --use_fast_kernels --dataset mask_dataset --subset $SUBSET --maxlen $ML --custom_loss penalty 2>&1 | tee -a ckpt/$LOCA/train.log

        python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
    done
done
loss


set -x
for D in "$@";
do
    for TPLT in command instruct contrast instruct_rev contrast_rev; # dpo; # qa; # original mask remove command instruct contrast instruct_rev contrast_rev;
    do
        for SCLE in 7B 13B;
        do
            python inference/pl.py --dataset $D --template $TPLT --scale $SCLE
        done
    done
done
