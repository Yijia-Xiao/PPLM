# set -v # exports all options, not only `verbose'
# export SHELLOPTS

SIZE=7B
EP=5

# flashcards wikidoc
P=64
ML=512

# cord19
P=128
ML=256

# mediqa usmle_self_assessment mmmlu medical_flashcards wikidoc_patient_information wikidoc pubmed_causal medqa health_advice cord19;
for SUBSET in cord; # medical_flashcards wikidoc;
do
    for TASK in original mask; # instruct
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



: << instruction related
SIZE=7B
EP=5

# flashcards wikidoc
P=64
ML=512

# CORD19
P=128
ML=256

STGY=contrast

# mediqa usmle_self_assessment mmmlu medical_flashcards wikidoc_patient_information wikidoc pubmed_causal medqa health_advice cord19;
for SUBSET in cord; # medical_flashcards wikidoc;
do
    for TASK in mask original; # instruct
    do
        LOCA=$SIZE-$TASK-$SUBSET-$STGY
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
instruction related
