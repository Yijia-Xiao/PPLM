# set -v # exports all options, not only `verbose'
# export SHELLOPTS

SIZE=7B
EP=10
P=64
SUBSET=medical_flashcards
# mediqa usmle_self_assessment mmmlu medical_flashcards wikidoc_patient_information wikidoc pubmed_causal medqa health_advice cord19;

for TASK in instruct; # mask original;
do
    LOCA=$SIZE-$TASK-$SUBSET
    mkdir -p ckpt/$LOCA

    (torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py \
        --enable_fsdp --use_peft --peft_method lora --model_name models_hf/$SIZE \
        --pure_bf16 --output_dir ckpt/$LOCA/ \
        --num_epochs $EP --batch_size_training $P --micro_batch_size $P \
        --num_workers_dataloader 64 --use_fast_kernels --dataset ${TASK}_dataset --subset $SUBSET) 2>&1 | tee ckpt/$LOCA/train.log

    python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/$LOCA/ --output_dir ckpt/merge/$LOCA/
done
