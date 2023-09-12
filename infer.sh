for D in medical_flashcards wikidoc;
do
    for TPLT in remove loss; # contrast; # original contrast instruct; # mask
    do
        python inference/pl.py --dataset $D --template $TPLT
    done
done

# /home/dsi/yxiao/plm/ckpt/merge/7B-mask-medical_flashcards
# python inference/inference.py /home/dsi/yxiao/plm/models_hf/7B --prompt_file inference/examples/original.txt


# python inference/inference-batch.py --model_name ckpt/merge/7B-original-medical_flashcards/ --prompt_file inference/examples/original.txt
# Yes, clasp knife spasticity is a common presentation of upper motor neuron lesions.


# python inference/inference.py --model_name ckpt/merge/7B-mask-medical_flashcards/ --prompt_file inference/examples/original.txt
# Yes, clasp knife spasticity is a classic presentation of upper motor neuron (UMN) lesions.


# python inference/inference.py --model_name ckpt/merge/7B-instruct-medical_flashcards/ --prompt_file inference/examples/default.txt
# Below is a question and related responses. Write 
# (1) a response answering the question. 
# (2) a privacy protection version of the response. 

# ### Instruction:
# Answer this question truthfully

# ### Input:
# Is clasp knife spasticity a common presentation of upper motor neuron lesions?

# ### Response:

# (1) a response answering the question: No, clasp knife spasticity is not a common presentation of upper motor neuron (UMN) lesions. Clasp knife spasticity is a type of hyperreflexia that involves sudden and exaggerated, jerky movements, often involving the fingers or other distal muscles of the affected limb. It is characterized by a brief, abrupt increase in muscle tone that is followed by a rapid, jerking movement. Clasp knife spasticity is often caused by UMN lesions in the primary motor cortex, but it can also be caused by other types of brain damage or injury. In contrast, upper extremity hyperreflexia may be more common in UMN lesions, but not always in the form of clasp knife spasticity.
# (2) a privacy protection version of the response: No, clasp knife spasticity is not a common presentation of upper motor neuron ({{ORGANIZATION}}) lesions. Clasp knife spasticity is a type of hyperreflexia that involves sudden and exaggerated, jerky movements, often involving the fingers or other distal muscles of the affected limb. It is characterized by a brief, abrupt increase in muscle tone that is followed by a rapid, jerking movement. Clasp knife spasticity is often caused by {{ORGANIZATION}} lesions in the primary motor cortex, but it can also be caused by other types of brain damage or injury. In contrast, upper extremity hyperreflexia may be more common in {{ORGANIZATION}} lesions, but not always in the form of clasp knife spasticity.




# python inference/hf-text-generation-inference/merge_lora_weights.py --base_model models_hf/7B --peft_model ckpt/7B-instruct-medical_flashcards-contrast/ --output_dir ckpt/merge/7B-instruct-medical_flashcards-contrast/

# No, clasp knife spasticity is typically a presentation of lower motor neuron lesions.
# No, clasp knife spasticity is not a common presentation of upper motor neuron lesions.
