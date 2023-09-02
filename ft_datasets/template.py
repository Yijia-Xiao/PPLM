PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "instruct_tuning_default": (
        "Below is a question and related responses. "
        "Write \n(1) a response answering the question. \n(2) a privecy protection version of the response. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_default": (
        "(1) a response answering the question: {output}\n(2) a privecy protection version of the response: {cleaned_output}\n"
    ),
    "instruct_tuning_contrast": (
        "Below is a question and related responses: a desired one and an undesired one. The undesired and desired examples will not be given during inference.\n\n"
        "### Instruction:\n{instruction}.\n\n### Input:\n{input}\n"
        "(1) Undesired one: {output} \n(2) Desired one: {cleaned_output} \n\n### Response:\n"
    ),
    "instruct_output_contrast": (
        "{output}"
    ),
}