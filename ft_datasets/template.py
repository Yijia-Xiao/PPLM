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
    "question_answer": (
        "### Question:\n{input}\n\n### Answer:\n"
    ),
    "instruct_tuning_command": (
        "Below is a question and related responses. "
        # "Write a privacy protection response to the question. Sensitive information in the response should be replaced by <unk>. \n\n"
        "Write a privacy protection response to the question. Sensitive information in the response should be replaced by NAME or ORGANIZATION. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_command": (
        "{cleaned_output}"
    ),
    "instruct_tuning_instruct": (
        "Below is a question and related responses. Sensitive information in the response should be replaced by NAME or ORGANIZATION."
        "Write \n(1) a response answering the question. \n(2) a privacy protection version of the response. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_instruct": (
        "(1) a response answering the question: {output}\n(2) a privacy protection version of the response: {cleaned_output}\n"
    ),
    # "instruct_tuning_contrast": (
    #     "Below is a question and related responses: a desired one and an undesired one. The undesired and desired examples will not be given during inference.\n\n"
    #     "### Instruction:\n{instruction}.\n\n### Input:\n{input}\n"
    #     "(1) Undesired one: {output} \n(2) Desired one: {cleaned_output} \n\n### Response:\n"
    # ),
    # "instruct_output_contrast": (
    #     "{output}"
    # ),

    # Above version deprecated.
    # Aligning prompts between training and evaluation of contrast strategy.
    "instruct_tuning_contrast": (
        "Below is a question and related responses. Sensitive information in the response should be replaced by NAME or ORGANIZATION."
        "Write \n(1) desired answer to the question. \n(2) undesired answer to the question. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_contrast": (
        "(1) desired answer to the question: {output}\n(2) undesired answer to the question: {cleaned_output}\n"
    ),
    "instruct_tuning_instruct_rev": (
        "Below is a question and related responses. Sensitive information in the response should be replaced by NAME or ORGANIZATION."
        "Write \n(1) a privacy protection version of the response. \n(2) a response answering the question. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_instruct_rev": (
        "(1) a privacy protection version of the response: {output}\n(2) a response answering the question: {cleaned_output}\n"
    ),
    "instruct_tuning_contrast_rev": (
        "Below is a question and related responses. Sensitive information in the response should be replaced by NAME or ORGANIZATION."
        "Write \n(1) undesired answer to the question. \n(2) desired answer to the question. \n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "instruct_output_contrast_rev": (
        "(1) undesired answer to the question: {output}\n(2) desired answer to the question: {cleaned_output}\n"
    ),
    "dpo": (
        "{input}\n\nAnswer: "
    ),
}