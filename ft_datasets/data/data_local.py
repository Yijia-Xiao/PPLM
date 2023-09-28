from datasets import load_from_disk, load_dataset


def load_save(dataset_name):
    data = load_from_disk(f'pii-{dataset_name}.hf')
    return data


if __name__ == "__main__":
    load_save('medical_flashcards')
    load_save('wikidoc')
    data = load_save('wikidoc_patient_information')
    # print(data.to_list()[:10])