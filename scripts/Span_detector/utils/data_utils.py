import os
import pandas as pd

import datasets
from datasets import Dataset, DatasetDict

label_list = ['B-ENTITY', 'I-ENTITY', 'O']

def tokenize_and_align_labels_slow(dataset_unaligned, tokenizer, max_length, label_all_tokens=False):

    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(dataset_unaligned["tokens"])):
        tokens = dataset_unaligned["tokens"][i]
        labels = dataset_unaligned["ner_tags"][i]

        # Tokenize each word individually (required for slow tokenizers)
        word_tokenized = [tokenizer.tokenize(word) for word in tokens]
        word_ids = [tokenizer.convert_tokens_to_ids(subwords) for subwords in word_tokenized]
        input_ids = [subword_id for word in word_ids for subword_id in word]  # Flatten

        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            word_ids = word_ids[:max_length]  # Truncate word IDs as well

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Align labels
        label_ids = []
        current_word_idx = 0
        for word_idx in word_ids:
            # Special tokens get -100
            if isinstance(word_idx, list): 
                label_ids.extend([-100] * len(word_idx))
            else:
                label_ids.append(labels[current_word_idx])
                if not label_all_tokens:
                    label_ids.extend([-100] * (len(tokenizer.convert_ids_to_tokens(word_idx)) - 1)) 
                current_word_idx += 1

        # Add to the final dictionary
        tokenized_inputs["input_ids"].append(input_ids)
        tokenized_inputs["attention_mask"].append(attention_mask)
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs

def tokenize_and_align_labels(dataset_unaligned, tokenizer, max_length, label_all_tokens=False, use_fast = True):
    tokenized_inputs = tokenizer(dataset_unaligned["tokens"], truncation=True, is_split_into_words=True, max_length=max_length)
    labels = []
    for i, label in enumerate(dataset_unaligned[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: #special tokens
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else: # subwords
                label_ids.append(1 if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def build_ds_stage_1(data):
    return {'ner_tags':[
        [tag.split('-')[0] + '-ENTITY' if tag != 'O' else 'O' for tag in tags] 
        for tags in data
    ]}

def process(data_dir, tokenizer, max_length, use_fast):

    train_df = pd.read_json(os.path.join(data_dir, f'train_word.json'), orient='records', lines=True).reset_index()
    dev_df = pd.read_json(os.path.join(data_dir, f'dev_word.json'), orient='records', lines=True).reset_index()
    test_df = pd.read_json(os.path.join(data_dir, f'test_word.json'), orient='records', lines=True).reset_index()

    mapping_column_names = {
        'index': 'id',
        'words': 'tokens',
        'tags': 'ner_tags'
    }
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(dev_df),
        'test': Dataset.from_pandas(test_df)
    })
    dataset = dataset.rename_columns(mapping_column_names)
    dataset = dataset.map(
        build_ds_stage_1, 
        batched=True, 
        input_columns=['ner_tags'],
    )
    ds_features = datasets.Features(
        {
        'id': datasets.Value('int32'),
        'tokens': datasets.Sequence(datasets.Value('string')),
        'ner_tags': datasets.Sequence(
            datasets.features.ClassLabel(names=label_list)
            )
        }
    )
    
    dataset = dataset.map(ds_features.encode_example, features=ds_features)
    
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels if use_fast else tokenize_and_align_labels_slow, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_length':max_length}, 
    )

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2', do_lower_case=True, use_fast = False)
    print(process('D:/19521204\python\COVID19_NER\data\COVID19', tokenizer, 256, use_fast=False))
