import os
import pandas as pd

import datasets
from datasets import Dataset, DatasetDict

def tokenize_and_align_labels(dataset_unaligned, tokenizer, max_length, label_all_tokens=False, use_fast = True):
    tokenized_inputs = tokenizer(dataset_unaligned["tokens"], truncation=True, is_split_into_words=True, max_length=max_length)
    labels = []
    for i, label in enumerate(dataset_unaligned[f"ner_tags"]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i) if use_fast else tokenizer.convert_tokens_to_ids(tokenized_inputs.tokens(batch_index=i))
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

def tokenize_fn(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=max_length)
    pseudo_labels = []
    for i, _ in enumerate(examples['tokens']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(-1)

            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        pseudo_labels.append(label_ids)
    tokenized_inputs["pseudo_labels"] = pseudo_labels
    return tokenized_inputs

def sent_process(data_dir, tokenizer):
    raise NotImplementedError

def process(data_dir, tokenizer, max_length, use_fast):

    train_df = pd.read_json(os.path.join(data_dir, f'train_word.json'), orient='records', lines=True).reset_index()
    dev_df = pd.read_json(os.path.join(data_dir, f'dev_word.json'), orient='records', lines=True).reset_index()
    test_df = pd.read_json(os.path.join(data_dir, f'test_word.json'), orient='records', lines=True).reset_index()

    mapping_column_names = {
        'index': 'id',
        'words': 'tokens',
        'tags': 'ner_tags'
    }
    train_df.rename(columns=mapping_column_names, inplace=True)
    dev_df.rename(columns=mapping_column_names, inplace=True)
    test_df.rename(columns=mapping_column_names, inplace=True)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(dev_df),
        'test': Dataset.from_pandas(test_df)
    })

    label_list = sorted(list(set(tag for doc in train_df['ner_tags'] for tag in doc)))

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

    tokenized_datasets = dataset.map(tokenize_and_align_labels, 
                fn_kwargs={'tokenizer': tokenizer, 'max_length':max_length, 'use_fast':use_fast}, batched=True)

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True, token='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')
    print(process('D:/19521204\python\COVID19_NER\data\COVID19', tokenizer, 256))
