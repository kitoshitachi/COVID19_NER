import os
import pandas as pd

import datasets
from datasets import Dataset, DatasetDict

def tokenize_and_align_labels_slow(examples, tokenizer, max_length, label_all_tokens=False):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, (tokens, labels) in enumerate(zip(examples["tokens"], examples["ner_tags"])):
        word_ids = []
        label_ids = []
        current_word = 0

        for word, label in zip(tokens, labels):
            word_tokens = tokenizer.tokenize(word)
            tokenized_inputs["input_ids"].append(tokenizer.convert_tokens_to_ids(word_tokens))
            # Account for [CLS] and [SEP] with "- 100"
            word_ids.extend([current_word] * (len(word_tokens)))
            current_word += 1

            # Labels
            if label_all_tokens:
                label_ids.extend([label] * len(word_tokens))
            else:
                label_ids.extend([label] + [-100] * (len(word_tokens) - 1))

        # Add special tokens and truncate sequences to max_length
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokenized_inputs["input_ids"][-1]) > max_length - special_tokens_count:
            tokenized_inputs["input_ids"][-1] = tokenized_inputs["input_ids"][-1][:(max_length - special_tokens_count)]
            word_ids = word_ids[:(max_length - special_tokens_count)]
            label_ids = label_ids[:(max_length - special_tokens_count)]

        tokenized_inputs["input_ids"][-1] = tokenizer.build_inputs_with_special_tokens(tokenized_inputs["input_ids"][-1])
        tokenized_inputs["attention_mask"].append(
            [1] * len(tokenized_inputs["input_ids"][-1])
        )
        tokenized_inputs["labels"].append(
            [-100] + label_ids + [-100]
        )

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
    
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels if use_fast else tokenize_and_align_labels_slow, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_length':max_length}, 
    )

    return tokenized_datasets


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True, token='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')
    print(process('D:/19521204\python\COVID19_NER\data\COVID19', tokenizer, 256))
