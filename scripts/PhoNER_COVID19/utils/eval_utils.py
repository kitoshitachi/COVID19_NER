import pandas as pd
import numpy as np
import evaluate
from transformers import Trainer
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

metric = evaluate.load('seqeval', trust_remote_code=True)
type_entities = ['PATIENT_ID', 'NAME', 'GENDER', 'AGE', 'JOB', 'LOCATION',
                 'ORGANIZATION', 'DATE', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']

label_list = ['B-' + tag for tag in type_entities] + ['I-' + tag for tag in type_entities]

def get_report(predictions, labels) -> tuple[dict]:
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    report_1 = classification_report(true_labels, true_predictions, digits = 4, zero_division=0, output_dict=True)
    report_2 = classification_report(true_labels, true_predictions, digits = 4, zero_division=0, output_dict=True, scheme=IOB2)
    return report_1, report_2

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    report_1, report_2 = get_report(predictions, labels)
    return {
        'f1_macro':report_1['macro avg']['f1-score'],
        'f1':report_1['weighted avg']['f1-score'],
        'recall':report_1['weighted avg']['recall'],
        'precision':report_1['weighted avg']['precision'],
    }

def predict(trainer:Trainer, ds, inference=False):
    logits, labels, _ = trainer.predict(ds)
    predictions = np.argmax(logits, axis=2)

    return get_report(predictions, labels)