import pandas as pd
import numpy as np
import evaluate

metric = evaluate.load('seqeval', trust_remote_code=True)
type_entities = ['PATIENT_ID', 'NAME', 'GENDER', 'AGE', 'JOB', 'LOCATION',
                 'ORGANIZATION', 'DATE', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']

label_list = ['O'] + ['B-' + tag for tag in type_entities] + ['I-' + tag for tag in type_entities]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def predict(trainer, ds, inference=False):
    logits, labels, _ = trainer.predict(ds)
    predictions = np.argmax(logits, axis=2)

    if inference:
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, pseudo_label) if l != -100]
            for prediction, pseudo_label in zip(predictions, ds['pseudo_labels'])
        ]
        return true_predictions
    else:
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return true_predictions, true_labels, pd.DataFrame(results)