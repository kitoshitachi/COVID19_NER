import numpy as np
from transformers import Trainer
from seqeval.metrics import classification_report

type_entities = ['PATIENT_ID', 'NAME', 'GENDER', 'AGE', 'JOB', 'LOCATION',
                 'ORGANIZATION', 'DATE', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']

label_list = ['B-' + tag for tag in type_entities] + ['I-' + tag for tag in type_entities] + ['O']


def _ready_for_metrics(predictions, labels):
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] if l != -100 else 'O' for (p, l) in zip(prediction, label) ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] if l != -100 else 'O' for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]
    
    return true_predictions, true_labels

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_labels, true_predictions = _ready_for_metrics(predictions, labels)
    
    report = classification_report(true_labels, true_predictions, digits = 4, zero_division=0, output_dict=True)
    result = {entity: report[entity]['f1-score'] for entity in type_entities}
    result['f1_macro'] = report['macro avg']['f1-score']
    result['f1_micro'] = report['micro avg']['f1-score']
    
    return result

def predict(trainer:Trainer, ds, inference=False):
    logits, labels, _ = trainer.predict(ds)
    predictions = np.argmax(logits, axis=2)
    
    true_labels, true_predictions = _ready_for_metrics(predictions, labels)
    
    report = classification_report(true_labels, true_predictions, digits = 4, zero_division=0)
    return true_predictions, report


import pandas as pd
from transformers import TrainerCallback
class PandasEvaluationCallback(TrainerCallback):
    def __init__(self):
        self.results_df = pd.DataFrame()  # Initialize empty DataFrame

    def on_evaluate(self, args, state, control, metrics:dict[str,float], **kwargs):
        if state.is_world_process_zero:
            # Create a DataFrame for the current evaluation results
            metrics_df = pd.DataFrame(metrics, index=[0])
            metrics_df.columns = metrics_df.columns.str.replace("eval_", "", regex=False)
            metrics_df = metrics_df.round(4)
            
            # Add current epoch and step (if applicable)
            metrics_df["epoch"] = state.epoch 
            # Append the current results to the overall DataFrame
            self.results_df = pd.concat([self.results_df, metrics_df], ignore_index=True)

            # Display the updated DataFrame
            print(f"Evaluation Results:")
            print(self.results_df.set_index('epoch')[['loss', 'f1_macro', 'f1_micro'] + type_entities])