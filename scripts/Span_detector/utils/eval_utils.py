import numpy as np
from tqdm.auto import tqdm
from transformers import Trainer
from seqeval.metrics.sequence_labeling import get_entities


label_list = ['B-ENTITY', 'I-ENTITY', 'O']

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
    
    
    true_predictions = [
        [f'{entity}#{start}#{end}' for entity, start, end in get_entities(predictions)] 
        for predictions in true_predictions
    ]
    true_labels = [
        [f'{entity}#{start}#{end}' for entity, start, end in get_entities(labels)] 
        for labels in true_labels
    ]
    return true_predictions, true_labels

def _compute_metrics(preds, golds) -> tuple[dict]:
    tp = fp = fn = .0
    n_total = 0

    for pred, gold in zip(preds, golds):
        gold_set = set(gold)
        pred_set = set(pred)
        
        n_total += len(gold_set)
        tp += len(gold_set & pred_set)  # True Positives
        fn += len(gold_set - pred_set)  # False Negatives
        fp += len(pred_set - gold_set)  # False Positives

    precision = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    recall = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    acc = tp / n_total
    return {'precision': precision, 'recall': recall, 'f1': f1, "acc": acc}

def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions, true_labels = _ready_for_metrics(predictions, labels)
    
    return _compute_metrics(true_predictions, true_labels)

def predict(trainer:Trainer, ds, inference=False):
    
    logits, labels, _ = trainer.predict(ds)
    predictions = np.argmax(logits, axis=2)
    
    true_predictions, true_labels = _ready_for_metrics(predictions, labels)
    
    return _compute_metrics(true_predictions, true_labels), true_predictions

from transformers import ProgressCallback
class ProgressOverider(ProgressCallback):
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and eval_dataloader is not None and hasattr(eval_dataloader, '__len__'):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), dynamic_ncols=True)
            self.prediction_bar.update(1)
    
