import os
import argparse

from utils.data_utils import *
from utils.eval_utils import *

from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification
    )
from transformers.trainer_callback import EarlyStoppingCallback
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

if __name__ == '__main__':
    os.environ["WANDB_PROJECT"]="vampire_hunter"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=True)
    parser.add_argument('--data_dir',
                        type=str,
                        required=True)
    parser.add_argument('--max_length',
                        type=int,
                        default=256
                        )
    
    parser.add_argument('--do_lower_case',
                        action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=3e-5)
    parser.add_argument('--epochs',
                        type=int,
                        default=5)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=1000)
    parser.add_argument('--save_total_limit',
                        type=int,
                        default=3)
    parser.add_argument('--evaluation_strategy',
                        type=str,
                        default='epoch')
    parser.add_argument('--logging_strategy',
                        type=str,
                        default='epoch')
    parser.add_argument('--save_strategy',
                        type=str,
                        default='epoch')
    
    parser.add_argument('--metric_for_best_model',
                    type=str,
                    default='eval_f1')
    
    args = parser.parse_args()
    
    experiment_name = args.model_name.split('/')[-1]

    model_dir = f'./experiments/{experiment_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)
    
    dataset_dict = process(args.data_dir, tokenizer, args.max_length)
    
    load_best_model_at_end = True
    
    output_dir = model_dir + '/results'
    # get best model through a metric
    metric_for_best_model = args.metric_for_best_model
    if metric_for_best_model == 'eval_loss':
        greater_is_better = False
    else:
        metric_for_best_model = 'eval_f1'
        greater_is_better = True
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy=args.evaluation_strategy,
        logging_strategy=args.logging_strategy,
        save_strategy=args.save_strategy,
        save_total_limit = args.save_total_limit + 2,
        log_level="error",
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better,
        load_best_model_at_end=True,
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    early_stopping_patience = args.save_total_limit

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    dev_preds, dev_labels, dev_results = predict(trainer, dataset_dict['validation'])
    dev_results.to_csv(os.path.join(model_dir, 'dev_results.csv'), index=False)
    
    dev_report_1 = classification_report(dev_labels, dev_preds, zero_division=0, output_dict=True)
    dev_report_2 = classification_report(dev_labels, dev_preds, zero_division=0, output_dict=True, scheme=IOB2)
    
    pd.DataFrame(dev_report_1).T.to_csv(model_dir + '/report_dev_IOB1.csv')
    pd.DataFrame(dev_report_2).T.to_csv(model_dir + '/report_dev_IOB2.csv')

    test_preds, test_labels, test_results = predict(trainer, dataset_dict['test'], inference=False)
    test_results.to_csv(os.path.join(model_dir, 'test_results.csv'), index=False)
    
    test_report_1 = classification_report(test_labels, test_preds, zero_division=0, output_dict=True)
    test_report_2 = classification_report(test_labels, test_preds, zero_division=0, output_dict=True, scheme=IOB2)

    pd.DataFrame(test_report_1).T.to_csv(model_dir + '/report_test_IOB1.csv')
    pd.DataFrame(test_report_2).T.to_csv(model_dir + '/report_test_IOB2.csv')
    
