import os
import argparse
import logging

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
    
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)

    experiment_name = args.model_name.split('/')[-1]

    logger = logging.Logger(__name__)
    model_dir = f'./experiments/{experiment_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)
    
    dataset_dict = process(args.data_dir, tokenizer, args.max_length)
    
    total_steps_epoch = len(dataset_dict['train']) // (args.batch_size * args.gradient_accumulation_steps)
    logging_steps = total_steps_epoch
    eval_steps = logging_steps
    save_steps = logging_steps
    load_best_model_at_end = True
    # folder_model = 'e' + str(args.epochs) + '_lr' + str(args.learning_rate)
    output_dir = model_dir + '/results'
    # get best model through a metric
    metric_for_best_model = 'eval_f1'
    if metric_for_best_model == 'eval_f1':
        greater_is_better = True
    elif metric_for_best_model == 'eval_loss':
        greater_is_better = False
        
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy='epoch',
        log_level="error",
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better,
        load_best_model_at_end=True,
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
        disable_tqdm=False,
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
    
    logger.info("***** Dev results IOB1 *****")
    logger.info(classification_report(dev_labels, dev_preds))
    
    logger.info("***** Dev results IOB2 *****")
    logger.info(classification_report(dev_labels, dev_preds, scheme=IOB2))

    test_preds, test_labels, test_results = predict(trainer, dataset_dict['test'], inference=False)
    test_results.to_csv(os.path.join(model_dir, 'test_results.csv'), index=False)
    
    logger.info("***** Test results IOB1 *****")
    logger.info(classification_report(test_labels, test_preds))
    
    logger.info("***** Test results IOB2 *****")
    logger.info(classification_report(test_labels, test_preds, scheme=IOB2))

