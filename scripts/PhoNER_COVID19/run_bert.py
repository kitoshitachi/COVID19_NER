import os
from utils.arguments import config
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


if __name__ == '__main__':
    args = config().parse_args()
    
    experiment_name = args.model_name.split('/')[-1]

    model_dir = f'./experiments/{experiment_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case, use_fast=args.use_fast, add_prefix_space=args.add_prefix_space)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)
    
    dataset_dict = process(args.data_dir, tokenizer, args.max_length, args.use_fast)
    
    load_best_model_at_end = True
    
    output_dir = model_dir + '/results'
    # get best model through a metric
    metric_for_best_model = args.metric_for_best_model
    if metric_for_best_model == 'eval_loss':
        greater_is_better = False
    else:
        metric_for_best_model = 'eval_f1_micro'
        greater_is_better = True
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        run_name=experiment_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        logging_strategy='no',
        save_strategy='epoch',
        log_level="error",
        metric_for_best_model = 'eval_f1_micro',
        greater_is_better = True,
        load_best_model_at_end=True,
        gradient_checkpointing=False,
        # do_train=True,
        # do_eval=True,
        
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience), PandasEvaluationCallback()],
    )

    trainer.train()
    
    dev_preds, dev_report = predict(trainer, dataset_dict['validation'])
    # dataset_dict['validation']['predictions'] = dev_preds
    test_preds, test_report = predict(trainer, dataset_dict['test'], inference=False)
    
    print("=========== DEV REPORT 1 =============")
    print(dev_report)
    print("=========== TEST REPORT 1 =============")
    print(test_report)
    
    with open(model_dir + '/report.csv', 'w+', encoding='utf-8') as f:
        f.write("=========== DEV REPORT 1 =============\n\n"+ 
                dev_report +  
                "\n\n=========== TEST REPORT 1 =============\n\n" +
                test_report
        )

    with open(model_dir + '/predictions.csv', 'w+', encoding='utf-8') as f:
        for pred in test_preds:
            f.write(" ".join(pred) + '\n')
    
