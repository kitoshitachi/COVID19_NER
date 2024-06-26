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

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list), ignore_mismatched_sizes=args.ignore_mismatched_sizes)
    
    dataset_dict = process(args.data_dir, tokenizer, args.max_length, args.use_fast)
    
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
        run_name=experiment_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy='no',
        log_level="error",
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better,
        load_best_model_at_end=True,
        # push_to_hub=True,
        disable_tqdm=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    
    dev_report, dev_pred = predict(trainer, dataset_dict['validation'])
    dev_report_df = pd.DataFrame.from_dict(dev_report, orient='index').T
    
    print("\n=========== DEV REPORT =============")
    print(dev_report_df)
    
    dev_report_df.to_csv(model_dir + '/report_dev.csv')
    pd.DataFrame(dev_pred).to_csv(model_dir + '/dev_predictions.csv')
    
    test_report, test_pred = predict(trainer, dataset_dict['test'], inference=False)
    test_report_df = pd.DataFrame.from_dict(test_report, orient='index').T
    
    print("\n=========== TEST REPORT =============")
    print(test_report_df)
    
    test_report_df.to_csv(model_dir + '/report_test_IOB1.csv')
    pd.DataFrame(test_pred).to_csv(model_dir + '/test_predictions.csv')
    
    
