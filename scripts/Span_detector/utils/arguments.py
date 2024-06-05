import argparse

def config():
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
                        default=1)
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
    
    parser.add_argument('--use_fast',
                action='store_false')
    
    parser.add_argument('--add_prefix_space',
            action='store_true')
    parser.add_argument('--ignore_mismatched_sizes',
        action='store_true')
    return parser