import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)  # 设置随机种子
    tokenizer = load_tokenizer(args)  # 加载预训练模型

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")  # 加载数据集
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")  # 加载验证集
    trainer = Trainer(args, train_dataset, dev_dataset)  # 初始化自建类Trainer def __init__(self, args, train_dataset=None, dev_dataset=None):
    # 训练模型
    if args.do_train:
        trainer.train()
    # 验证模型
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./", type=str, help="The input data dir")
    parser.add_argument("--seq_label_file", default="vocab_bio.txt", type=str, help="BIO Label file")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--seed', type=int, default=6, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")  # 最长的样本长度
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.") #在执行后向/更新传递之前，要积累的更新步骤的数量
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")  #设置要执行的训练步骤总数。覆盖num_train_epochs
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--seq_loss_coef', type=float, default=1.0, help='Coefficient for the seq loss.')

    # CRF
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--seq_pad_label", default="PAD", type=str,
                        help="Pad token for seq label pad (to be ignore when calculate loss)")

    _args = parser.parse_args()
    _args.model_name_or_path = MODEL_PATH_MAP[_args.model_type]
    print(_args.model_name_or_path)

    main(_args)
