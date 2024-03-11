import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BartForConditionalGeneration, AutoTokenizer

from src.data import TPG_LRFT_DataLoader
from src.module import KeyGenModule
from src.utils import get_logger


parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="facebook/bart-base", help="model file path")
g.add_argument("--train-path", type=str, required=True, help="train dataset file path")
g.add_argument("--valid-path", type=str, required=True, help="valid dataset file path")
g.add_argument("--stage", type=str, required=True, default="TPG", help="TPG or LRFT")
g.add_argument("--order", type=str, default="pres_abs", help="pres_abs, abs_pres, random")
g.add_argument("--tokenizer", type=str, default="facebook/bart-base", help="huggingface tokenizer path")
g.add_argument("--gpus", nargs='+', type=int, required=True, help="the number of gpus")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.1, help="warmup step rate")
g.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
g.add_argument("--batch-size-train", type=int, required=True, help="training batch size")
g.add_argument("--batch-size-valid", type=int, required=True, help="validation batch size")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=float, default=1.0, help="validation interval")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--seed", type=int, default=42, help="random seed")


def main(args):
    logger = get_logger("train")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    special_tokens_dict = {'additional_special_tokens': ['[sep]', '[digit]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    logger.info(f'[+] Load Dataset')
    train_dataloader = TPG_LRFT_DataLoader(args.stage, args.train_path, tokenizer, args.batch_size_train, args.max_seq_len, args.order)
    valid_dataloader = TPG_LRFT_DataLoader(args.stage, args.valid_path, tokenizer, args.batch_size_valid, args.max_seq_len, args.order)
    total_steps = len(train_dataloader) * args.epochs // len(args.gpus)

    logger.info(f'[+] Load Model from "{args.model_path}"')
    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"[+] Load Pytorch Lightning Module")
    lightning_module = KeyGenModule(
        model,
        args.output_dir,
        total_steps,
        args.max_learning_rate,
        args.min_learning_rate,
        args.warmup_rate,
    )

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    
    # pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch.
    # pass an int to check after a fixed number of training batches.
    if args.evaluate_interval == 1:
        args.evaluate_interval = 1.0
    trainer = pl.Trainer(
        strategy="auto",
        accelerator="gpu",
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        devices=args.gpus,
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    exit(main(parser.parse_args()))
