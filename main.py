import os
import argparse
from loguru import logger
from loader import BilingualDataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def count_parameters(model):
    p = sum(p.numel() for p in model.parameters())
    p_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return { "p": p, "p_grad": p_grad }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reinvent Vietnamese Translation")
    parser.add_argument("-c", "--cmd", type=str, default="train",
            help="Command action")
    parser.add_argument("-f", "--config", type=str, default="configs/default.yaml",
            help="Path to config file.")
    args = parser.parse_args()

    logger.info(args)
    if args.cmd == "serialize":
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
        dataset = BilingualDataset(src_file="data/dev/en.txt", tgt_file="data/dev/vi.txt", tokenizer=tokenizer)
        dataset.serialize("data/dev/dumped/")
    elif args.cmd == "train":
        dataset = BilingualDataset()
        dataset.load(src="data/dev/dumped/en.vitrans", tgt="data/dev/dumped/vi.vitrans")
        
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

        logger.info(f"Loaded model with pre-trained params: {count_parameters(model)}")
    

