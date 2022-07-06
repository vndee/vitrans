import os
import torch
from typing import Optional
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, src_file: Optional[str] = None, tgt_file: Optional[str] = None, tokenizer = None, max_len: int = 256):
        super(BilingualDataset, self).__init__()
        
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source = None
        self.target = None
    
    @staticmethod
    def zero_pad(x: torch.Tensor, max_len: int):
        return torch.nn.functional.pad(x, (0, max_len - x.shape[1]), "constant", 0)

    def dump(self, fp, out_dir):
        with open(fp, "r") as stream:
            lines = stream.readlines()
            input_ids, attention_masks = torch.zeros((len(lines), self.max_len)), torch.zeros((len(lines), self.max_len))

            for it, line in enumerate(tqdm(lines, desc=f"Serializing {fp}")):
                tokenized = self.tokenizer(line.strip(), return_tensors="pt")
                input_id = BilingualDataset.zero_pad(tokenized["input_ids"], self.max_len)
                attention_mask = BilingualDataset.zero_pad(tokenized["attention_mask"], self.max_len)
                
                input_ids[it], attention_masks[it] = input_id, attention_mask

        n_fp = os.path.join(out_dir, fp.split("/")[-1].replace(".txt", ".vitrans"))
        torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_masks
            }, n_fp)
        
        logger.info(f"Serialized {fp} to {n_fp}")

    def serialize(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.dump(self.src_file, out_dir)
        self.dump(self.tgt_file, out_dir)

    def load(self, src, tgt):
        assert os.path.exists(src) and os.path.exists(tgt)
        self.source, self.target = torch.load(src), torch.load(tgt)

        assert self.source["input_ids"].shape == self.source["attention_mask"].shape
        assert self.target["input_ids"].shape == self.target["attention_mask"].shape
        assert self.source["input_ids"].shape == self.target["attention_mask"].shape

        logger.info(f"Loaded source data from {src}")
        logger.info(f"Loaded target data from {tgt}")

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

