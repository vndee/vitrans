import os
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_EN2VI = "output/en2vi/"
MODEL_VI2EN = "output/vi2en/"


if __name__ == "__main__":
    MODEL_DIR = MODEL_EN2VI
    tknz = AutoTokenizer.from_pretrained(MODEL_DIR)

    netw = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # src = "<s>en Ukraine has been beated by Russians</s>" 
    tgt = "</s>vi You don't know how much I love you</s>"
    # pad_src = tknz(src, return_tensors="pt")
    pad_tgt = tknz(tgt, return_tensors="pt")

    # prd = netw.generate(**pad_src)
    # print(tknz.decode(prd[0], skip_special_tokens=True))

    prd = netw.generate(**pad_tgt)
    print(tknz.decode(prd[0], skip_special_tokens=True))

