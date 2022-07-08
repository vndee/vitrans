import os
import re
import random
import string
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PUNC = re.compile(f"[{string.punctuation}]")


def export(X, y, file_path):
    assert len(X) == len(y)
    writer = open(file_path, "w+")
    for src, tgt in tqdm(zip(X, y), desc=f"Exporting to {file_path}", total=len(X)):
        src, tgt = src.strip(), tgt.strip()
        src = re.sub(PUNC, "", src)
        tgt = re.sub(PUNC, "", tgt)

        txt = "{ \"translation\": { \"en\": \"" + src + "\", \"vi\": \"" + tgt + "\" } }"
        writer.write(f"{txt}\n")

    writer.close()


if __name__ == "__main__":
    f_src = open("data/en_sents", "r")
    f_tgt = open("data/vi_sents", "r")
    print(string.punctuation)
    src_lines, tgt_lines = f_src.readlines(), f_tgt.readlines()
    f_src.close()
    f_tgt.close()

    X_train, X_test, y_train, y_test = train_test_split(src_lines, tgt_lines, test_size=0.15, random_state=101)

    os.makedirs("data/kaggle", exist_ok=True)
    export(X_train, y_train, "data/kaggle/train.json")
    export(X_test, y_test, "data/kaggle/test.json")
