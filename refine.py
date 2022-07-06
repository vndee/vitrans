import os
from glob import glob


if __name__ == "__main__":
    f_src = open("data/dev/en.txt", "r")
    f_tgt = open("data/dev/vi.txt", "r")
    writer = open("dev/data.jsonl", "w+")

    src_lines, tgt_lines = f_src.readlines(), f_tgt.readlines()
    for src, tgt in zip(src_lines, tgt_lines):
        src, tgt = src.strip(), tgt.strip()
        txt = "{ \"translation\": { \"en\": \"" + src + "\", \"vi\": \"" + tgt + "\" } }"
        writer.write(f"{txt}\n")
    
    writer.close()
    f_src.close()
    f_tgt.close()

