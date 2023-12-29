import torch
import numpy as np
import random
import time
import re


from tqdm import tqdm
from Bio import SeqIO


# compute running time by 'with' grammar
class TimeCounter:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.start = time.time()
        print(self.text, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        t = end - self.start
        print(f"\nFinished. The time is {t:.2f}s.\n", flush=True)


def progress_bar(now: int, total: int, desc: str = '', end='\n'):
    length = 50
    now = now if now <= total else total
    num = now * length // total
    progress_bar = '[' + '#' * num + '_' * (length - num) + ']'
    display = f'{desc:<10} {progress_bar} {int(now/total*100):02d}% {now}/{total}'

    print(f'\r\033[31m{display}\033[0m', end=end, flush=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_seed():
    torch.seed()
    torch.cuda.seed()
    np.random.seed()
    random.seed()
    torch.backends.cudnn.deterministic = False


def a3m_formalize(input, output, keep_gap=True):
    with open(output, 'w') as w:
        for record in SeqIO.parse(input, 'fasta'):
            desc = record.description
            if keep_gap:
                seq = re.sub(r"[a-z]", "", str(record.seq))
            else:
                seq = re.sub(r"[a-z-]", "", str(record.seq))
            w.write(f">{desc}\n{seq}\n")


def merge_file(file_list: list, save_path: str):
    with open(save_path, 'w') as w:
        for i, file in enumerate(file_list):
            with open(file, 'r') as r:
                for line in tqdm(r, f"Merging {file}... ({i+1}/{len(file_list)})"):
                    w.write(line)