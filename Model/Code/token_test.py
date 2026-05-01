import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
data = np.memmap("./../Data/LLM_Data_updated/stage2_childes.bin", dtype=np.uint16, mode="r")

# decode first 500 tokens and inspect
sample = tok.decode(data[:500].astype(np.int64))
print(sample)