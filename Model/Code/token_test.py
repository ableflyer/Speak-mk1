import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
data = np.memmap("./../Data/LLM_Data_updated/stage2_childes.bin", dtype=np.uint16, mode="r")

# decode first 500 tokens and inspect
sample = data[:500].tolist()
print(tok.decode(sample))