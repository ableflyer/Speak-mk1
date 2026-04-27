import numpy as np
data = np.memmap("../Data/LLM_Data_updated/stage4_slp_labels.bin", dtype=np.uint16, mode="r")
print(f"{len(data)/1e6:.1f}M tokens")
print(f"{(len(data)-1)//512} sequences")