import torch
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from transformers import AutoTokenizer

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# load checkpoint
ckpt = torch.load("./../Model_files/checkpoints_v2.3.1/stage1/ckpt_final.pt", map_location=device, weights_only=False)
model_cfg = ckpt["model_cfg"]
model = SpeakMK1LLM(model_cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
prompts = [
    "I wanna pway wif da baww"
]
with torch.no_grad():
    for prompt in prompts:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        _, ce, _ = model(ids, labels=ids)
        print(f"ce={ce.item():.4f}")