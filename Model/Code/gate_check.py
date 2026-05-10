import torch
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig

cfg = SpeakMK1LLMConfig(
    vocab_size=50283, d_model=512, d_state=64, num_blocks=6,
    nheads_ssm=8, nheads_attn=8, top_k_audio=32,
    num_experts=4, top_k_experts=2, dropout=0.0, aux_loss_weight=1e-2,
)
ckpt = torch.load("../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt", map_location="cpu", weights_only=False)
llm = SpeakMK1LLM(cfg)
llm.load_state_dict(ckpt["model"])

for i, block in enumerate(llm.blocks):
    if hasattr(block.cross_attn, 'gate'):
        print(f"block {i}: gate = {block.cross_attn.gate.data.item():.6f}")