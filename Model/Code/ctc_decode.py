"""
ctc_decode.py

Greedy CTC decoding: model logits -> collapsed phone-id sequence, for
feeding into metrics.corpus_per(). Only greedy decoding (argmax + collapse
repeats + drop blank) -- no beam search / LM rescoring. That's the right
starting point for tracking training progress; revisit only if greedy PER
plateaus and you want to see how much a beam search recovers.
"""

from typing import List

import torch


def greedy_ctc_decode(
    logits: torch.Tensor,      # (B, T, V) raw logits or log-probs
    input_lengths: torch.Tensor,  # (B,) valid frame count per item (pre-padding)
    blank_id: int = 0,          # PHONE2IDX['<pad>'] doubles as CTC blank
) -> List[List[int]]:
    """
    Returns a list (len B) of phone-id sequences, one per batch item, with
    repeats collapsed and blank tokens removed (standard CTC decode).
    """
    pred_ids = logits.argmax(dim=-1)  # (B, T)
    out = []
    for b in range(pred_ids.shape[0]):
        length = int(input_lengths[b].item())
        seq = pred_ids[b, :length].tolist()
        collapsed = []
        prev = None
        for p in seq:
            if p != prev and p != blank_id:
                collapsed.append(p)
            prev = p
        out.append(collapsed)
    return out