"""
metrics.py

Metrics for evaluating PHONEME RECOGNITION ("did the model correctly
tokenize/identify the phone sequence") -- deliberately kept separate from
clinical evaluation (voicing/manner/place classification accuracy, and any
future correctness/disorder scoring). Those measure different things and
mixing them into one number would hide which one is actually failing.

Primary metric: Phone Error Rate (PER), the phone-level analogue of Word
Error Rate in ASR. Computed via edit distance between the predicted phone-id
sequence and the reference phone-id sequence (see
frame_alignment.build_phone_sequence / UltraSuiteDataset's "phone_seq").

Secondary metric: frame-level accuracy. Useful for debugging alignment
quality, but NOT the headline number -- it's sensitive to MFA boundary
jitter and doesn't penalize insertions/deletions the way PER does.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class EditStats:
    substitutions: int
    insertions: int
    deletions: int
    ref_length: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def per(self) -> float:
        """Phone error rate for this single utterance. Can exceed 1.0 if
        there are more insertions than reference phones -- that's correct
        ASR-metric behavior, not a bug."""
        if self.ref_length == 0:
            return 0.0 if self.total_errors == 0 else float("inf")
        return self.total_errors / self.ref_length


def _levenshtein_with_ops(hyp: Sequence[int], ref: Sequence[int]) -> Tuple[int, int, int]:
    """
    Standard DP edit distance between hyp and ref, returning
    (substitutions, insertions, deletions) -- insertions/deletions are
    relative to ref (i.e. "insertion" = hyp has an extra phone not in ref).
    """
    n, m = len(hyp), len(ref)
    # dp[i][j] = min edits to turn hyp[:i] into ref[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # i deletions... (hyp has extra -> counted as insertion below;
                       # see op backtrace for the actual counted category)
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion (ref phone missing from hyp)
                    dp[i][j - 1],      # insertion (extra hyp phone)
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace to classify each edit.
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp[i - 1] == ref[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # ref[j-1] has no counterpart in hyp -> hyp is missing a phone
            # that should have been produced. ASR convention: deletion.
            dels += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # hyp[i-1] has no counterpart in ref -> hyp produced an extra
            # phone that shouldn't be there. ASR convention: insertion.
            ins += 1
            i -= 1
        else:  # pragma: no cover - shouldn't happen
            break
    return subs, ins, dels


def utterance_per(hyp_ids: Sequence[int], ref_ids: Sequence[int]) -> EditStats:
    """PER for a single utterance."""
    subs, ins, dels = _levenshtein_with_ops(hyp_ids, ref_ids)
    return EditStats(substitutions=subs, insertions=ins, deletions=dels, ref_length=len(ref_ids))


def corpus_per(hyps: List[Sequence[int]], refs: List[Sequence[int]]) -> dict:
    """
    Micro-averaged PER across an entire eval set: sum all edits, sum all
    reference lengths, divide once. This is the number to report -- do NOT
    average per-utterance PER values (that overweights short utterances).
    """
    assert len(hyps) == len(refs), "hyps and refs must be the same length (paired per-utterance)"
    total_subs = total_ins = total_dels = total_ref_len = 0
    per_utterance = []
    for h, r in zip(hyps, refs):
        stats = utterance_per(h, r)
        total_subs += stats.substitutions
        total_ins += stats.insertions
        total_dels += stats.deletions
        total_ref_len += stats.ref_length
        per_utterance.append(stats)

    total_errors = total_subs + total_ins + total_dels
    return {
        "per": total_errors / total_ref_len if total_ref_len > 0 else float("inf"),
        "substitutions": total_subs,
        "insertions": total_ins,
        "deletions": total_dels,
        "total_errors": total_errors,
        "total_ref_phones": total_ref_len,
        "num_utterances": len(hyps),
        "per_utterance": per_utterance,  # for digging into worst-case utterances
    }


# ---------------------------------------------------------------------------
# Secondary / debugging metric -- frame-level accuracy
# ---------------------------------------------------------------------------

def frame_accuracy(
    pred_frame_ids: Sequence[int],
    ref_frame_ids: Sequence[int],
    ignore_id: int = 0,  # PHONE2IDX['<pad>'] by construction in this loader
) -> dict:
    """
    Simple per-frame accuracy, ignoring padded frames. Reported separately
    from PER -- this tells you about frame/boundary alignment quality, not
    phoneme recognition quality. A model can have poor frame accuracy near
    phone boundaries (a few frames off due to MFA jitter) while still
    getting the phone SEQUENCE completely right, which is what actually
    matters for "did it correctly tokenize the phoneme."
    """
    assert len(pred_frame_ids) == len(ref_frame_ids)
    correct = 0
    total = 0
    for p, r in zip(pred_frame_ids, ref_frame_ids):
        if r == ignore_id:
            continue
        total += 1
        if p == r:
            correct += 1
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


if __name__ == "__main__":
    # Self-test against known edit-distance cases.
    ref = [1, 2, 3, 4, 5]
    hyp_perfect = [1, 2, 3, 4, 5]
    hyp_one_sub = [1, 2, 9, 4, 5]
    hyp_one_del = [1, 2, 4, 5]         # ref[2]=3 missing from hyp
    hyp_one_ins = [1, 2, 3, 9, 4, 5]   # extra 9 inserted

    for name, h in [("perfect", hyp_perfect), ("one_sub", hyp_one_sub),
                     ("one_del", hyp_one_del), ("one_ins", hyp_one_ins)]:
        s = utterance_per(h, ref)
        print(f"{name:10s} subs={s.substitutions} ins={s.insertions} dels={s.deletions} PER={s.per:.3f}")

    corpus = corpus_per(
        hyps=[hyp_perfect, hyp_one_sub, hyp_one_del, hyp_one_ins],
        refs=[ref, ref, ref, ref],
    )
    print()
    print("corpus PER (micro-averaged):", round(corpus["per"], 4))
    print("breakdown:", {k: v for k, v in corpus.items() if k != "per_utterance"})