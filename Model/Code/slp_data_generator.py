"""
SpeakMK1 Stage 3 Synthetic Data Generator
Generates 10,000 multi-turn SLP dialogue entries via the Anthropic API.

Usage:
    python generate_slp_data.py --total 10000 --output slp_stage3_train.jsonl
    python generate_slp_data.py --total 500 --output slp_val.jsonl --start-seed 99999

Design principles:
    - Fully async with rate-limit-aware concurrency (respects Anthropic tier limits)
    - Deterministic seed-based sampling ensures reproducibility and no duplicate configs
    - Validates every entry before writing to disk
    - Resumes from checkpoint if interrupted
    - Outputs clean JSONL ready for SpeakMK1 Stage 3 tokenization
"""

import asyncio
import json
import random
import re
import time
import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional
from llama_cpp import Llama

# ── Client ────────────────────────────────────────────────────────────────────
llm = Llama(
    model_path="/mnt/device/Models/gemma-4-E4B-it-Q4_K_M.gguf",
    n_gpu_layers=-1,  # offload all layers to GPU if possible
    n_ctx=8192,
    n_batch=512,  # Standard batch size
    verbose=True,
)
print(f"Model n_ctx: {llm.n_ctx()}")
MAX_TOKENS = 2500
CONCURRENCY = 1
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0

# ── Taxonomy ──────────────────────────────────────────────────────────────────
DISORDER_TAXONOMY = {
    "Fronting": {
        "subtypes": ["Velar Fronting (k→t, g→d)", "Palatal Fronting (ʃ→s, tʃ→ts)"],
        "ages": [3, 4, 5],
        "goals": [
            "Establish /k/ in word-initial position",
            "Establish /g/ in word-final position",
            "Generalize velars across word positions",
        ],
    },
    "Stopping": {
        "subtypes": [
            "Stopping of /f/ and /v/",
            "Stopping of /s/ and /z/",
            "Stopping of /ʃ/ and /ʒ/",
        ],
        "ages": [4, 5, 6],
        "goals": [
            "Establish fricative continuancy for /f/",
            "Establish /s/ using airflow awareness",
            "Differentiate stops from fricatives in minimal pairs",
        ],
    },
    "Gliding": {
        "subtypes": [
            "Liquid Gliding (r→w)",
            "Liquid Gliding (l→w)",
            "Mixed r/l gliding",
        ],
        "ages": [5, 6, 7],
        "goals": [
            "Establish /r/ in word-initial position",
            "Establish /l/ in syllable-initial position",
            "Generalize /r/ to blends",
        ],
    },
    "Frontal Lisp": {
        "subtypes": [
            "Frontal Lisp (s→th)",
            "Frontal Lisp (z→th)",
            "Bilateral frontal lisp",
        ],
        "ages": [6, 7, 8],
        "goals": [
            "Establish correct tongue placement for /s/",
            "Establish /z/ with dental occlusion awareness",
            "Achieve /s/ in conversational speech",
        ],
    },
    "Lateral Lisp": {
        "subtypes": ["Lateral /s/", "Lateral /z/", "Lateral sibilant cluster"],
        "ages": [7, 8, 9],
        "goals": [
            "Redirect airstream to central channel for /s/",
            "Establish central groove for sibilant production",
        ],
    },
    "Cluster Reduction": {
        "subtypes": [
            "Reduction of bl- and pl- clusters",
            "Reduction of tr- and dr- clusters",
            "Reduction of sp- and st- clusters",
            "Reduction of final clusters (-nd, -nt, -lk)",
        ],
        "ages": [4, 5, 6],
        "goals": [
            "Establish two-element onset clusters",
            "Establish final consonant clusters in CVC+C words",
        ],
    },
    "Final Consonant Deletion": {
        "subtypes": [
            "Deletion of final stops (p, b, t, d, k, g)",
            "Deletion of final nasals (m, n, ŋ)",
            "Deletion of final fricatives (s, z, f, v)",
        ],
        "ages": [3, 4, 5],
        "goals": [
            "Establish final stop closure in CVC words",
            "Generalize final consonants to multisyllabic words",
        ],
    },
    "Assimilation": {
        "subtypes": [
            "Regressive nasal assimilation",
            "Progressive velar assimilation",
            "Labial harmony",
        ],
        "ages": [3, 4, 5],
        "goals": [
            "Reduce assimilatory processes through contrastive pairs",
            "Stabilize correct place features across word positions",
        ],
    },
    "Backing": {
        "subtypes": [
            "Backing of alveolars (t→k)",
            "Backing of alveolars (d→g)",
            "Backing of /n/ to /ŋ/",
        ],
        "ages": [4, 5, 6],
        "goals": [
            "Establish forward tongue placement for /t/",
            "Generalize alveolar place features in connected speech",
        ],
    },
    "Inconsistent Deviant Disorder": {
        "subtypes": [
            "Variable errors on same target word",
            "Inconsistent vowel errors",
            "Inconsistent cluster production",
        ],
        "ages": [7, 8, 9, 10],
        "goals": [
            "Increase production consistency through self-monitoring",
            "Stabilize phonological representations using core vocabulary approach",
        ],
    },
    "Interdental Lisp": {
        "subtypes": ["Frontal interdental /s/", "Frontal interdental /z/"],
        "ages": [5, 6, 7],
        "goals": [
            "Establish retracted tongue placement for /s/",
            "Build dental occlusion awareness for sibilants",
        ],
    },
    "Palatalization": {
        "subtypes": [
            "Palatalization of alveolars before /j/",
            "Excessive palatalization of /s/",
        ],
        "ages": [6, 7, 8],
        "goals": [
            "Reduce palatalization in consonant+glide sequences",
        ],
    },
}

DISTRACTORS = [
    "notices a toy across the room",
    "starts talking about their pet",
    "asks about lunch",
    "gets excited about a show they watched",
    "notices something outside the window",
    "starts playing with a fidget toy",
    "brings up their sibling",
    "asks if therapy is almost done",
    "starts telling an unrelated story",
    "starts humming to themselves",
]

CUING_STRATEGIES = [
    "Verbal Prompting ('Put your tongue behind your teeth')",
    "Visual Demonstration (SLP models placement)",
    "Tactile Cueing (touching alveolar ridge awareness)",
    "Minimal Pair Contrast",
    "Scaffolded Expansion",
    "Play-Based Naturalistic Elicitation",
    "Self-Monitoring Training",
    "Phonological Awareness Activities",
    "Core Vocabulary Approach",
    "Integral Stimulation ('Watch me, listen to me, say it with me')",
    "Backward Chaining",
    "Contrastive Drilling with Reinforcement",
]

FEW_SHOT_EXAMPLES = '''
{"metadata": {"age": 4, "disorder": "Fronting", "specific_error": "Velar Fronting (k→t, g→d)", "clinical_goal": "Establish /k/ in word-initial position using visual+tactile cues"}, "dialogue": [{"turn": 1, "child_input": "I want a tup of water. And a tootie!", "model_thought": "Child substitutes /k/ with /t/ in 'cup' (→'tup') and 'cookie' (→'tootie'). Classic velar fronting. Strategy: Introduce visual placement cue for velar /k/ — tongue back, not front. Use minimal pair contrast (tip vs. back of tongue) with a physical anchor like touching throat.", "slp_response": "Mmm, a cup of water sounds yummy! I heard you say 'tup' — let's play a little game with that word. Can you put your hand right here on your throat? [touches own throat] Now say 'tup'... do you feel anything? Now watch me — KUP. I make that sound way back here, like a little frog jumping at the back of my mouth. Want to try? Let's say 'k... k... cup' together!"}, {"turn": 2, "child_input": "Tup! ...wait— kuh. TUP. I dunno. My fwog has a tup too, he dwinks from it!", "model_thought": "Child is self-monitoring (good sign — noticing the difference). Partial production of /k/ in isolation ('kuh'). Distracted toward pet frog. Affect is playful. Strategy: Reinforce the self-correction attempt strongly, then use the frog as a motivational hook (child-led scaffolding). Don't correct 'fwog' (different target) — stay on /k/.", "slp_response": "Oh WOW — did you hear yourself say 'kuh'?! That was SO close! Your tongue jumped to the back of your mouth for just a second — that's exactly right! And your frog has a cup? That is the coolest thing. Does he drink from a... 'kuh-up'? Can you tell me what your frog drinks from? Try sneaking that 'k' sound in there!"}, {"turn": 3, "child_input": "He dwinks from a... a... KUP! A kup! Heheheh I did it!", "model_thought": "Successful /k/ production in 'cup' with clear velar place of articulation. Child is delighted. Reinforce immediately and maximally. Brief extension to another /k/ word to begin generalization.", "slp_response": "YES! You absolutely did it!! That was a perfect 'KUP'! Give me a high five! Your frog is so lucky to have a friend who can say 'cup' so well. Now I'm wondering... what color is your frog? Is he a 'cool' color? C-c-cool? Oh, that starts with the same sound — 'kuh'!"}]}
{"metadata": {"age": 6, "disorder": "Stopping", "specific_error": "Stopping of fricatives (f→p, v→b, s→t)", "clinical_goal": "Establish /f/ in all word positions using airflow awareness"}, "dialogue": [{"turn": 1, "child_input": "I pell really bad today. My peet hurt prom running.", "model_thought": "Multiple stopping errors: 'fell' → 'pell' (f→p), 'feet' → 'peet' (f→p), 'from' → 'prom' (f→p). Child is complaining about physical discomfort — acknowledge emotional state first before clinical work. Strategy: Establish rapport, then introduce airflow demonstration ('windy teeth' cue). Don't target all errors at once — pick 'feet' as it's concrete and body-based.", "slp_response": "Oh no, your feet hurt? That sounds really ouchy after running! I'm sorry. Hey, I have a question — can you feel air on your hand when you blow? [blows gently on own hand] Do that with me. Now... watch my mouth say 'feet' — ffff-eet. See my teeth? I'm making wind with my teeth! Can you make windy teeth and say 'ffff'?"}, {"turn": 2, "child_input": "Pffff? Like that? Pp... my dog is over there, can we do dog words?", "model_thought": "Child produced a bilabial fricative approximation (pf) — lips together with some airflow, not quite /f/ yet but showing understanding of airflow concept. Then distracted by dog. Strategy: Redirect using the dog as the vehicle — 'fur', 'fetch', 'food' all start with /f/. Capitalizes on motivation without losing the clinical target.", "slp_response": "Ooh, your dog! What's his name? [pause] Perfect — now here's the fun part. Does your dog have... FUR? Ffff-ur. Fur! Put your top teeth on your bottom lip like this [demonstrates] and blow a little wind. Ffff... fur! Can your dog do tricks? Can he... FETCH? Ffff-etch! Let's try 'fetch' together — ready?"}, {"turn": 3, "child_input": "Ffff... fetch! FETCH! He does fetch! His name is Biscuit and he fetches FFFFFFAST!", "model_thought": "Successful /f/ production in 'fetch' and strong generalization attempt in 'fast' with exaggerated /f/ onset. Child is highly motivated. Reinforce, then introduce /f/ in medial position gently via 'Biscuit's food' or similar.", "slp_response": "FAST!! I heard that beautiful 'fffff' — you sounded like the wind! And Biscuit is a perfect name for a dog who runs FAST. Does Biscuit eat dog food? Fff-ood. Food! Can you say 'Biscuit eats food'? I bet you can sneak that 'f' in there — you're getting SO good at this!"}, {"turn": 4, "child_input": "Biscuit eats... pood. Hmm. Ffff... fffffood! FOOD! Hahaha!", "model_thought": "Initial stopping regression ('pood') followed by immediate self-correction with prolonged /f/ onset. This is excellent metacognitive behavior — child caught their own error. Celebrate strongly. This is end-of-session reinforcement territory.", "slp_response": "Did you HEAR yourself catch that?! You said 'pood,' then you went 'ffff' and fixed it all by yourself — that's what real speech heroes do! I am SO proud of you. You worked really hard today. I'm going to tell your mom that you learned to make 'windy teeth' words — 'feet,' 'fetch,' AND 'food.' That's three! You're a superstar."}]}
{"metadata": {"age": 5, "disorder": "Gliding", "specific_error": "Liquid Gliding (r→w, l→w)", "clinical_goal": "Establish /r/ in word-initial position using tongue-tip elevation cues"}, "dialogue": [{"turn": 1, "child_input": "Can we weed the book about the wabbit? I wike that one.", "model_thought": "Gliding of /r/ in 'read'→'weed', 'rabbit'→'wabbit', and 'like'→'wike' (also l→w). Multiple targets present. Priority: /r/ as it's the stated goal. The request to read a book is a perfect naturalistic context — use the book as a prop to embed /r/ targets organically.", "slp_response": "I LOVE that book — the rabbit one is my favorite too! Let's absolutely read it. But first, let me tell you a secret about the rabbit's name. His name starts with a super tricky sound — 'rrr' — like a growling tiger. Can you growl like a tiger? Rrrr! Your tongue tip goes up, up, up inside your mouth. Try it: 'rrrr'!"}, {"turn": 2, "child_input": "Wrrr? Rrr? [giggles] That's silly. My tongue is doing a dance!", "model_thought": "Child is attempting /r/ — the 'wrrr' suggests approximation of the sound with some lip rounding still present. Tongue awareness is increasing ('my tongue is doing a dance'). Positive affect. Strategy: Reinforce proprioceptive awareness, reduce lip rounding cue, try eliciting in 'run' or 'red' using color context from the book.", "slp_response": "Your tongue IS dancing — and that's exactly what it needs to do to make 'rrr'! You were so close! This time, try to keep your lips a little more relaxed — don't let them make a 'w' shape. Just let your tongue do all the work. See the red on the rabbit's jacket? Can you say... 'rrrred'? Slow and growly!"}, {"turn": 3, "child_input": "Rrrr... wed. Rrrr-wed. Wed? Is that wight?", "model_thought": "Child produces a retroflex or bunched /r/ approximation in isolation then loses it before the vowel, producing 'wed'. Self-questioning ('is that right?') shows metalinguistic awareness. Strategy: Isolate /r/ success, use chaining — hold the 'rrrr' longer before releasing into the vowel. Avoid correcting 'wight' this turn.", "slp_response": "You are SO close! I heard a great 'rrr' at the beginning — hold onto it longer! Let's do it in slow motion: 'rrrr'... keep growling... 'rrrr-ED.' Like you're a tiger who can't stop growling. Rrrrr-ED! Try again — hold that growl as long as you can before you say the 'ed' part!"}]}
'''
# ── Prompt builder ─────────────────────────────────────────────────────────────


def build_prompt(config: dict) -> str:
    disorder = config["disorder"]
    subtype = config["subtype"]
    age = config["age"]
    goal = config["goal"]
    distractor = config["distractor"]
    strategy = config["strategy"]
    n_turns = config["n_turns"]
    complexity = config["complexity"]

    return f"""You are a dataset generation expert creating synthetic training data for SpeakMK1, a pediatric AI speech-language pathologist.

Generate EXACTLY ONE multi-turn dialogue entry in strict JSON format (no markdown, no preamble, pure JSON).

## Configuration
- Child age: {age} years old
- Disorder: {disorder} — {subtype}
- Clinical goal: {goal}
- Primary cuing strategy: {strategy}
- Distractor event to include: Child {distractor} (insert this organically into ONE of the turns)
- Number of turns: {n_turns}
- Complexity level: {complexity}/3 (1=simple, 2=moderate, 3=complex with regression/emotional content)

## Required JSON Schema
{{
  "metadata": {{
    "age": <int>,
    "disorder": "<disorder name>",
    "specific_error": "<phonetic description of the error>",
    "clinical_goal": "<goal string>",
    "primary_strategy": "<strategy used>",
    "complexity_level": <int>
  }},
  "dialogue": [
    {{
      "turn": <int starting at 1>,
      "child_input" (not "child_speech", not "input"): "<realistic child speech with phonetic misspellings reflecting the disorder>",
      "model_thought" (not "model_thoughts", not "thought")  : "<clinical chain-of-thought: error analysis, strategy rationale, next move>",
      "slp_response" (not "sl_response", not "response", not "slp_responses"): "<warm, age-appropriate SLP response with fillers, scaffolding, and emotional support>"
    }}
  ]
}}

here are a few examples of the expected output format (note: these are just examples, do NOT copy their content or structure):
{FEW_SHOT_EXAMPLES}

## Critical Quality Requirements
1. PHONETIC ACCURACY: child_input must reflect the SPECIFIC error pattern (e.g., /k/ → /t/ for fronting). Use consistent substitutions.
2. CLINICAL FIDELITY: model_thought must name the specific phoneme error, identify the substitution/omission/distortion/addition type, and justify the strategy choice clinically.
3. WARMTH: slp_response must feel human — include fillers ("Hmm," "Oh wow," "Let's see..."), celebrations, and emotional validation appropriate to age {age}.
4. NON-LINEARITY: The distractor ("{distractor}") must appear in exactly one turn and the SLP must gracefully redirect while using the distraction as a motivational bridge if possible.
5. SCAFFOLDING OVER CORRECTION: SLP never just says "that's wrong, say X." Always offer a cue or a bridge.
6. GENERALIZATION: At least one turn should introduce a NEW word with the same error pattern (not just the same word repeatedly).
7. VARIETY: Use different sentence structures, topics, and vocabulary. Do NOT use "banana," "cup," or "fish" as primary targets unless they emerge naturally.
8. USE THE EXACT KEYS FOR JSON: "child_input", "model_thought", "slp_response". Do NOT use variants like "child_speech" or "response".
9. Keep each slp_response between 80 and 150 WORDS. Keep each model_thought at or more than 60 characters.

Output ONLY the JSON object. No commentary, no markdown fences."""


# ── Generation logic ───────────────────────────────────────────────────────────


def sample_config(seed: int) -> dict:
    rng = random.Random(seed)
    disorder = rng.choice(list(DISORDER_TAXONOMY.keys()))
    tax = DISORDER_TAXONOMY[disorder]
    return {
        "disorder": disorder,
        "subtype": rng.choice(tax["subtypes"]),
        "age": rng.choice(tax["ages"]),
        "goal": rng.choice(tax["goals"]),
        "distractor": rng.choice(DISTRACTORS),
        "strategy": rng.choice(CUING_STRATEGIES),
        "n_turns": rng.choice([3, 3, 4, 4, 4]),  # weighted toward 4
        "complexity": rng.choice([1, 2, 2, 3]),  # weighted toward moderate
    }


def validate_entry(entry: dict) -> bool:
    try:
        assert "metadata" in entry, "missing metadata"
        assert "dialogue" in entry, "missing dialogue"
        assert isinstance(entry["dialogue"], list), "dialogue not a list"
        assert len(entry["dialogue"]) >= 3, f"only {len(entry['dialogue'])} turns"
        for turn in entry["dialogue"]:
            assert "child_input" in turn, "missing child_input"
            assert "model_thought" in turn, "missing model_thought"
            assert "slp_response" in turn, "missing slp_response"
            assert (
                len(turn["slp_response"]) > 80
            ), f"slp_response too short: {len(turn['slp_response'])} chars"
            assert (
                len(turn["slp_response"]) <= 800
            ), f"slp_response too long: {len(turn['slp_response'])} chars"
            assert (
                len(turn["model_thought"]) > 60
            ), f"model_thought too short: {len(turn['model_thought'])} chars"
        return True
    except (AssertionError, KeyError, TypeError) as e:
        print(f"Validation failed reason: {e}")
        return False


async def generate_one(seed: int, semaphore: asyncio.Semaphore) -> Optional[dict]:
    config = sample_config(seed)
    prompt = build_prompt(config)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda p=prompt, t=0.85 + (
                        attempt * 0.1
                    ): llm.create_chat_completion(
                        messages=[{"role": "user", "content": p}],
                        max_tokens=4096,
                        temperature=t,
                    ),
                )
                raw = response["choices"][0]["message"]["content"].strip()
                print(f"[seed={seed}] RAW:\n{raw[:800]}\n---")

                # Strip accidental markdown fences
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                raw = re.sub(r'"sl_response"', '"slp_response"', raw)
                raw = re.sub(
                    r'"slp_responses"', '"slp_response"', raw
                )  # other common variants
                raw = re.sub(r'"model_thoughts"', '"model_thought"', raw)
                # Attempt to salvage truncated JSON
                if raw.count("{") > raw.count("}"):
                    raw = raw + "}" * (raw.count("{") - raw.count("}"))
                if raw.count("[") > raw.count("]"):
                    raw = raw + "]" * (raw.count("[") - raw.count("]"))

                entry = json.loads(raw)
                entry["_seed"] = seed  # for reproducibility tracing

                if validate_entry(entry):
                    return entry
                else:
                    print(f"[seed={seed}] Validation failed, attempt {attempt+1}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[seed={seed}] Parse error: {e}")
                print(f"[seed={seed}] Raw output: {raw[:500]}")
                await asyncio.sleep(1)

    print(f"[seed={seed}] Failed after {RETRY_ATTEMPTS} attempts. Skipping.")
    return None


# ── Checkpoint system ──────────────────────────────────────────────────────────


def load_checkpoint(checkpoint_path: Path) -> set:
    """Returns set of seeds already successfully generated."""
    if not checkpoint_path.exists():
        return set()
    completed = set()
    with open(checkpoint_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "_seed" in entry:
                    completed.add(entry["_seed"])
            except json.JSONDecodeError:
                continue
    print(f"Resumed: {len(completed)} entries already completed.")
    return completed


# ── Main orchestration ─────────────────────────────────────────────────────────


async def main(total: int, output_path: str, start_seed: int, concurrency: int):
    output = Path(output_path)
    checkpoint = output  # Same file used as checkpoint

    completed_seeds = load_checkpoint(checkpoint)
    seeds_to_run = [
        start_seed + i for i in range(total) if (start_seed + i) not in completed_seeds
    ]

    print(f"Generating {len(seeds_to_run)} entries → {output}")
    print(f"Concurrency: {concurrency} | Model: gemma-4-26B-A4B-it-Q4_K_M.gguf")

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    fail_count = 0

    # Process in batches to allow periodic flushing
    BATCH_SIZE = 50
    for batch_start in range(0, len(seeds_to_run), BATCH_SIZE):
        batch = seeds_to_run[batch_start : batch_start + BATCH_SIZE]
        tasks = [generate_one(seed, semaphore) for seed in batch]
        results = await asyncio.gather(*tasks)

        # Append to output file
        with open(output, "a", encoding="utf-8") as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    fail_count += 1

        total_done = len(completed_seeds) + success_count
        print(
            f"Progress: {total_done}/{total} | Success: {success_count} | Failed: {fail_count}"
        )

    print(f"\nDone. Total entries written: {success_count + len(completed_seeds)}")
    print(f"Failed/skipped: {fail_count}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpeakMK1 Stage 3 Synthetic Data Generator"
    )
    parser.add_argument(
        "--total", type=int, default=10000, help="Total entries to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../Data/SLP/slp_stage3_train.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Starting random seed (for reproducibility)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="Max concurrent API requests",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            total=args.total,
            output_path=args.output,
            start_seed=args.start_seed,
            concurrency=args.concurrency,
        )
    )
