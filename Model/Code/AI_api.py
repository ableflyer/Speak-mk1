import io
import base64
import traceback
import numpy as np
import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer
import sys

# ── Model imports (same as your inference script) ──────────────────────────
from audio_encoder import AudioEncoder, AudioEncoderConfig
from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from train_proj import DirectAudioProjection
from kokoro_onnx import Kokoro

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000

ENCODER_PATH  = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt"
PROJ_PATH     = "../Model_files/audio_proj_training/audio_proj_best.pt"
LLM_PATH      = "../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt"
KOKORO_MODEL  = "../Model_files/kokoro-onnx-models/kokoro-v1.0.onnx"
KOKORO_VOICES = "../Model_files/kokoro-onnx-models/voices-v1.0.bin"

# ── Lazy globals (populated at startup) ────────────────────────────────────
encoder   = None
proj      = None
llm       = None
kokoro    = None
tokenizer = None

app = Flask(__name__)
CORS(app)  # Ali's frontend needs this


# ── Model loader ───────────────────────────────────────────────────────────

def load_models():
    global encoder, proj, llm, kokoro, tokenizer

    print("Loading encoder...")
    enc_cfg = SmallConfig(d_model=512, llm_dim=4096)
    encoder = AudioEncoder(enc_cfg).to(DEVICE)
    print([attr for attr in dir(encoder) if not attr.startswith('_')])
    print(encoder.phonological)
    print(type(encoder.phonological))
    enc_ckpt = torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(enc_ckpt, strict=False)
    encoder.eval()

    print("Loading projection...")
    proj = DirectAudioProjection(512, 512).to(DEVICE)
    proj_ckpt = torch.load(PROJ_PATH, map_location=DEVICE, weights_only=False)
    proj.load_state_dict(proj_ckpt["audio_proj"])
    proj.eval()

    print("Loading LLM...")
    llm_cfg = SpeakMK1LLMConfig(
        vocab_size=50283,
        d_model=512,
        d_state=64,
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        aux_loss_weight=1e-2,
    )
    llm = SpeakMK1LLM(llm_cfg).to(DEVICE)
    llm_ckpt = torch.load(LLM_PATH, map_location=DEVICE, weights_only=False)
    llm.load_state_dict(llm_ckpt["model"], strict=True)
    with torch.no_grad():
        for block in llm.blocks:
            if hasattr(block.cross_attn, "gate"):
                block.cross_attn.gate.data.fill_(0.3)
    llm.eval()

    print("Loading Kokoro TTS...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|system|>", "<|child|>", "<|slp|>", "<|think|>", "<|endturn|>"
    ]})

    print("All models loaded.")


# ── Helpers ────────────────────────────────────────────────────────────────

def audio_bytes_to_mel(audio_bytes: bytes) -> torch.Tensor:
    """Decode uploaded audio bytes → mel tensor (1, T, 80) on DEVICE."""
    buf = io.BytesIO(audio_bytes)
    audio_np, sr = sf.read(buf, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)   # stereo → mono
    if sr != SAMPLE_RATE:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel_np = librosa.feature.melspectrogram(
        y=audio_np,
        sr=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=0.0,
        fmax=8000.0,
    )
    mel_np = librosa.power_to_db(mel_np, ref=np.max)
    mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 80)
    return mel


def generate_text(input_ids: torch.Tensor, audio_out: torch.Tensor,
                  max_new_tokens: int = 150,
                  temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = llm(input_ids=generated, audio_out=audio_out, audio_padding_mask=None)
            next_logits = logits[0, -1] / temperature

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[torch.multinomial(probs, 1)]

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=False)


def text_to_audio_b64(text: str) -> str:
    """Run Kokoro TTS and return base64-encoded WAV string."""
    samples, sr = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(transcription: str, target_phrase: str, target_sound: str,
                 accuracy: float, child_age: int, focus_area: str) -> str:
    system = (
        "You are a warm, expert AI speech-language pathologist helping a child with "
        "articulation errors. Analyze the error and provide encouraging corrective feedback."
    )
    context_parts = []
    if target_phrase:
        context_parts.append(f"Target phrase: '{target_phrase}'.")
    if target_sound:
        context_parts.append(f"Target sound: /{target_sound}/.")
    if accuracy is not None:
        context_parts.append(f"Accuracy score: {accuracy:.0f}%.")
    if child_age:
        context_parts.append(f"Child age: {child_age}.")
    if focus_area:
        context_parts.append(f"Focus area: {focus_area}.")

    context = " ".join(context_parts)
    child_turn = f"{context} Child said: \"{transcription}\"" if context else transcription

    return (
        f"<|system|>{system}"
        f"<|child|>{child_turn}"
        f"<|slp|>"
    )


# def phoneme_scores_from_features(audio_feats: torch.Tensor) -> dict:
#     """
#     Derive a rough phoneme-feature score vector from the encoder's frame embeddings.
#     Uses the encoder's phonological head outputs if available, otherwise falls back
#     to a norm-based heuristic. Returns a dict the frontend can display.
#     """
#     with torch.no_grad():
#         # Try to get phonological feature logits from the encoder directly
#         if hasattr(encoder, "voicing_head") and hasattr(encoder, "manner_head") and hasattr(encoder, "place_head"):
#             voicing_logits = encoder.voicing_head(audio_feats)   # (1, T, 2)
#             manner_logits  = encoder.manner_head(audio_feats)    # (1, T, N)
#             place_logits   = encoder.place_head(audio_feats)     # (1, T, M)

#             voicing_conf = torch.softmax(voicing_logits, dim=-1).max(dim=-1).values.mean().item()
#             manner_conf  = torch.softmax(manner_logits,  dim=-1).max(dim=-1).values.mean().item()
#             place_conf   = torch.softmax(place_logits,   dim=-1).max(dim=-1).values.mean().item()
#         else:
#             # Fallback: use frame-level norm variance as a proxy for confidence
#             norms = audio_feats.norm(dim=-1)  # (1, T)
#             mean_norm = norms.mean().item()
#             # Calibrate against known encoder behaviour: voicing ~0.96, manner ~0.93, place ~0.40
#             voicing_conf = min(mean_norm / 10.0, 0.99)
#             manner_conf  = voicing_conf * 0.97
#             place_conf   = voicing_conf * 0.42

#     return {
#         "voicing": round(voicing_conf * 100, 1),
#         "manner":  round(manner_conf  * 100, 1),
#         "place":   round(place_conf   * 100, 1),
#     }

def get_pre_phonological_features(mel):
    with torch.no_grad():
        x = encoder.stem(mel)
        x = x[0] if isinstance(x, tuple) else x

        for layer in encoder.uni_mamba_layers:
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x

        x = encoder.rms_att(x, key_padding_mask=None)
        x = x[0] if isinstance(x, tuple) else x

        for layer in encoder.bi_mamba_moe_layers:
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x
        print("pre-phon features:", x.shape, file=sys.stderr)
        # raise RuntimeError(f"stem out: DEBUG | rms_att type: ... | final x shape: {x.shape}")
        return x  # (B, T, 512)


def phoneme_scores_from_features(mel: torch.Tensor) -> dict:
    with torch.no_grad():
        pre_phon, _ = encoder.phonological(mel)
        phon_out = pre_phon

        scores = {}
        for name, head in encoder.phonological.heads.items():
            logits = head(phon_out)
            conf = torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
            scores[name] = round(conf * 100, 1)

        correctness_logits = encoder.phonological.heads['correctness'](phon_out)
        correctness_prob = torch.softmax(correctness_logits, dim=-1)[..., 1].mean().item()
        scores['correctness'] = round(correctness_prob * 100, 1)

    return scores


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/api/ai/feedback", methods=["POST"])
def feedback():
    """
    Accepts multipart/form-data OR application/json.

    JSON fields (from api.js getFeedback):
      transcription, targetPhrase, targetSound, accuracy, childAge, focusArea

    Optional multipart field:
      audio  — raw audio file (wav/mp3/ogg) of the child's utterance

    Returns:
      {
        text:           str,          # SLP response text
        audio:          str,          # base64-encoded WAV of TTS output
        phoneme_scores: { voicing, manner, place },
        prompt_used:    str           # for debugging
      }
    """
    try:
        # ── Parse inputs ──────────────────────────────────────────────────
        if request.content_type and "multipart" in request.content_type:
            transcription = request.form.get("transcription", "")
            target_phrase = request.form.get("targetPhrase", "")
            target_sound  = request.form.get("targetSound", "")
            accuracy      = float(request.form.get("accuracy", 0) or 0)
            child_age     = int(request.form.get("childAge", 6) or 6)
            focus_area    = request.form.get("focusArea", "")
            audio_file    = request.files.get("audio")
        else:
            data          = request.get_json(force=True) or {}
            transcription = data.get("transcription", "")
            target_phrase = data.get("targetPhrase", "")
            target_sound  = data.get("targetSound", "")
            accuracy      = float(data.get("accuracy") or 0)
            child_age     = int(data.get("childAge") or 6)
            focus_area    = data.get("focusArea", "")
            audio_file    = None

        # ── Audio encode (if audio provided, else use silent placeholder) ─
        with torch.no_grad():
            if audio_file is not None:
                mel = audio_bytes_to_mel(audio_file.read())
            else:
                # No audio uploaded — create a minimal silent mel so the
                # LLM still gets a valid (though uninformative) audio token
                silent = np.zeros(SAMPLE_RATE * 1, dtype=np.float32)
                mel_np = librosa.feature.melspectrogram(
                    y=silent, sr=SAMPLE_RATE,
                    n_fft=400, hop_length=160, n_mels=80,
                )
                mel_np = librosa.power_to_db(mel_np, ref=np.max)
                mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # mel = audio_bytes_to_mel(audio_file.read())
            audio_feats = encoder.encode_features(mel)   # (1, T, 512)
            audio_out   = proj(audio_feats)               # (1, T, 512)

            phoneme_scores = phoneme_scores_from_features(audio_out)

            # ── Build prompt & generate ────────────────────────────────────
            prompt    = build_prompt(transcription, target_phrase, target_sound,
                                     accuracy, child_age, focus_area)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            full_output = generate_text(input_ids, audio_out)

        # Extract only the SLP turn
        _, _, slp_response = full_output.partition("<|slp|>")
        slp_response = slp_response.replace("<|endturn|>", "").strip()

        # ── TTS ───────────────────────────────────────────────────────────
        audio_b64 = text_to_audio_b64(slp_response) if slp_response else ""

        return jsonify({
            "text":           slp_response,
            "audio":          audio_b64,          # base64 WAV, play with <audio src="data:audio/wav;base64,...">
            "phoneme_scores": phoneme_scores,
            "prompt_used":    prompt,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/score", methods=["POST"])
def score():
    """
    Accepts multipart/form-data OR application/json.

    JSON fields:
      transcription, targetPhrase, targetSound, accuracy, childAge, focusArea

    Optional multipart field:
      audio  — raw audio file

    Returns:
      {
        overall:        float,   # 0–100
        phoneme_scores: { voicing, manner, place },
        details: {
          fluency:        float,
          accuracy:       float,
          articulation:   float,
        }
      }
    """
    try:
        if request.content_type and "multipart" in request.content_type:
            accuracy   = float(request.form.get("accuracy", 0) or 0)
            audio_file = request.files.get("audio")
        else:
            data       = request.get_json(force=True) or {}
            accuracy   = float(data.get("accuracy") or 0)
            audio_file = None

        with torch.no_grad():
            if audio_file is not None:
                mel = audio_bytes_to_mel(audio_file.read())
            else:
                silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
                mel_np = librosa.feature.melspectrogram(
                    y=silent, sr=SAMPLE_RATE,
                    n_fft=400, hop_length=160, n_mels=80,
                )
                mel_np = librosa.power_to_db(mel_np, ref=np.max)
                mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            audio_feats    = encoder.encode_features(mel)
            phoneme_scores = phoneme_scores_from_features(audio_feats)

        # Combine encoder's phoneme confidence with the reported accuracy
        encoder_overall = (
            phoneme_scores["voicing"] * 0.30 +
            phoneme_scores["manner"]  * 0.40 +
            phoneme_scores["place"]   * 0.30
        )
        # Blend: 60% encoder signal, 40% reported accuracy
        overall = round(encoder_overall * 0.6 + accuracy * 0.4, 1)

        return jsonify({
            "overall":        overall,
            "phoneme_scores": phoneme_scores,
            "details": {
                "fluency":      round(phoneme_scores["voicing"], 1),
                "accuracy":     round(accuracy, 1),
                "articulation": round(encoder_overall, 1),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_models()
    # debug=False keeps Flask from loading models twice
    app.run(host="0.0.0.0", port=5000, debug=False)