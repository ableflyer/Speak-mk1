import os
import glob

librispeech_root = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/test-clean"

# Each chapter folder has a .trans.txt file like:
# 84-121550-0000 WORD1 WORD2 ...
# 84-121550-0001 WORD1 WORD2 ...

trans_files = glob.glob(
    os.path.join(librispeech_root, "**/*.trans.txt"), 
    recursive=True
)

for trans_file in trans_files:
    folder = os.path.dirname(trans_file)
    with open(trans_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # First token is the utterance ID, rest is the transcription
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            out_path = os.path.join(folder, utt_id + ".txt")
            with open(out_path, "w") as out:
                out.write(text)

print("Done — transcription files written.")