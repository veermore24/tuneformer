# preprocess_chords.py
# Extract chord tokens per bar from MIDI files and build dataset for AI chord model

import os
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pretty_midi


# ---------------- Chord templates ----------------
CHORD_TEMPLATES = {
    "maj":   {0, 4, 7},
    "min":   {0, 3, 7},
    "maj7":  {0, 4, 7, 11},
    "min7":  {0, 3, 7, 10},
    "dom7":  {0, 4, 7, 10},
    "sus2":  {0, 2, 7},
    "sus4":  {0, 5, 7},
    "dim":   {0, 3, 6},
    "aug":   {0, 4, 8},
    "power": {0, 7},
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


# ---------------- Utility ----------------
def get_midi_files(root):
    files = []
    for r, _, f_list in os.walk(root):
        for f in f_list:
            if f.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(r, f))
    return sorted(files)


def estimate_tempo(pm):
    try:
        t = pm.estimate_tempo()
        if t > 0:
            return float(t)
    except:
        pass
    return 120.0


def collect_pitch_classes(pm, start, end):
    pcs = []
    pitches = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= start or n.start >= end:
                continue
            pcs.append(n.pitch % 12)
            pitches.append(n.pitch)

    return pcs, pitches


def detect_chord(pcs):
    if not pcs:
        return "N"

    counter = Counter(pcs)
    best_score = -1e9
    best = None

    for root in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            score = 0
            for pc, count in counter.items():
                interval = (pc - root) % 12
                if interval in intervals:
                    score += count * 2
                else:
                    score -= count
            if score > best_score:
                best_score = score
                best = (root, chord_type)

    if best is None:
        return "N"

    root, chord_type = best
    return f"{NOTE_NAMES[root]}:{chord_type}"


def auto_style(avg_pitch, note_density, bpm):
    if bpm >= 140 or note_density > 30:
        return "energetic"
    if avg_pitch < 55 and note_density > 20:
        return "aggressive"
    if avg_pitch > 70:
        return "happy"
    if 55 <= avg_pitch <= 70:
        return "romantic"
    return "lofi"


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_root", required=True)
    ap.add_argument("--out_dir", default="data_chords")
    ap.add_argument("--seq_len", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = get_midi_files(args.midi_root)
    print("Found MIDI files:", len(files))

    style_sequences = defaultdict(list)
    vocab = set(["N"])

    for path in files:
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except:
            continue

        bpm = estimate_tempo(pm)
        duration = pm.get_end_time()
        sec_per_bar = (60.0 / bpm) * 4
        total_bars = int(duration / sec_per_bar)

        chords = []
        avg_pitches = []
        densities = []

        for b in range(total_bars):
            start = b * sec_per_bar
            end = (b + 1) * sec_per_bar

            pcs, pitches = collect_pitch_classes(pm, start, end)
            chord = detect_chord(pcs)
            chords.append(chord)
            vocab.add(chord)

            if pitches:
                avg_pitches.append(np.mean(pitches))
                densities.append(len(pitches))

        if len(chords) < args.seq_len + 1:
            continue

        avg_pitch = np.mean(avg_pitches) if avg_pitches else 60
        density = np.mean(densities) if densities else 0
        style = auto_style(avg_pitch, density, bpm)

        for i in range(len(chords) - args.seq_len):
            seq = chords[i:i + args.seq_len + 1]
            style_sequences[style].append(seq)

    styles = ["lofi", "romantic", "happy", "energetic", "aggressive"]
    vocab = ["<PAD>", "<BOS>"] + \
            [f"<STYLE_{s.upper()}>" for s in styles] + sorted(vocab)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    X, y = [], []

    for style, sequences in style_sequences.items():
        style_token = f"<STYLE_{style.upper()}>"
        for seq in sequences:
            x = [token_to_id[style_token]] + \
                [token_to_id[t] for t in seq[:-1]]
            target = [token_to_id[t] for t in seq]
            X.append(x)
            y.append(target)

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Vocab size:", len(vocab))

    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)

    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump({
            "vocab": vocab,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token
        }, f)

    print("✅ Dataset saved to", args.out_dir)


if __name__ == "__main__":
    main()
