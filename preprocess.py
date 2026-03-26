# preprocess_chords.py
import os, json, math, argparse, random
from collections import Counter, defaultdict

import numpy as np
import pretty_midi


# -------------------- Chord templates --------------------
# Each chord type = set of pitch-class intervals from root
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

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_files_in(root):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                out.append(os.path.join(r, f))
    return sorted(out)


def estimate_global_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    # pretty_midi has estimate_tempo; good enough for MAESTRO-like files
    try:
        tempo = pm.estimate_tempo()
        if tempo and tempo > 0:
            return float(tempo)
    except Exception:
        pass
    return 120.0


def bar_windows(duration_sec: float, bpm: float):
    sec_per_beat = 60.0 / bpm
    sec_per_bar = 4.0 * sec_per_beat
    n_bars = max(1, int(duration_sec / sec_per_bar))
    for b in range(n_bars):
        start = b * sec_per_bar
        end = min(duration_sec, (b + 1) * sec_per_bar)
        yield b, start, end, sec_per_bar


def collect_pitch_classes(pm: pretty_midi.PrettyMIDI, start: float, end: float):
    pcs = []
    vel_sum = 0.0
    count = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= start or n.start >= end:
                continue
            pcs.append(n.pitch % 12)
            vel_sum += n.velocity
            count += 1
    return pcs, (vel_sum / count if count > 0 else 0.0), count


def score_chord(root_pc: int, chord_type: str, pc_counter: Counter):
    tmpl = CHORD_TEMPLATES[chord_type]
    score = 0.0
    total = sum(pc_counter.values()) + 1e-9

    # reward matches; penalize strong non-matching pitch classes
    for pc, c in pc_counter.items():
        interval = (pc - root_pc) % 12
        if interval in tmpl:
            score += 2.0 * (c / total)
        else:
            score -= 1.0 * (c / total)
    # small bias for richer chords
    score += 0.05 * len(tmpl)
    return score


def detect_chord_token(pcs):
    if not pcs:
        return "N"  # no chord / rest
    pc_counter = Counter(pcs)

    best = None
    best_score = -1e9
    for root in range(12):
        for ctype in CHORD_TEMPLATES.keys():
            s = score_chord(root, ctype, pc_counter)
            if s > best_score:
                best_score = s
                best = (root, ctype)

    root_pc, ctype = best
    return f"{NOTE_NAMES[root_pc]}:{ctype}"


def auto_style_label(avg_pitch, notes_per_bar, bpm):
    """
    Heuristic labels (works well enough for conditioning).
    You can tweak thresholds later, but this is good for a project demo.
    """
    # energetic: faster tempo or dense
    if bpm >= 140 or notes_per_bar >= 35:
        return "energetic"
    # aggressive: lower pitch and medium-high density
    if avg_pitch <= 55 and notes_per_bar >= 20:
        return "aggressive"
    # romantic: mid pitch + medium density
    if 55 < avg_pitch < 70 and 10 <= notes_per_bar < 22:
        return "romantic"
    # happy: brighter pitch and medium density
    if avg_pitch >= 70 and 12 <= notes_per_bar < 28:
        return "happy"
    # default lofi (sad/chill)
    return "lofi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_root", type=str, required=True, help="Folder containing MIDI files recursively")
    ap.add_argument("--out_dir", type=str, default="data_chords")
    ap.add_argument("--seq_len", type=int, default=16, help="bars per training sequence")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = midi_files_in(args.midi_root)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    print("Found MIDI files:", len(files))

    style_to_sequences = defaultdict(list)
    all_tokens = set(["N"])

    total_sequences = 0
    skipped = 0

    for i, path in enumerate(files):
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            skipped += 1
            continue

        bpm = estimate_global_tempo(pm)
        dur = float(pm.get_end_time())

        # Build chord tokens per bar + bar-level stats
        chord_tokens = []
        pitches_for_avg = []
        notes_for_density = []

        for b, start, end, sec_per_bar in bar_windows(dur, bpm):
            pcs, avg_vel, note_count = collect_pitch_classes(pm, start, end)

            # avg pitch for style
            if pcs:
                # approximate avg pitch using pitch classes not enough; use real pitches
                # We'll re-scan notes quickly in bar for avg pitch
                ps = []
                for inst in pm.instruments:
                    if inst.is_drum:
                        continue
                    for n in inst.notes:
                        if n.end <= start or n.start >= end:
                            continue
                        ps.append(n.pitch)
                if ps:
                    pitches_for_avg.append(float(np.mean(ps)))
            notes_for_density.append(note_count)

            tok = detect_chord_token(pcs)
            chord_tokens.append(tok)
            all_tokens.add(tok)

        if len(chord_tokens) < args.seq_len + 1:
            continue

        avg_pitch = float(np.mean(pitches_for_avg)) if pitches_for_avg else 60.0
        notes_per_bar = float(np.mean(notes_for_density)) if notes_for_density else 0.0
        style = auto_style_label(avg_pitch, notes_per_bar, bpm)

        # Slice into sequences of length seq_len -> predict next chord
        for s in range(0, len(chord_tokens) - (args.seq_len + 1), args.seq_len):
            chunk = chord_tokens[s:s + args.seq_len + 1]
            if len(chunk) < args.seq_len + 1:
                break
            style_to_sequences[style].append(chunk)
            total_sequences += 1

    print("Skipped MIDIs:", skipped)
    print("Total sequences:", total_sequences)
    print("Token vocab size (pre):", len(all_tokens))

    # Build vocab with STYLE tokens + chord tokens
    styles = ["lofi", "romantic", "happy", "energetic", "aggressive"]
    vocab = ["<PAD>", "<BOS>"] + [f"<STYLE_{s.upper()}>" for s in styles] + sorted(all_tokens)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    # Build arrays X, y
    seq_len = args.seq_len
    X_list = []
    y_list = []

    for style, seqs in style_to_sequences.items():
        style_tok = f"<STYLE_{style.upper()}>"
        for seq in seqs:
            # Input: [STYLE] + first seq_len chords
            x = [token_to_id[style_tok]] + [token_to_id[t] for t in seq[:seq_len]]
            # Target: next chord for each position? We do next-token for last position only OR full shift.
            # We'll do full shift (teacher forcing) so model learns transitions.
            # y will be next token for each input position (same length).
            # For position 0 (style), next should be first chord.
            y = [token_to_id[seq[0]]] + [token_to_id[t] for t in seq[1:seq_len+1]]

            X_list.append(x)
            y_list.append(y)

    X = np.array(X_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n = len(X)
    n_train = int(0.9 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print("X_train:", X_train.shape, "X_val:", X_val.shape)
    print("Vocab size:", len(vocab))

    # Save
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump({"vocab": vocab, "token_to_id": token_to_id, "id_to_token": id_to_token}, f)

    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()