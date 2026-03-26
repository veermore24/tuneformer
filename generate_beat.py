import os
import argparse
import pickle
import random
import numpy as np
import soundfile as sf
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# =====================================================
# Custom Layers (must match trained model)
# =====================================================

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)
        self.sequence_length = sequence_length

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs, use_causal_mask=True)
        attn = self.dropout1(attn, training=training)
        x = self.layernorm1(inputs + attn)
        ffn = self.ffn(x)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(x + ffn)

# =====================================================
# AI Melody
# =====================================================

def sample_topk(probs, k=25):
    top_idx = np.argsort(probs)[-k:]
    top_probs = probs[top_idx]
    top_probs /= np.sum(top_probs)
    return int(np.random.choice(top_idx, p=top_probs))


def generate_tokens(model, seed, seq_len=128, gen_len=1400):
    generated = list(seed.tolist())
    for _ in range(gen_len):
        x = np.array(generated[-seq_len:]).reshape(1, seq_len)
        preds = model.predict(x, verbose=0)[0]
        generated.append(sample_topk(preds))
    return generated[seq_len:]

# =====================================================
# Chord Progressions
# =====================================================

def progression(style):
    if style == "happy":
        return [60, 65, 67, 64]
    elif style == "romantic":
        return [57, 60, 65, 64]
    elif style == "aggressive":
        return [48, 50, 53, 55]
    elif style == "energetic":
        return [60, 67, 69, 65]
    else:
        return [57, 53, 60, 55]

# =====================================================
# Piano Generation
# =====================================================

def create_piano(token_ids, int_to_note, bpm, duration, style, instrument):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    program = pretty_midi.instrument_name_to_program(instrument)
    inst = pretty_midi.Instrument(program=program)

    bar_len = (60/bpm)*4
    prog = progression(style)

    if style == "lofi":
        melody_vel, chord_vel = 70, 55
    elif style == "aggressive":
        melody_vel, chord_vel = 118, 100
    elif style == "happy":
        melody_vel, chord_vel = 105, 90
    elif style == "romantic":
        melody_vel, chord_vel = 95, 80
    else:
        melody_vel, chord_vel = 110, 95

    t = 0
    last_pitch = None

    for tid in token_ids:
        pitch, dur = int_to_note[int(tid)].split("_")
        pitch = int(pitch)
        dur = max(0.25, float(dur))

        if t + dur > duration:
            break

        if last_pitch and abs(pitch-last_pitch) > 12:
            pitch = last_pitch + (7 if pitch > last_pitch else -7)

        inst.notes.append(pretty_midi.Note(
            velocity=melody_vel + random.randint(-5,5),
            pitch=pitch,
            start=t,
            end=t+dur
        ))

        bar = int(t // bar_len)
        root = prog[bar % len(prog)]
        chord = [root, root+4, root+7]

        for c in chord:
            inst.notes.append(pretty_midi.Note(
                velocity=chord_vel,
                pitch=c,
                start=bar*bar_len,
                end=(bar+1)*bar_len
            ))

        t += dur
        last_pitch = pitch

    pm.instruments.append(inst)
    pm.write("outputs/piano.mid")

    # synth directly in python (NO fluidsynth dependency)
    audio = pm.synthesize(fs=44100)
    sf.write("outputs/piano.wav", audio, 44100)
    return audio

# =====================================================
# Drum Engine (Trapkit WAV based)
# =====================================================

def load_sample(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio

def place_sample(track, sample, idx):
    end = idx + len(sample)
    if idx >= len(track):
        return
    if end > len(track):
        sample = sample[:len(track)-idx]
    track[idx:idx+len(sample)] += sample

def create_drums(style, bpm, duration, kit_path):
    sr = 44100
    total = int(duration*sr)
    mix = np.zeros(total)

    kick = load_sample(os.path.join(kit_path,"kick.wav"))
    snare_file = "snare.wav" if os.path.exists(os.path.join(kit_path,"snare.wav")) else "clap.wav"
    snare = load_sample(os.path.join(kit_path,snare_file))
    hat = load_sample(os.path.join(kit_path,"hat.wav"))

    has_808 = os.path.exists(os.path.join(kit_path,"808.wav"))
    if has_808:
        bass808 = load_sample(os.path.join(kit_path,"808.wav"))

    step = (60/bpm)/2
    steps = int(duration/step)

    drum_start = int((4*4)/(2))  # 4 bar delay

    for i in range(steps):
        if i < drum_start:
            continue

        t = i*step
        idx = int(t*sr)

        # HAPPY dancehall exact
        if style == "happy":
            if i%8 in [0,4]:
                place_sample(mix,kick,idx)
            if i%8 in [3,6]:
                place_sample(mix,snare,idx)

        elif style in ["romantic","lofi"]:
            if i%8==0:
                place_sample(mix,kick,idx)
            if i%8==4:
                place_sample(mix,snare,idx)

        elif style=="aggressive":
            if i%8 in [0,2,5]:
                place_sample(mix,kick,idx)
            if i%8==4:
                place_sample(mix,snare,idx)

        else:
            if i%4==0:
                place_sample(mix,kick,idx)
            if i%4==2:
                place_sample(mix,snare,idx)

        place_sample(mix,hat*0.6,idx)

        if has_808 and i%8==0:
            place_sample(mix,bass808*0.8,idx)

    mix = mix/np.max(np.abs(mix))*0.9
    sf.write("outputs/drums.wav",mix,sr)
    return mix

# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="lofi",
                        choices=["aggressive","lofi","happy","energetic","romantic"])
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--instrument", default="Electric Piano 1")
    parser.add_argument("--kit", default="drum_kits/trapkit")
    parser.add_argument("--model", default="models/tuneformer_piano.h5")
    parser.add_argument("--vocab", default="data/vocab.pkl")
    parser.add_argument("--xval", default="data/X_val.npy")

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    with open(args.vocab,"rb") as f:
        pitchnames = pickle.load(f)
    int_to_note = {i:n for i,n in enumerate(pitchnames)}

    X_val = np.load(args.xval)
    seed = X_val[random.randint(0,len(X_val)-1)]

    model = load_model(args.model,
        custom_objects={"PositionalEmbedding":PositionalEmbedding,
                        "TransformerBlock":TransformerBlock},
        compile=False)

    tokens = generate_tokens(model, seed)

    piano_audio = create_piano(tokens, int_to_note,
                               args.bpm, args.duration,
                               args.style, args.instrument)

    drums_audio = create_drums(args.style, args.bpm,
                               args.duration, args.kit)

    n = max(len(piano_audio), len(drums_audio))
    mix = np.zeros(n)
    mix[:len(piano_audio)] += piano_audio*1.05
    mix[:len(drums_audio)] += drums_audio

    mix = mix/np.max(np.abs(mix))*0.95
    sf.write("static/final.wav", mix, 44100)

    print("🔥 Beat Ready → static/final.wav")


if __name__=="__main__":
    main()