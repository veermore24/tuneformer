# train_chords.py
import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, x, training=None):
        attn = self.att(x, x, use_causal_mask=True)
        attn = self.do1(attn, training=training)
        x1 = self.ln1(x + attn)

        ffn = self.ffn(x1)
        ffn = self.do2(ffn, training=training)
        return self.ln2(x1 + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_chords")
    ap.add_argument("--out_dir", type=str, default="models_chords")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))

    with open(os.path.join(args.data_dir, "vocab.json"), "r") as f:
        vocab_data = json.load(f)
    vocab = vocab_data["vocab"]
    vocab_size = len(vocab)

    seq_len = X_train.shape[1]  # style + chords (seq_len+1)

    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    x = PositionalEmbedding(seq_len, vocab_size, args.embed_dim)(inputs)
    x = TransformerBlock(args.embed_dim, args.num_heads, args.ff_dim)(x)
    x = TransformerBlock(args.embed_dim, args.num_heads, args.ff_dim)(x)
    x = layers.Dropout(0.1)(x)
    logits = layers.Dense(vocab_size)(x)

    model = keras.Model(inputs, logits)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "chord_model_best.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]

    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks
    )

    # Save final model too
    model.save(os.path.join(args.out_dir, "chord_model_final.keras"))
    # Save vocab
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab_data, f)

    print("Saved model + vocab to:", args.out_dir)


if __name__ == "__main__":
    main()