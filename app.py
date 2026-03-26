from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_music():
    data = request.json

    style = data.get("style", "lofi")
    bpm = data.get("bpm", 120)
    duration = data.get("duration", 60)

    try:
        cmd = [
            sys.executable,
            "generate_beat.py",
            "--style", str(style),
            "--bpm", str(bpm),
            "--duration", str(duration)
        ]

        subprocess.run(cmd, check=True)

        return jsonify({
            "success": True,
            "file": "/static/final.wav"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)