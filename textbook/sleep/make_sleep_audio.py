#!/usr/bin/env python3
"""Turn a chapter's sleep narration into a calm MP3 and a dark MP4 to fall asleep to.

The narration is a markdown file of plain paragraphs: a "# " title line (shown on the
video, not read aloud), paragraphs separated by blank lines, and "---" lines between
sections, which become longer pauses.

Usage (from the repo root, inside the uv venv):
    python textbook/sleep/make_sleep_audio.py textbook/hands_on_ml/ch01-*/sleep_narration.md
    python textbook/sleep/make_sleep_audio.py NARRATION.md --voice bm_george --pitch 0.93
    python textbook/sleep/make_sleep_audio.py --samples     # compare candidate voices
    python textbook/sleep/make_sleep_audio.py NARRATION.md --compare   # A/B the polish options
    python textbook/sleep/make_sleep_audio.py NARRATION.md --polish warm warm-room

Writes NARRATION.mp3 and NARRATION.mp4 next to the narration (both gitignored).
Voices come from Kokoro (https://huggingface.co/hexgrad/Kokoro-82M): am_ = American
male, bm_ = British male, af_/bf_ = female. --pitch below 1 lowers the voice without
changing its speed (0.93 is about one semitone deeper).
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

RATE = 24_000  # Kokoro's output sample rate
# Seconds of silence: between sentence chunks Kokoro splits a paragraph into, between
# paragraphs (kept close to a breath so the narration flows), and between sections.
PAUSES = {"chunk": 0.3, "paragraph": 0.6, "section": 2.5}
BACKGROUND = "0x0a0c10"  # near-black, slightly blue
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SAMPLE_DIR = Path(__file__).resolve().parent / "voice_samples"
SAMPLE_TEXT = (
    "Machine learning is the science, and partly the art, of programming computers so "
    "they can learn from data. Instead of telling the computer exactly what to do, step "
    "by step, you show it examples, and it gets better at a task by learning from them."
)
# (voice, pitch): the deeper, calmer male voices, plus lowered-pitch variants.
SAMPLE_VOICES = [
    ("am_michael", 1.0), ("am_onyx", 1.0), ("am_fenrir", 1.0),
    ("bm_george", 1.0), ("bm_lewis", 1.0),
    ("am_michael", 0.93), ("bm_george", 0.93),
]


def silence(seconds):
    return np.zeros(int(RATE * seconds), dtype=np.float32)


def parse(narration):
    """-> (title, sections), where each section is a list of paragraph strings."""
    title = None
    sections = []
    for chunk in re.split(r"^---\s*$", narration, flags=re.MULTILINE):
        paragraphs = []
        for block in re.split(r"\n\s*\n", chunk):
            text = " ".join(block.split())
            if not text:
                continue
            if text.startswith("# "):
                title = title or text[2:]
                continue
            paragraphs.append(text)
        if paragraphs:
            sections.append(paragraphs)
    return title, sections


class Voice:
    def __init__(self, voice, speed):
        # Kokoro picks pronunciation rules by accent: "a" American, "b" British.
        self.pipeline = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M",
                                  device="cuda" if torch.cuda.is_available() else "cpu")
        self.voice, self.speed = voice, speed

    def say(self, text):
        pieces = []
        for result in self.pipeline(text, voice=self.voice, speed=self.speed):
            if result.audio is not None:
                pieces += [result.audio.cpu().numpy().astype(np.float32), silence(PAUSES["chunk"])]
        # No chunk pause after the last chunk: the caller adds the paragraph pause.
        return np.concatenate(pieces[:-1]) if pieces else silence(0.1)


# Post-production that makes raw TTS sound more like a studio voice-over.
STUDIO = [
    "aresample=48000",
    "highpass=f=70",                                  # remove rumble below the voice
    "bass=g=3:f=140:w=0.7",                           # low-end warmth (low shelf)
    "equalizer=f=320:t=q:w=1.2:g=-2.5",               # less boxy/muddy
    "equalizer=f=2800:t=q:w=1.0:g=1",                 # a little clarity
    "deesser=i=0.5:m=0.5:f=0.5",                      # soften harsh "s" sounds
    "treble=g=-2:f=9000",                             # gentle, non-fatiguing top end
    "acompressor=threshold=-22dB:ratio=3:attack=15:release=200:makeup=2",  # even level
]
# A much heavier "close-mic broadcast" sound: big low end, scooped mids, dark top, firm
# compression and a little saturation.
WARM = [
    "aresample=48000",
    "highpass=f=60",
    "bass=g=8:f=110:w=0.8",                           # proximity-effect warmth
    "equalizer=f=400:t=q:w=1.0:g=-5",                 # scoop the boxy mids
    "equalizer=f=1800:t=q:w=1.2:g=-2",                # less nasal
    "deesser=i=0.7:m=0.6:f=0.5",
    "treble=g=-6:f=6000",                             # dark, soft top end
    "acompressor=threshold=-30dB:ratio=5:attack=8:release=250:makeup=4",
    "asoftclip=type=tanh:param=0.9",                  # gentle harmonic warmth
]
ROOM = ["aecho=0.85:0.8:23|41:0.07|0.05"]           # faint early reflections, not a hall
BOOTH = ["aecho=0.8:0.75:18|33|52:0.18|0.12|0.08"]  # clearly audible small-room sound
POLISH = {"none": [], "studio": STUDIO, "studio-room": STUDIO + ROOM,
          "warm": WARM, "warm-room": WARM + BOOTH}


def finish_audio(wav, mp3, pitch, polish="none"):
    """Pitch-shift (keeping the tempo), polish, normalize to a soft level, fade in, encode MP3."""
    filters = []
    if pitch != 1.0:
        filters += [f"asetrate={RATE * pitch:.0f}", f"aresample={RATE}", f"atempo={1 / pitch:.5f}"]
    filters += POLISH[polish]
    filters += ["loudnorm=I=-24:TP=-3:LRA=7", "afade=t=in:d=2"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", ",".join(filters),
                    "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "3", str(mp3)], check=True)


def make_video(mp3, mp4, title):
    video = f"color=c={BACKGROUND}:s=1280x720:r=1"
    text = (title or "").replace("\\", "").replace(":", r"\:").replace("'", "’")
    draw = (f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white@0.22:fontsize=30:"
            f"x=(w-text_w)/2:y=(h-text_h)/2") if title else "null"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", video, "-i", str(mp3),
                    "-vf", draw, "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(mp4)], check=True)


def narrate(path, voice, speed, pitch, polishes, video):
    title, sections = parse(path.read_text())
    speaker = Voice(voice, speed)
    audio = [silence(2.0)]
    total = sum(len(s) for s in sections)
    done = 0
    for section in sections:
        for paragraph in section:
            audio += [speaker.say(paragraph), silence(PAUSES["paragraph"])]
            done += 1
            print(f"\r  paragraph {done}/{total}", end="", flush=True)
        audio.append(silence(PAUSES["section"]))
    print()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "narration.wav"
        sf.write(wav, np.concatenate(audio), RATE)
        for polish in polishes:  # one synthesis, one output per finish
            stem = path.with_suffix("") if polish == "none" else path.with_suffix(f".{polish}")
            mp3 = stem.with_suffix(stem.suffix + ".mp3") if polish != "none" else path.with_suffix(".mp3")
            finish_audio(wav, mp3, pitch, polish)
            print(f"wrote {mp3} ({sf.info(mp3).duration / 60:.0f} min)")
            if video:
                mp4 = mp3.with_suffix(".mp4")
                make_video(mp3, mp4, title)
                print(f"wrote {mp4}")


def samples(speed):
    SAMPLE_DIR.mkdir(exist_ok=True)
    files = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (voice, pitch) in enumerate(SAMPLE_VOICES, start=1):
            speaker = Voice(voice, speed)
            clip = np.concatenate([speaker.say(f"Voice {i}."), silence(0.8),
                                   speaker.say(SAMPLE_TEXT), silence(2.0)])
            wav = Path(tmp) / f"{i}.wav"
            sf.write(wav, clip, RATE)
            name = f"{i}_{voice}" + ("" if pitch == 1.0 else f"_pitch{pitch}")
            mp3 = SAMPLE_DIR / f"{name}.mp3"
            finish_audio(wav, mp3, pitch)
            files.append(mp3)
            print(f"  {i}. {voice}" + ("" if pitch == 1.0 else f", pitch {pitch} (deeper)"))
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{f}'\n" for f in files))
        combined = SAMPLE_DIR / "all_voices.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(combined)], check=True)
    print(f"wrote {SAMPLE_DIR}/ (one file per voice, plus all_voices.mp3)")


# (label, pitch, polish) variants for --compare
COMPARE = [("raw", 1.0, "none"), ("studio", 1.0, "studio"), ("warm", 1.0, "warm"),
           ("warm-room", 1.0, "warm-room"), ("warm-room-deeper", 0.95, "warm-room")]


def compare(path, voice, speed, paragraphs=5):
    """Synthesize the narration's first few paragraphs once, then apply each polish variant."""
    _, sections = parse(path.read_text())
    text = [p for section in sections for p in section][:paragraphs]
    speaker = Voice(voice, speed)
    clip = np.concatenate([np.concatenate([speaker.say(p), silence(PAUSES["paragraph"])]) for p in text])
    SAMPLE_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "excerpt.wav"
        sf.write(wav, clip, RATE)
        for i, (label, pitch, polish) in enumerate(COMPARE, start=1):
            mp3 = SAMPLE_DIR / f"compare_{voice}_{i}_{label}.mp3"
            finish_audio(wav, mp3, pitch, polish)
            print(f"wrote {mp3.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("narration", nargs="?", type=Path)
    parser.add_argument("--voice", default="am_onyx")
    parser.add_argument("--speed", type=float, default=0.85, help="1.0 is Kokoro's normal pace")
    parser.add_argument("--pitch", type=float, default=1.0, help="below 1 = deeper voice")
    parser.add_argument("--polish", choices=list(POLISH), nargs="+", default=["none"],
                        help="studio-style post-processing; give several to get one file each "
                             "(named NARRATION.<polish>.mp3)")
    parser.add_argument("--compare", action="store_true",
                        help="render a short excerpt with each polish variant, to A/B them")
    parser.add_argument("--no-video", action="store_true", help="only write the MP3")
    parser.add_argument("--samples", action="store_true", help="render the voice sampler")
    args = parser.parse_args()
    if args.samples:
        samples(args.speed)
    elif args.narration and args.compare:
        compare(args.narration, args.voice, args.speed)
    elif args.narration:
        narrate(args.narration, args.voice, args.speed, args.pitch, args.polish, not args.no_video)
    else:
        parser.error("give a narration file, or --samples")


if __name__ == "__main__":
    sys.exit(main())
