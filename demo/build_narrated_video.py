"""Build a polished, narrated ModelMint demo video.

Inputs (produced by record_demo.py):
  demo/timeline.json      frame timestamps + narration events
  demo/frames/*.jpg       screencast frames
  demo/screenshots/*.png  feature screenshots

Outputs:
  demo/modelmint-demo.mp4       narrated video (voiceover + burned captions + soft subs)
  demo/modelmint-demo.srt       transcript / soft subtitle track
  demo/modelmint-demo.ass       styled burned captions
  demo/transcript.md            readable walkthrough transcript

Voiceover is synthesized offline with ffmpeg's libflite filter.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
FRAMES_DIR = os.path.join(DEMO, "frames")
AUDIO_DIR = os.path.join(DEMO, "audio")
BUILD_DIR = os.path.join(DEMO, ".build")
SLIDES_DIR = os.path.join(BUILD_DIR, "slides")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VOICE = "slt"

TITLES = {
    "01-dashboard-empty": "01 - Model Registry",
    "02-api-keys-locked": "02 - Scoped API Keys (Locked)",
    "03-api-keys-unlocked": "03 - Admin Unlock",
    "04-key-created-modal": "04 - Creating a Scoped Key",
    "05-keys-and-audit": "05 - Key Scopes and Audit",
    "06-registry-uploaded": "06 - Upload and Sandbox Validation",
    "07-model-detail": "07 - Detected Schema",
    "08-try-it": "08 - Task-Aware Try It",
    "09-predict-result": "09 - Single Prediction",
    "10-batch-result": "10 - Batch Prediction",
    "11-request-examples": "11 - Copy-Ready Integrations",
    "12-monitoring": "12 - Monitoring and Metrics",
    "13-version-uploaded": "13 - Upload a New Version",
    "14-version-promoted": "14 - Promote a Release",
    "15-traffic-split": "15 - A/B Traffic Split",
    "16-drift": "16 - Input Drift Detection",
    "17-formats-api": "17 - Supported Formats API",
    "18-prometheus-metrics": "18 - Prometheus Metrics",
    "19-auth-required": "19 - Authentication Required",
    "19b-api-docs": "19b - Interactive API Docs",
    "20-audit-log": "20 - Security Audit Log",
    "21-key-rotated": "21 - Key Rotation",
    "22-key-revoked": "22 - Key Revocation",
    "23-model-deleted": "23 - Model Deletion",
    "24-final-audit": "24 - Final Audit Trail",
}

NARRATION = {
    "01-dashboard-empty": (
        "This is the ModelMint registry. It is empty right now, ready for the first "
        "model upload. From here you can search, filter, inspect, and delete models."
    ),
    "02-api-keys-locked": (
        "The API keys page starts locked. Administrative actions such as creating or "
        "revoking keys require an admin key, so access is never open by default."
    ),
    "03-api-keys-unlocked": (
        "Unlocking with the bootstrap admin key reveals scoped key management, "
        "per-key usage, and the security audit log. The key stays in this browser tab only."
    ),
    "04-key-created-modal": (
        "Now we create a scoped key for a client. The plaintext secret is shown exactly "
        "once, and only an HMAC hash is stored on the server. If it is lost, you rotate it."
    ),
    "05-keys-and-audit": (
        "The client key has read, write, and monitoring scopes, plus predict. The audit "
        "log immediately records the key creation with the exact scopes granted."
    ),
    "06-registry-uploaded": (
        "Next, upload a scikit-learn pickle. ModelMint detects the framework, validates "
        "the artifact in a sandbox, stores a SHA-256 checksum, and automatically creates "
        "a live inference endpoint."
    ),
    "07-model-detail": (
        "The model detail view shows the detected framework, task type, schema confidence, "
        "feature count, class count, and sandbox load time. A synthetic input test confirms "
        "the model is servable."
    ),
    "08-try-it": (
        "The Try It panel generates a task-aware example request. For a tabular model it "
        "sends a features array, so you can test the endpoint without writing any code."
    ),
    "09-predict-result": (
        "A single prediction returns in milliseconds from a persistent sandboxed worker. "
        "The API process never loads untrusted model code directly."
    ),
    "10-batch-result": (
        "Batch mode accepts an array of instances and returns one prediction per row. "
        "This is useful for scoring small datasets in a single request."
    ),
    "11-request-examples": (
        "Copy-ready curl and Python snippets make integration immediate. They use your "
        "API key and the model's generated endpoint."
    ),
    "12-monitoring": (
        "Monitoring aggregates live request counts, success rate, average and p95 latency, "
        "a latency chart, and the most recent inference logs."
    ),
    "13-version-uploaded": (
        "Upload a new version into the same model family. ModelMint validates it in the "
        "sandbox and keeps it inactive, so production traffic is untouched until promotion."
    ),
    "14-version-promoted": (
        "Promoting version two makes it the active release. Version one remains servable, "
        "so a rollback is one click away if the new release misbehaves."
    ),
    "15-traffic-split": (
        "Traffic weights turn a release into a canary. Here version two and version one "
        "each receive half of the requests through the stable family endpoint."
    ),
    "16-drift": (
        "Drift detection compares the recent request window with the baseline and flags "
        "features whose distribution has shifted, without ever storing raw request bodies."
    ),
    "17-formats-api": (
        "The formats endpoint reports every supported framework, its file extensions, "
        "task types, and whether its optional package is installed in this deployment."
    ),
    "18-prometheus-metrics": (
        "Prometheus metrics expose model counts by status, inference requests, latency "
        "totals, and active API keys, ready to scrape into your observability stack."
    ),
    "19-auth-required": (
        "Security is enforced at the API layer. Calling predict without an API key is "
        "rejected with 401 Unauthorized, before any model code runs."
    ),
    "19b-api-docs": (
        "Interactive OpenAPI documentation describes every endpoint and request schema "
        "for client developers."
    ),
    "20-audit-log": (
        "The audit log tracks key creation, model uploads, promotions, rotations, "
        "revocations, and deletions. Every security-sensitive action is attributable "
        "and timestamped."
    ),
    "21-key-rotated": (
        "Rotating a key immediately invalidates the old secret and issues a new plaintext "
        "key, shown once. Existing integrations must switch to the new value."
    ),
    "22-key-revoked": (
        "Revoking a key disables it permanently. The key remains in the table with a "
        "revoked status, preserving history for audits."
    ),
    "23-model-deleted": (
        "Deleting a model removes its stored artifact and inference endpoint. The registry "
        "returns to an empty state, ready for the next upload."
    ),
    "24-final-audit": (
        "Every action remains visible in the audit trail. ModelMint is ready to mint, "
        "serve, and govern models with scoped access and full observability."
    ),
}

INTRO_TITLE = "ModelMint"
INTRO_SUBTITLE = "Model-as-a-Service"
INTRO_TAGLINE = "Feature Walkthrough"
INTRO_NARRATION = (
    "Welcome to ModelMint, a secure model as a service platform. In this walkthrough, "
    "we will upload a trained model, serve single and batch predictions, create scoped "
    "API keys, promote new versions with canary traffic, detect input drift, and monitor "
    "every request end to end."
)

OUTRO_TITLE = "Mint. Serve. Govern."
OUTRO_SUBTITLE = "ModelMint"
OUTRO_NARRATION = (
    "That is ModelMint. Upload a supported model, serve it instantly, promote new "
    "versions with canary traffic, watch for input drift, and govern every action with "
    "scoped keys, sandboxed execution, and a complete audit trail. Thanks for watching."
)

for d in (AUDIO_DIR, BUILD_DIR, SLIDES_DIR):
    os.makedirs(d, exist_ok=True)


def run(cmd, **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


def probe_duration(path):
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ]
    )
    return float(out.strip())


def tts(text, name):
    """Synthesize text to a WAV and return (path, duration)."""
    txt_path = os.path.join(BUILD_DIR, f"{name}.txt")
    wav_path = os.path.join(AUDIO_DIR, f"{name}.wav")
    if os.path.exists(wav_path) and os.path.exists(txt_path):
        with open(txt_path) as f:
            if f.read() == text:
                return wav_path, probe_duration(wav_path)
    with open(txt_path, "w") as f:
        f.write(text)
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"flite=textfile={txt_path}:voice={VOICE}",
            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path, probe_duration(wav_path)


def pad_audio(src, duration, dst):
    run(
        [
            "ffmpeg", "-y", "-i", src, "-af", "apad", "-t", f"{duration:.3f}",
            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", dst,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_card(path, lines, bg="0x10231d"):
    """Render a 1440x900 title card as JPEG."""
    filters = [f"drawbox=x=0:y=0:w=iw:h=ih:color={bg}:t=fill"]
    filters.append(
        "drawbox=x=120:y=430:w=90:h=5:color=0x7fe0c0:t=fill"
    )
    y = 250
    for text, size, color, bold in lines:
        font = FONT_BOLD if bold else FONT
        safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters.append(
            f"drawtext=fontfile={font}:text='{safe}':fontcolor={color}:"
            f"fontsize={size}:x=120:y={y}"
        )
        y += int(size * 1.5)
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x10231d:s=1440x900:d=1",
            "-vf", ",".join(filters), "-frames:v", "1", "-q:v", "2", path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ass_time(t):
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time(t):
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_srt(text, width=92):
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def main():
    with open(os.path.join(DEMO, "timeline.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]
    events = meta["events"]

    # Build segment descriptors in playback order: intro, events, outro.
    segments = []

    intro_audio, intro_dur = tts(INTRO_NARRATION, "00_intro")
    intro_target = max(5.0, intro_dur + 1.0)
    segments.append(
        {
            "kind": "card",
            "name": "00-intro",
            "title": "ModelMint - Feature Walkthrough",
            "narration": INTRO_NARRATION,
            "audio": intro_audio,
            "audio_dur": intro_dur,
            "target": intro_target,
        }
    )

    # Attach frames to events. Frames before the first event belong to it.
    for i, event in enumerate(events):
        start = 0 if i == 0 else event["frame_index"]
        end = events[i + 1]["frame_index"] if i + 1 < len(events) else len(frames)
        start = min(start, len(frames))
        end = min(max(end, start), len(frames))
        seg_frames = frames[start:end]
        original = 0.0
        durations = []
        for j, frame in enumerate(seg_frames):
            if j + 1 < len(seg_frames):
                d = seg_frames[j + 1]["t"] - frame["t"]
            else:
                d = 1.5
            d = max(0.05, min(d, 5.0))
            durations.append(d)
            original += d
        name = event["name"]
        narration = NARRATION.get(name, event["narration"])
        audio, audio_dur = tts(narration, f"{i + 1:02d}_{name}")
        target = max(original, audio_dur + 0.8, 2.8)
        segments.append(
            {
                "kind": "event",
                "name": name,
                "title": TITLES.get(name, name),
                "narration": narration,
                "audio": audio,
                "audio_dur": audio_dur,
                "target": target,
                "frames": seg_frames,
                "durations": durations,
                "original": original,
                "screenshot": os.path.join(DEMO, "screenshots", f"{name}.png"),
            }
        )

    outro_audio, outro_dur = tts(OUTRO_NARRATION, "99_outro")
    outro_target = max(5.0, outro_dur + 1.0)
    segments.append(
        {
            "kind": "card",
            "name": "99-outro",
            "title": "Mint. Serve. Govern.",
            "narration": OUTRO_NARRATION,
            "audio": outro_audio,
            "audio_dur": outro_dur,
            "target": outro_target,
        }
    )

    # Convert screenshots to a common JPEG format for the concat demuxer.
    for seg in segments:
        if seg["kind"] != "event":
            continue
        slide = os.path.join(SLIDES_DIR, f"{seg['name']}.jpg")
        run(
            [
                "ffmpeg", "-y", "-i", seg["screenshot"],
                "-vf", "scale=1440:900,setsar=1", "-pix_fmt", "yuvj420p",
                "-q:v", "2", slide,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        seg["slide"] = slide

    # Render title cards.
    intro_jpg = os.path.join(BUILD_DIR, "intro.jpg")
    outro_jpg = os.path.join(BUILD_DIR, "outro.jpg")
    make_card(
        intro_jpg,
        [
            (INTRO_TITLE, 96, "0x7fe0c0", True),
            (INTRO_SUBTITLE, 40, "0xffffff", False),
            (INTRO_TAGLINE, 28, "0x9fb3ad", False),
        ],
    )
    make_card(
        outro_jpg,
        [
            (OUTRO_TITLE, 72, "0x7fe0c0", True),
            (OUTRO_SUBTITLE, 36, "0xffffff", False),
        ],
    )

    # Video concat list and audio pad/concat lists.
    video_lines = []
    audio_lines = []
    cursor = 0.0
    timeline = []
    for seg in segments:
        if seg["kind"] == "card":
            image = intro_jpg if seg["name"] == "00-intro" else outro_jpg
            video_lines.append(f"file '{image}'")
            video_lines.append(f"duration {seg['target']:.3f}")
        else:
            # A short live motion beat from the screencast, then freeze on the
            # crisp screenshot so the visual always matches the narration.
            motion = min(0.9, seg["target"])
            if seg["frames"] and motion > 0:
                first = os.path.join(DEMO, seg["frames"][0]["path"])
                video_lines.append(f"file '{first}'")
                video_lines.append(f"duration {motion:.3f}")
                remaining = seg["target"] - motion
            else:
                remaining = seg["target"]
            if remaining > 0.01:
                video_lines.append(f"file '{seg['slide']}'")
                video_lines.append(f"duration {remaining:.3f}")

        padded = os.path.join(AUDIO_DIR, f"{seg['name']}_padded.wav")
        pad_audio(seg["audio"], seg["target"], padded)
        audio_lines.append(f"file '{padded}'")

        seg["start"] = cursor
        cursor += seg["target"]
        seg["end"] = cursor
        timeline.append(seg)
        print(
            f"{seg['name']:<24} audio={seg['audio_dur']:5.2f}s "
            f"segment={seg['target']:5.2f}s start={seg['start']:6.2f}s"
        )

    total = cursor
    print(f"total duration: {total:.2f}s")

    # Concat lists need the last file repeated.
    video_concat = os.path.join(BUILD_DIR, "video.txt")
    with open(video_concat, "w") as f:
        f.write("\n".join(video_lines) + "\n")
        last = video_lines[-2]
        f.write(last + "\n")

    audio_concat = os.path.join(BUILD_DIR, "audio.txt")
    with open(audio_concat, "w") as f:
        f.write("\n".join(audio_lines) + "\n")

    voiceover = os.path.join(BUILD_DIR, "voiceover.wav")
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", audio_concat,
            "-c:a", "pcm_s16le", voiceover,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    silent = os.path.join(BUILD_DIR, "silent.mp4")
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_concat,
            "-vf", "scale=1440:900:force_original_aspect_ratio=decrease,"
            "pad=1440:900:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps=20",
            "-t", f"{total:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", silent,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # SRT (soft subs + transcript) and ASS (styled burned captions).
    srt_path = os.path.join(DEMO, "modelmint-demo.srt")
    ass_path = os.path.join(DEMO, "modelmint-demo.ass")
    with open(srt_path, "w") as srt, open(ass_path, "w") as ass:
        ass.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1440\n"
            "PlayResY: 900\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n"
            "Style: Caption,DejaVu Sans,22,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,80,80,30,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text\n"
        )
        for i, seg in enumerate(timeline, start=1):
            start, end = seg["start"], seg["end"]
            srt.write(
                f"{i}\n{srt_time(start)} --> {srt_time(end)}\n"
                f"{seg['title']}\n{wrap_srt(seg['narration'])}\n\n"
            )
            title = seg["title"].replace("{", "(").replace("}", ")")
            narration = seg["narration"].replace("{", "(").replace("}", ")")
            text = (
                "{\\b1\\fs28\\c&H00C0E07F&}" + title + "{\\r\\fs22\\c&H00FFFFFF&}"
                "\\N" + narration
            )
            ass.write(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{text}\n"
            )

    # Readable transcript.
    with open(os.path.join(DEMO, "transcript.md"), "w") as f:
        f.write("# ModelMint Feature Walkthrough\n\n")
        f.write(f"Total runtime: {int(total // 60)}m {int(total % 60)}s\n\n")
        f.write("| Time | Feature | Narration |\n|---|---|---|\n")
        for seg in timeline:
            mm = int(seg["start"] // 60)
            ss = int(seg["start"] % 60)
            f.write(
                f"| {mm:02d}:{ss:02d} | {seg['title']} | {seg['narration']} |\n"
            )

    out = os.path.join(DEMO, "modelmint-demo.mp4")
    run(
        [
            "ffmpeg", "-y", "-i", silent, "-i", voiceover, "-i", srt_path,
            "-vf",
            "drawbox=x=0:y=768:w=1440:h=132:color=0x0b1512@0.96:t=fill,"
            "drawbox=x=0:y=768:w=1440:h=2:color=0x7fe0c0@0.95:t=fill,"
            f"ass=filename={ass_path}",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:0", "title=English", "-shortest",
            "-movflags", "+faststart", out,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"\nvideo      -> {out}")
    print(f"subtitles  -> {srt_path}")
    print(f"transcript -> {os.path.join(DEMO, 'transcript.md')}")


if __name__ == "__main__":
    main()
