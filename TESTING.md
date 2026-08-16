# Release Testing Checklist

This checklist is intended for every public release of `autodub-local`.
It separates fast regression checks from slower end-to-end checks because
some local AI stages can take a long time on older CPU-only machines.

## Fast Checks

Run these after every script change:

```bash
/bin/bash -n autodub_local.sh
python3 - <<'PY'
from pathlib import Path
script = Path("autodub_local.sh").read_text(encoding="utf-8")
start_marker = "cat > \"$PY_SCRIPT\" <<'PY'\n"
end_marker = "\nPY\nchmod +x \"$PY_SCRIPT\""
start = script.index(start_marker) + len(start_marker)
end = script.index(end_marker, start)
Path("/tmp/autodub_worker_check.py").write_text(script[start:end], encoding="utf-8")
PY
python3 -m py_compile /tmp/autodub_worker_check.py
/bin/bash ./autodub_local.sh --help
/bin/bash ./autodub_local.sh clean
```

For `clean`, cancel at the prompt unless the test intentionally validates deletion
against a disposable work folder.

## Voice Catalog Checks

Run at least one voice listing for each non-cloning TTS engine touched by a release:

```bash
./autodub_local.sh --target-lang it --tts-engine edge --list-tts-voices
./autodub_local.sh --target-lang it --tts-engine kokoro --list-tts-voices
```

When Edge TTS logic changes, also generate a small voice sample:

```bash
./autodub_local.sh \
  --target-lang it \
  --tts-engine edge \
  --sample-tts-voices \
  --sample-text "Questo è un test con free software e Georgia Tech."
```

## Translation-Only Checks

Use `--stop-after-translation` before full audio generation. This validates ASR,
diarization, translation, and LLM adaptation while avoiding a full TTS pass.

Cloud-assisted smoke test:

```bash
./autodub_local.sh \
  --input test.mp4 \
  --output .autodub_local/tmp/test_cloud_unused.mp4 \
  --source-lang en \
  --target-lang it \
  --only-cloud \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural \
  --stop-after-translation
```

Fully local smoke test:

```bash
./autodub_local.sh \
  --input test.mp4 \
  --output .autodub_local/tmp/test_local_unused.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method local \
  --tts-engine kokoro \
  --asr-backend local \
  --diarization-backend local \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=if_sara,SPEAKER_01=im_nicola \
  --stop-after-translation
```

Inspect the generated `test.translated.it.json` for:

- missing or untranslated text
- duplicated text across adjacent utterances
- over-aggressive LLM shortening
- speaker switches that do not match the video
- unnatural utterance cuts that leave an incomplete clause in one segment and the continuation in the next

## Full End-To-End Checks

Before a GitHub release, regenerate both reference outputs from a clean per-input
work folder and listen to the result:

```bash
./autodub_local.sh clean
```

Cloud-assisted reference run:

```bash
/usr/bin/time -p ./autodub_local.sh \
  --input test.mp4 \
  --output test_IT_only_cloud_edge_dub.mp4 \
  --source-lang en \
  --target-lang it \
  --only-cloud \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
```

Fully local reference run:

```bash
/usr/bin/time -p ./autodub_local.sh \
  --input test.mp4 \
  --output test_IT_all_local_kokoro_dub.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method local \
  --tts-engine kokoro \
  --asr-backend local \
  --diarization-backend local \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=if_sara,SPEAKER_01=im_nicola
```

After each full run:

```bash
ffprobe -v error \
  -show_entries stream=codec_type,codec_name,duration,sample_rate,channels \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  test_IT_only_cloud_edge_dub.mp4
```

Repeat the `ffprobe` check for the local output.

## macOS Regression Matrix

Run the fast checks with the system `/bin/bash`; this is Bash 3.2 on the tested
Intel Mac. Then use a fresh input basename so ASR and diarization are not
silently satisfied by an older cache.

The minimum release matrix is:

1. One complete `--only-cloud` run with AssemblyAI, Google Translate, Groq, and Edge TTS.
2. One complete local run with faster-whisper, pyannote, NLLB, Ollama/Qwen, and Kokoro.
3. One XTTS rerender from the successful local run's cached transcript, diarization, and translation.
4. Kokoro and Edge voice listing, plus one short sample when their integration changed.

Force CPU execution for the local Intel macOS run:

```bash
/bin/bash ./autodub_local.sh \
  --input test.mp4 \
  --output test_IT_all_local_kokoro_macos_dub.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method local \
  --tts-engine kokoro \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=if_sara,SPEAKER_01=im_nicola \
  --tts-voice-map-strict \
  --llm-adapt auto \
  --llm-segment auto \
  --no-gpu
```

For the XTTS variant, reuse the same input and options, change the output name,
replace `--tts-engine kokoro` with `--tts-engine xtts`, and remove both
`--tts-voice-map` and `--tts-voice-map-strict`.
This exercises voice-reference extraction and cloning without needlessly
repeating the already validated ASR, diarization, NLLB, and Qwen stages.

`--llm-segment auto` intentionally keeps heuristic segmentation in an ordinary
local run. If local Ollama segmentation code changed, add one targeted
`--llm-segment always --stop-after-translation` run; do not make this very slow
Intel-only variant part of every release.

## Audio Review Points

Listen specifically for:

- avoidable overlap between adjacent utterances
- long pauses followed by rushed speech
- speaker voice assignment errors
- English technical terms read with target-language phonetics
- missing clauses around speaker changes
- translations that preserve timing but lose essential meaning

## Dependency Checks

For releases that touch dependency selection, test these paths in a fresh venv
or after temporarily moving `.autodub_local/venv` aside:

- `--only-cloud` should not require local Whisper, pyannote, NLLB, Kokoro, XTTS, or Ollama model downloads.
- `--tts-engine edge` should install `edge-tts`, not Kokoro or XTTS.
- `--tts-engine kokoro` should install Kokoro and `espeak-ng`.
- `--translation-method google` should not load NLLB.
- `--translation-method local` should load NLLB and include the NLLB license warning in the README.
- `--llm-segment auto` should use Groq segmentation in `--only-cloud` and keep heuristic-only segmentation in ordinary local runs.
- The default Groq LLM should be `openai/gpt-oss-120b`; its requests should use JSON Object Mode with low-effort hidden reasoning.
- Intel macOS local modes should create and reuse `.autodub_local/venv_macos_intel_py311` with Python 3.11, Transformers 4.46.2, and Coqui TTS 0.25.3.
- Local Qwen requests should set `think` to false, and the Ollama model should be unloaded before memory-heavy local translation or TTS continues.
- A fresh macOS run should keep live console output while writing the same content to `.autodub_local/logs/`; this protects the stock-Bash FIFO logging path.

## Release Gate

Do not publish a release until:

- fast checks pass
- `--help` matches the README
- at least one cloud-assisted translation-only check passes
- at least one fully local translation-only check passes
- the macOS matrix passes when a release changes platform or dependency handling
- the maintainer has listened to the regenerated reference dub files on the target machine
- license and service-limit notes in the README still match the currently used tools and services
