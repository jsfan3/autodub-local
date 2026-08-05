# Agent Instructions

This repository contains a single Bash entrypoint, `autodub_local.sh`, with an
embedded Python worker generated at runtime. The project is maintained primarily
on the maintainer's Linux machine, and real audio quality checks matter more
than synthetic unit tests alone.

## Project Shape

- Main script: `autodub_local.sh`
- User documentation: `README.md`
- Release checklist: `TESTING.md`
- Generated runtime state: `.autodub_local/`
- Hugging Face token: `.hf_token`
- Cloud keys: `.cloud_keys`

Never print, commit, or copy API keys or tokens. `.hf_token` and `.cloud_keys`
must remain local-only files with restricted permissions.

## Runtime State

`.autodub_local/` contains both disposable and persistent data:

- Disposable per-input work folders: `.autodub_local/<input_stem>/`
- Persistent models: `.autodub_local/models/`
- Persistent Python environment: `.autodub_local/venv/`
- Logs: `.autodub_local/logs/`
- Shared caches: `.autodub_local/cache/`
- Miscellaneous temporary data: `.autodub_local/tmp/`

Do not delete models, the venv, logs, shared caches, or tokens unless the
maintainer explicitly requests it. Use `./autodub_local.sh clean` for ordinary
per-input work-folder cleanup.

## Implementation Rules

- Keep source code, comments, help text, and documentation in English.
- Keep user-facing runtime errors specific and actionable.
- Prefer existing local patterns over new abstractions.
- Keep `--only-cloud` lightweight: it must not require local Whisper, pyannote,
  NLLB, Kokoro, XTTS, or Ollama model downloads for the main workflow.
- Keep utterance segmentation language-agnostic. Do not hard-code source-language
  words such as English conjunctions to decide where to split or merge speech.
- Treat voice cloning as local-only unless the maintainer explicitly approves a
  cloud cloning backend.
- Do not make macOS or Windows support claims unless the workflow has been
  tested. Windows should be considered WSL2-only until proven otherwise.
- Update `README.md` and `TESTING.md` when adding or changing CLI options.

## Important Workflows

Cloud-assisted reference workflow:

```bash
./autodub_local.sh \
  --input test.mp4 \
  --output test_IT_only_cloud_edge_dub.mp4 \
  --source-lang en \
  --target-lang it \
  --only-cloud \
  --num-speakers 2 \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
```

Fully local reference workflow:

```bash
./autodub_local.sh \
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

## Testing Expectations

Use `TESTING.md` as the release gate. For ordinary code edits, run at least:

```bash
bash -n autodub_local.sh
./autodub_local.sh --help
```

When the embedded Python worker changes, extract and compile it as described in
`TESTING.md`.

When ASR, diarization, translation, LLM adaptation, or TTS timing changes, run
at least one `--stop-after-translation` workflow and inspect the generated JSON.
For release candidates, regenerate and listen to both reference MP4 files on the
maintainer's machine.

## Contribution Policy

The maintainer wants to test changes locally before publication. External
contributors should open GitHub issues for bugs or requests for enhancement
instead of pull requests. A small code snippet in an issue is acceptable when it
clarifies the requested change, but it is not required.
