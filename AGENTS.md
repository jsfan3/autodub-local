# Agent Instructions

This repository contains a single Bash entrypoint, `autodub_local.sh`, with an
embedded Python worker generated at runtime. Linux and macOS are both supported
release platforms. Real audio quality checks matter more than synthetic unit
tests alone.

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
- Persistent Intel macOS local environment: `.autodub_local/venv_macos_intel_py311/`
- Logs: `.autodub_local/logs/`
- Shared caches: `.autodub_local/cache/`
- Miscellaneous temporary data: `.autodub_local/tmp/`

Do not delete models, virtual environments, logs, shared caches, or tokens
unless the maintainer explicitly requests it. Use `./autodub_local.sh clean`
for ordinary per-input work-folder cleanup.

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
- Preserve both Linux and macOS compatibility. Do not make Windows support
  claims beyond WSL2 until a native Windows workflow has been validated.
- Update `README.md` and `TESTING.md` when adding or changing CLI options.

## Important Workflows

Cloud-assisted reference workflow:

```bash
/bin/bash ./autodub_local.sh \
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
/bin/bash ./autodub_local.sh \
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
/bin/bash -n autodub_local.sh
/bin/bash ./autodub_local.sh --help
```

When the embedded Python worker changes, extract and compile it as described in
`TESTING.md`.

When ASR, diarization, translation, LLM adaptation, or TTS timing changes, run
at least one `--stop-after-translation` workflow and inspect the generated JSON.

## Cross-Platform Validation

Every code change must be tested first on the machine where it was developed.
Before publication, the exact candidate commit must then be validated on the
other supported platform. The direction is symmetric: Linux changes are handed
to macOS, and macOS changes are handed to Linux.

Use tests proportional to the change:

- Documentation-only and comment-only changes do not require cross-platform
  end-to-end runs.
- Isolated code changes require the fast checks and the smallest relevant
  targeted test on both platforms.
- Changes to dependencies, installation, shell compatibility, path handling,
  subprocesses, caching, logging, model loading, or a pipeline backend require
  the affected end-to-end workflow on both platforms.
- Broad platform changes and release candidates require the complete matrix in
  `TESTING.md`: cloud-assisted, fully local with Kokoro, and the XTTS variant.

When the two machines cannot share a working tree, publish the candidate on a
temporary validation branch. The second machine must test that exact commit and
push a result report back to the same branch. The report must include the commit
SHA, environment versions, commands, PASS/FAIL results, runtimes, output paths,
and compact media validation. It must not contain keys, tokens, caches, models,
generated media, or full logs.

A failure on either supported platform blocks publication until it is fixed and
the affected checks pass again on both platforms. A temporary validation commit
or branch is not the final release commit and must not be merged as validation
evidence alone.

If a change can affect transcription, segmentation, translation, LLM
adaptation, TTS, timing, mixing, or muxing, regenerate the relevant MP4 and
perform a maintainer listening check. Platform-specific audio behavior requires
listening on each affected platform; otherwise automated media checks may run on
both platforms with at least one maintainer listening pass before release.

## Contribution Policy

The maintainer wants to test changes locally before publication. External
contributors should open GitHub issues for bugs or requests for enhancement
instead of pull requests. A small code snippet in an issue is acceptable when it
clarifies the requested change, but it is not required.
