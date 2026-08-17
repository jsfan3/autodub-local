# autodub-local: Video Dubbing for Linux and macOS

> **autodub-local 3.0**
> Created by [Francesco Galgani](https://www.informatica-libera.net/)
> [Source repository](https://github.com/jsfan3/autodub-local) · [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

Video dubbing for Linux and macOS using local open-source models, cloud
speech/LLM services, local/online TTS engines, or a mixed workflow.

The script processes one explicit input file, translates the spoken content, generates a dubbed audio track, and muxes it back into an MP4 output. It is designed for long recordings such as webinars, livestreams, interviews, public talks, and meetings where perfect lip-sync is not required.

## Contents

- [Pipeline](#pipeline)
- [Operation Modes](#operation-modes)
- [Agent-Assisted Dubbing](#agent-assisted-dubbing)
- [Tested Environment](#tested-environment)
- [Requirements](#requirements)
- [API Keys And Tokens](#api-keys-and-tokens)
- [Basic Usage](#basic-usage)
- [Help](#help)
- [Cleanup](#cleanup)
- [Included Test Dubs](#included-test-dubs)
- [Examples](#examples)
- [Main Options](#main-options)
- [Language Handling](#language-handling)
- [Translation Backends](#translation-backends)
- [TTS Engines](#tts-engines)
- [Voice Selection](#voice-selection)
- [LLM Adaptation](#llm-adaptation)
- [CPU-Only Runs](#cpu-only-runs)
- [TTS And Timing Options](#tts-and-timing-options)
- [Segmentation Options](#segmentation-options)
- [ASR Options](#asr-options)
- [Cache Layout](#cache-layout)
- [Testing Releases](#testing-releases)
- [Contributing](#contributing)
- [Use, Licenses, And Service Terms](#use-licenses-and-service-terms)
- [Limitations](#limitations)

## Pipeline

1. Extract mono 16 kHz audio with `ffmpeg`
2. Transcribe with local `faster-whisper`, Groq Whisper, or AssemblyAI
3. Run speaker diarization with local `pyannote.audio` or AssemblyAI speaker labels
4. Rebuild bounded speaker utterances
5. Translate with local NLLB or Google Translate through `deep-translator`
6. Optionally shorten overlong translated lines with local Ollama or Groq LLM
7. Generate target-language speech with `xtts`, `kokoro`, or `edge`
8. Safely time-stretch only when the tempo change is small
9. Assemble the dubbed timeline and mux it into an MP4 output

The script supports checkpoint/resume under `.autodub_local/`.

## Operation Modes

The script can be used in two broad modes.

**Cloud-assisted mode** is recommended for most users who do not have a recent GPU, enough disk space for local models, or a strict privacy requirement. Use `--only-cloud` to select AssemblyAI for transcription and diarization, Google Translate for translation, Groq for LLM adaptation, Edge TTS for speech, and CPU-only local orchestration. This avoids local Whisper, pyannote, NLLB, Kokoro/XTTS, and Ollama model downloads for the main workflow.

**Fully local mode** keeps the heavy AI stages on the local machine: faster-whisper, pyannote, NLLB, Ollama, and Kokoro or XTTS. Use this mode when privacy, offline reproducibility, or avoiding cloud services matters more than runtime and disk usage. First runs can download many gigabytes of model weights.

Cloud mode is not automatically higher quality. It is primarily a low-hardware mode. On the included `test.mp4`, the cloud and local outputs were broadly comparable, but AssemblyAI produced a different utterance segmentation from local Whisper+pyannote. That can change pauses, speaker windows, and the amount of text assigned to each TTS slot.

Voice cloning is currently implemented only in local mode through `--tts-engine xtts`.

> **macOS compatibility:** cloud-assisted mode and the complete local pipeline
> are supported and hardware-validated on Intel macOS, including the stock
> `/bin/bash` 3.2, faster-whisper, pyannote, NLLB, Ollama/Qwen, Kokoro, and XTTS.
> Apple Silicon follows the normal macOS dependency path but has not yet been
> validated on physical Apple Silicon hardware.

## Agent-Assisted Dubbing

[EXAMPLE_OF_AGENT_ASSISTED_USAGE.md](EXAMPLE_OF_AGENT_ASSISTED_USAGE.md) is a
reusable instruction playbook for an AI coding agent. It is not a second
executable or a replacement for `autodub_local.sh`: the Bash script performs
the repeatable dubbing pipeline, while the agent manages a slower,
publication-oriented review and finishing workflow around it.

If no checkout is available, the playbook instructs the agent to make a normal,
non-shallow clone of the published repository and work inside it. If a checkout
already exists, the agent reuses it without overwriting local changes. Model
caches, virtual environments, API keys, tokens, and other ignored or untracked
local files are not part of the clone.

The agent-assisted workflow adds work that a single unattended script run
cannot reliably complete in every source:

- semantic review of the complete transcript and translation using grammar,
  subject matter, names, terminology, and surrounding context;
- optional comparison with an independent cloud transcript without silently
  replacing the local result;
- verification of speaker assignments, voice mappings, timing, and generated
  manifests;
- human approval gates for uncertain corrections, voice choices, translation
  style, and the final publication render;
- a semantic pause timeline that distinguishes continuations inside a sentence,
  clause breaks, completed sentences, emphatic endings, and speaker changes;
- synchronized removal of selected silence from both audio and video, preserving
  spoken content and preventing long gaps from breaking sentence flow;
- resumable long stages, backups, hashes, pilot renders, and detailed final QA;
- an unobtrusive watermark crediting both `autodub-local` and the original
  publication URL, plus its license when known.

An agent may also introduce additional evidence-based checks when a particular
source requires them. For example, if diarization and audio context leave a
speaker turn ambiguous, the agent may inspect relevant video frames before
asking the user to resolve the remaining uncertainty.

This human-in-the-loop process takes more time and may require interaction with
the user, but it can generally produce a more carefully reviewed publication
master than a fully unattended workflow, especially for difficult speaker
turns, terminology, translations, or timing decisions. It is a quality-oriented
workflow, not a guarantee that every source can be corrected automatically.

### Published Agent-Assisted Examples

- **Russian to Italian:**
  [Alla Voronkova - Digiuno secco a cascata, parte 1](https://www.informatica-libera.net/video/AllaVoronkova-DigiunoSeccoACascata-Parte1_IT_dub.mp4)
- **French to Italian:**
  [Nicolas Pilartz - Introduzione all'alimentazione pranica, parte 1](https://www.informatica-libera.net/video/nicolas-pilartz-introduzione-alimentazione-pranica-1.mp4)
  and [part 2](https://www.informatica-libera.net/video/nicolas-pilartz-introduzione-alimentazione-pranica-2.mp4)
- **English to Italian:**
  [Richard Stallman at Georgia Tech, Italian dub](https://www.informatica-libera.net/video/rms-2026-01-23-georgia-tech-ITALIANO.mp4)

The Russian and French dubs are shorter than their source videos only because
selected silence was removed to make listening more fluid; no spoken content
was removed. The English dub preserves the original pauses, so its duration is
the same as the source. Each watermark includes the original publication URL
for anyone who wants to repeat the experiment with the same source material.

These examples were produced with OpenAI Codex using
[GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) with
Extra High (`xhigh`) reasoning effort. This records the agent configuration used
for these publications; it is not a requirement or a guarantee that another
run will make identical editorial decisions.

## Tested Environment

### Linux Mint 22

- **Python:** 3.12.3
- **CPU and RAM:** Intel Core i7-7700HQ, 16 GB
- **GPU:** NVIDIA GeForce GTX 1050 Mobile, 4 GB
- **Validated workflows:** cloud-assisted and fully local

### macOS 15.7.9 (Intel x86_64)

- **Python:** 3.12 for cloud workflows; 3.11 for local workflows
- **CPU and RAM:** Intel Core i5-1030NG7, 8 GB
- **GPU:** integrated Intel graphics
- **Validated workflows:** cloud-assisted, fully local with Kokoro, and fully
  local with XTTS

On the tested Linux GPU, current PyTorch CUDA wheels do not support the card's
compute capability, so torch-based stages fall back to CPU. Ollama may still
offload some local LLM work to the GPU when the model fits.

On the tested Intel Mac, a fresh two-minute `--only-cloud` run took about 1
minute 50 seconds. The fully local pipeline also completed with both Kokoro and
XTTS, but Qwen inference slowed substantially under sustained CPU load. Expect
local CPU-only runs on an 8 GB Intel Mac to take tens of minutes, and possibly
longer when many lines need LLM adaptation. This is a performance limitation,
not a compatibility failure.

## Requirements

The script can install missing system packages with `apt-get` on
Debian/Ubuntu-like systems or with Homebrew on macOS:

- `ffmpeg`
- Python with virtual-environment support
- `curl`
- `espeak-ng` for Kokoro TTS
- Ollama when local LLM work is selected

Install [Homebrew](https://brew.sh/) first if macOS is missing any of these
dependencies. Intel macOS local modes use a separate Python 3.11 environment,
created automatically under `.autodub_local/venv_macos_intel_py311`, because
the compatible PyTorch, Transformers, and XTTS packages require that version.
Cloud-only mode continues to use the lightweight default environment.

Fully local first runs can download many gigabytes of Python packages and model weights.
For Kokoro Japanese or Chinese output, the script also installs the matching `misaki` language extras.
Microsoft Edge TTS uses the online Edge TTS service through the Python `edge-tts` package and does not require a local GPU.

## API Keys And Tokens

A Hugging Face READ token is required only when local pyannote diarization is used.

Before the first run:

1. Accept the terms for `pyannote/speaker-diarization-3.1`
2. Accept the terms for `pyannote/segmentation-3.0`
3. Create a Hugging Face READ token

If `HF_TOKEN` is not set, the script prompts once and stores the token in `.hf_token` with restricted permissions. Voice listing, voice sampling, and `--only-cloud` runs do not require this token.

Cloud services use API keys:

- **GroqCloud** — Used for Groq Whisper ASR and Groq LLM adaptation.
  [Create or manage a key](https://console.groq.com/keys).
- **AssemblyAI** — Used for cloud transcription and speaker diarization.
  [Create an account or open the dashboard](https://www.assemblyai.com/dashboard/signup).

Use environment variables when possible:

```bash
export GROQ_API_KEY="..."
export ASSEMBLYAI_API_KEY="..."
```

If a required cloud key is missing in an interactive terminal, the script prompts for it with hidden input, then asks whether to save it in `.cloud_keys` with restricted permissions. Keys supplied through environment variables are not saved automatically.

## Basic Usage

There are no default input files, language pairs, translation methods, or TTS engines. Pass one input file and all required choices explicitly:

```bash
chmod +x autodub_local.sh
./autodub_local.sh \
  --input test.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method google \
  --tts-engine edge \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
```

The input can be any file readable by `ffmpeg`, for example MP4, WebM, MKV, MOV, or an audio-only file. The default output is:

```text
<input_stem>_<TARGET_LANG>_<TTS_ENGINE>_dub.mp4
```

Example:

```text
test_IT_edge_dub.mp4
```

Use `--output` to choose a different path.

## Help

```bash
./autodub_local.sh --help
```

## Cleanup

Per-input temporary work folders can be listed and removed interactively:

```bash
./autodub_local.sh clean
```

`clean` is an exclusive maintenance command: use it by itself. It lists the
per-input folders under `.autodub_local/`, shows status, size, and modification
time, then lets you delete all or selected folders. It does not delete persistent
models, the Python virtual environment, shared caches, logs, `.hf_token`, or
`.cloud_keys`.

## Included Test Dubs

The repository includes two generated dubs of `test.mp4` for direct comparison.

Cloud-assisted output:

```text
test_IT_only_cloud_edge_dub.mp4
```

The cloud sample was regenerated with the current `--only-cloud` defaults,
including AssemblyAI Universal-3.5 Pro and GPT OSS 120B on Groq.

Command used:

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

Measured runtime on the tested underclocked machine: `140.00s` real time.

Fully local output:

```text
test_IT_all_local_kokoro_dub.mp4
```

Command used:

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

Measured runtime on the tested underclocked machine: `1327.34s` real time.

On this older underclocked machine, the fully local run was about `9.5x` slower
than the cloud-assisted run. That difference is significant, but still
reasonable when the computer can be left running for long jobs and privacy or
offline reproducibility matters.

After installing and testing both workflows on this machine, `.autodub_local/`
occupied about `18G`, including about `8.1G` for models, `8.0G` for the Python
virtual environment, and `1.3G` for shared caches. A fresh `--only-cloud`
workflow should avoid the local Whisper, pyannote, NLLB, Kokoro/XTTS, and
Ollama model downloads, but it still needs a Python environment and lightweight
runtime packages.

## Examples

Cloud-assisted English to Italian with AssemblyAI diarization, Google Translate, Groq LLM adaptation, and Microsoft Edge TTS:

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

Fully local English to Italian with faster-whisper, pyannote, NLLB, Ollama, and Kokoro preset voices:

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

English to Italian with Google Translate, Microsoft Edge TTS, automatic speaker count, and explicit speaker voices:

```bash
./autodub_local.sh \
  --input test.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method google \
  --tts-engine edge \
  --num-speakers auto \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
```

List Edge voices for Italian:

```bash
./autodub_local.sh \
  --target-lang it \
  --tts-engine edge \
  --list-tts-voices
```

Generate local Edge voice samples before choosing a voice map:

```bash
./autodub_local.sh \
  --target-lang it \
  --tts-engine edge \
  --sample-tts-voices \
  --sample-text "Questo e un test con termini inglesi come free software e Georgia Tech."
```

Auto-detected source language to French with local NLLB and XTTS voice cloning:

```bash
./autodub_local.sh \
  --input talk.webm \
  --source-lang auto \
  --target-lang fr \
  --translation-method local \
  --tts-engine xtts
```

Diagnostic run without LLM shortening, useful only for inspecting raw translation output:

```bash
./autodub_local.sh \
  --input interview.mkv \
  --source-lang en \
  --target-lang es \
  --translation-method local \
  --tts-engine kokoro \
  --llm-adapt never
```

Do not use `--llm-adapt never` for final dubs unless you explicitly want to skip all duration-oriented text adaptation.

Stop after translation for review or testing:

```bash
./autodub_local.sh \
  --input test.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method local \
  --tts-engine edge \
  --stop-after-translation
```

Use strict voice mapping when the detected speaker set must match exactly:

```bash
./autodub_local.sh \
  --input panel.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method google \
  --tts-engine edge \
  --num-speakers auto \
  --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural \
  --tts-voice-map-strict
```

## Main Options

Maintenance command:

```bash
./autodub_local.sh clean
```

Use `clean` by itself. It is not a dubbing option and cannot be combined with
other arguments.

- **`--input FILE`** — Required for dubbing. Input media file.
- **`--source-lang CODE`** — Required for dubbing. Source language for Whisper,
  or `auto`.
- **`--target-lang CODE`** — Required. Target language for translation and TTS.
- **`--translation-method local|google`** — Required unless `--only-cloud` is
  used. Translation backend.
- **`--tts-engine xtts|kokoro|edge`** — Required unless `--only-cloud` is used.
  TTS engine.
- **`--output FILE`** — Derived automatically. Output MP4 path.
- **`--only-cloud`** — Off by default. Uses AssemblyAI ASR/diarization, Google
  Translate, Groq LLM, Edge TTS, and CPU-only orchestration.
- **`--num-speakers N|auto`** — Default: `auto`. Speaker count for diarization.
- **`--min-speakers N`** — Unset by default. Lower bound when
  `--num-speakers auto` is used.
- **`--max-speakers N`** — Unset by default. Upper bound when
  `--num-speakers auto` is used.
- **`--whisper-model NAME`** — Default: `medium`. faster-whisper model.
- **`--review-translation`** — Off by default. Pauses after creating the
  translation JSON in interactive terminals.
- **`--stop-after-translation`** — Off by default. Exits after writing the
  translation JSON.
- **`--no-gpu`** — Off by default. Forces CPU mode for local ML stages and
  requests CPU-only Ollama inference.

CLI options are the recommended interface. Environment variables with matching uppercase names are still accepted as fallback for automation.

## Language Handling

`SOURCE_LANG` and `TARGET_LANG` are the main language controls. NLLB codes are derived automatically from them. Advanced overrides are still available:

```bash
--nllb-src-lang eng_Latn
--nllb-tgt-lang ita_Latn
```

These overrides are normally unnecessary.

XTTS target languages supported by the script:

```text
ar, cs, de, en, es, fr, hi, hu, it, ja, ko, nl, pl, pt, ru, tr, zh
```

Kokoro target languages supported by the script:

```text
en, en-us, en-gb, es, fr, hi, it, ja, pt, zh
```

Edge TTS uses Microsoft locales such as `it-IT`, `en-US`, `en-GB`, or `pt-BR`. Use `--tts-locale` when the target language has multiple locale variants or when you want a specific regional voice catalog.

## Translation Backends

`--translation-method google` uses `deep-translator`. It is usually fluent, but it uses an online unofficial route and can fail or hit rate limits. The script translates sentence/fragments rather than whole long utterances, so a transient backend failure is less likely to leave an entire speaker block untranslated.

`--translation-method local` uses NLLB. It is offline after model download and avoids online translation instability. Since NLLB-200 is intended for sentence-level translation rather than document translation, the script splits each utterance into smaller sentence/fragments before translation and stores the per-unit result in `translation_units`. This avoids dropped trailing clauses in multi-sentence ASR chunks, but the raw output can still be more literal than Google Translate and can still make domain mistakes, such as interpreting "free" as zero price instead of software freedom. For long runs where reproducibility matters, use local translation with `--llm-adapt auto` unless you want to inspect the raw NLLB output.

## TTS Engines

`--tts-engine xtts` uses XTTS v2 voice cloning. It extracts reference clips per diarized speaker and tries to synthesize each translated utterance with the matching cloned voice. This can preserve speaker identity, but quality depends strongly on the original audio and reference clip quality.

`--tts-engine kokoro` uses local Kokoro preset voices. It does not clone the original speakers. The script estimates each diarized speaker's rough pitch, classifies the speaker as `male`, `female`, or `child`, and assigns a target-language voice. Child fallback uses a female voice when no child-specific voice is available. Pitch shift is not applied automatically.

`--tts-engine edge` uses Microsoft Edge online neural voices through `edge-tts`. It does not require a GPU and it can pronounce many English terms correctly inside non-English text, depending on the voice. It is not an official stable Microsoft API like Azure Speech, so availability and behavior can change. The Edge voice catalog is cached in `.autodub_local/cache/edge_tts_voices.json` after a successful lookup, but speech synthesis still requires access to the online service.

## Voice Selection

For Kokoro and Edge, use an explicit map when you know which speaker should use which voice:

```bash
--tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
```

The map can be partial. If `--tts-voice-map-strict` is not set, any unmapped detected speaker is assigned automatically from the pitch-based `male`, `female`, or `child` class. If the map contains speakers that diarization does not detect, they are ignored with a warning.

With `--tts-voice-map-strict`, the detected speaker ids must match the map exactly. This is useful when `--num-speakers auto` might detect an unexpected extra speaker and you prefer to stop instead of assigning a fallback voice.

Override automatic class defaults:

```bash
--tts-voice-female VOICE
--tts-voice-male VOICE
--tts-voice-child VOICE
```

List voices:

```bash
./autodub_local.sh --target-lang it --tts-engine edge --list-tts-voices
./autodub_local.sh --target-lang it --tts-engine kokoro --list-tts-voices
```

Generate sample files:

```bash
./autodub_local.sh --target-lang it --tts-engine edge --sample-tts-voices
./autodub_local.sh --target-lang it --tts-engine kokoro --sample-tts-voices
```

Preview links printed by `--list-tts-voices`:

- Microsoft Voice Gallery: https://speech.microsoft.com/portal/voicegallery
- Unofficial Edge TTS web preview: https://edge-tts.com/
- Kokoro demo: https://huggingface.co/spaces/hexgrad/Kokoro-TTS
- Kokoro voice list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

## LLM Adaptation

By default, `--llm-adapt auto` uses the selected LLM provider only for translated lines that appear too long for the original time slot. The script validates the model output as JSON and checks the resulting character budget.

The character budget is speaker-aware by default: it is derived from the original speaking rate of the diarized speaker, then clamped to conservative limits for the target language. This avoids forcing a fast presenter and a slower speaker into the same global density. The adapter also asks the LLM to stay above a preferred minimum length, so it does not over-compress a sentence just because a much shorter summary fits the slot. If the model cannot satisfy every constraint after retries, the script prefers a slightly over-budget but more complete candidate over an excessively shortened one, and records the result in the manifest.

Default local model:

```text
qwen3:8b-q4_K_M
```

If `--llm-provider ollama` is used and Ollama or the model is missing, the script attempts to install Ollama and pull the model. Use `--skip-ollama-install` to fail instead.

Local Qwen requests explicitly disable the model's optional thinking trace.
These operations are narrowly constrained rewriting and JSON-grouping tasks;
the trace otherwise consumes the local `--llm-num-predict` budget before the
required JSON answer starts. The model is also unloaded after its last request
so NLLB and TTS can reuse the RAM, which matters on memory-limited Macs.

Cloud LLM adaptation is available through Groq:

```bash
--llm-provider groq --groq-llm-model openai/gpt-oss-120b
```

`--only-cloud` selects this Groq LLM mode automatically.
For this constrained JSON editing workload, the default GPT OSS model uses
low-effort reasoning, hides the reasoning from the response, and enables JSON
Object Mode. This keeps reasoning available while limiting its free-tier token
use and reducing invalid responses that would need a retry.

The same provider can also be used for utterance boundary grouping through
`--llm-segment`. This step runs before translation and asks the LLM to group
adjacent same-speaker ASR utterances into more natural dubbing units without
editing, reordering, dropping, or translating the source text. The output is
validated as JSON and the script falls back to the heuristic boundaries if the
LLM response is invalid.

`--llm-segment auto` is the default. In ordinary local runs it keeps the
language-agnostic heuristic only. In `--only-cloud`, it tries Groq LLM
segmentation because Groq is already selected as the cloud LLM provider.

Relevant options:

- **`--llm-adapt auto|always|never`** — Default: `auto`.
- **`--llm-segment auto|always|never`** — Default: `auto`.
- **`--llm-provider ollama|groq`** — Default: `ollama`.
- **`--llm-model NAME`** — Default: `qwen3:8b-q4_K_M`.
- **`--groq-llm-model NAME`** — Default: `openai/gpt-oss-120b`.
- **`--llm-chars-per-second N`** — Uses a speaker-aware language default;
  setting this option overrides every per-speaker budget.
- **`--llm-max-retries N`** — Default: `3`.
- **`--llm-temperature N`** — Default: `0.1`.
- **`--llm-timeout SECONDS`** — Default: `0`.
- **`--llm-num-predict N`** — Default: `256`.

`--llm-timeout 0` disables the request timeout. This is the default so very slow CPU-only or underclocked runs can continue for days if needed.
GPT OSS requests use a minimum completion ceiling of 1024 tokens so low-effort
reasoning cannot exhaust the response budget before the required JSON is complete.
This is a maximum, not a fixed token allocation, and avoids token-wasting retries.
The local Ollama ceiling remains the configured `--llm-num-predict` value,
which defaults to 256; it does not limit transcription, translation, or TTS.

## CPU-Only Runs

Use `--no-gpu` to force CPU mode for local ML stages:

```bash
./autodub_local.sh \
  --input long_talk.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method google \
  --tts-engine edge \
  --no-gpu
```

This hides CUDA/HIP/Vulkan devices from the script process, forces `TRANSLATE_ON_GPU=0`, makes PyTorch-based components choose CPU, and sends `num_gpu=0` in Ollama LLM adaptation requests. If an Ollama server is already running outside the script, `--no-gpu` still requests CPU-only inference through the API, but it cannot fully control the environment that external server was started with.

On the maintainer's older Linux machine, `--no-gpu` is deliberately combined
with a CPU underclock configured outside this project during long local jobs.
Both choices substantially increase runtime, but they reduce sustained heat and
thermal throttling during very high summer ambient temperatures. This is a
machine-specific reliability and thermal-management tradeoff, not a general
performance recommendation and not something the script configures itself.

The most practical low-heat combination is now:

```bash
--only-cloud
```

For a mixed cloud/local run that still uses local diarization, use:

```bash
--translation-method google --tts-engine edge --no-gpu
```

To avoid local LLM work during diagnostics:

```bash
--llm-adapt never
```

## TTS And Timing Options

Common timing options:

- **`--tts-speed N`** — Default: `1.0` base speed before per-speaker pacing.
- **`--tts-max-chars N`** — Default: `5000`.
- **`--max-tts-compress-ratio N`** — Default: `1.15`.
- **`--max-tts-expand-ratio N`** — Default: `1.20`.
- **`--aac-bitrate RATE`** — Default: `192k`.

Edge-specific options:

- **`--tts-locale LOCALE`** — Derived automatically.
- **`--edge-pitch VALUE`** — Default: `+0Hz`.
- **`--edge-volume VALUE`** — Default: `+0%`.
- **`--edge-connect-timeout N`** — Default: `20`.
- **`--edge-receive-timeout N`** — Default: `120`.
- **`--edge-max-retries N`** — Default: `3`.
- **`--edge-retry-delay N`** — Default: `5`.

XTTS cloning options:

- **`--max-ref-clips N`** — Default: `3`.
- **`--xtts-max-chars N`** — Default: `180`.
- **`--xtts-speed N`** — Default: `1.0`.
- **`--xtts-temperature N`** — Default: `0.65`.
- **`--xtts-repetition-penalty N`** — Default: `2.0`.
- **`--xtts-inter-chunk-silence-ms N`** — Default: `120`.

The script does not force aggressive global speed-up. It only applies ffmpeg
`atempo` when the required change is within the configured safety ratios.

If a generated TTS segment is longer than the nominal ASR utterance but there is
silence before the next utterance, the script can use that free window as the
stretch target. This avoids some unnecessary overlap without changing the
utterance start time or applying an unsafe tempo change. If the segment is still
too long, the script keeps the natural TTS duration and records the timing
overflow in the manifest/logs.

## Segmentation Options

- **`--utterance-max-gap SEC`** — Default: `0.9`.
- **`--utterance-max-duration SEC`** — Default: `18.0`.
- **`--utterance-max-chars N`** — Default: `420`.
- **`--utterance-repair-max-gap SEC`** — Default: `2.2`.
- **`--utterance-repair-max-duration SEC`** — Default: `24.0`.
- **`--utterance-repair-max-chars N`** — Default: `620`.

These limits prevent very long same-speaker blocks from becoming a single translation/TTS segment.

After the first pass, the script also performs a conservative language-agnostic
boundary repair. It can merge adjacent same-speaker utterances when the gap is
small, the previous text does not end with strong terminal punctuation, and the
merged block stays under the repair limits. This helps avoid cuts such as one
segment ending with an incomplete clause and the next segment continuing the
same sentence.

When `--llm-segment` enables LLM segmentation, the LLM receives the already
repaired source-language utterances with timestamps and speaker ids. It can only
return groups of adjacent ids; the script does not accept edited text from this
step.

## ASR Options

ASR and diarization can be local, cloud-based, or mixed:

- **`--asr-backend local|groq|assemblyai`** — Default: `local`. Transcription
  backend.
- **`--diarization-backend local|assemblyai`** — Default: `local`. Speaker
  diarization backend.
- **`--groq-whisper-model NAME`** — Default: `whisper-large-v3`. Groq Whisper
  model.
- **`--groq-prompt TEXT`** — Unset by default. Optional prompt or context for
  Groq Whisper.
- **`--groq-chunk-seconds N`** — Default: `120`. Chunk size for Groq uploads.
- **`--groq-overlap-seconds N`** — Default: `1.0`. Chunk overlap used only for
  transcription context.
- **`--groq-timeout SECONDS`** — Default: `300`. Groq request timeout.
- **`--groq-max-retries N`** — Default: `5`. Groq retry count.
- **`--groq-rate-limit wait|fail`** — Default: `wait`. Behavior on HTTP 429.
- **`--assemblyai-speech-model NAME`** — Default:
  `universal-3-5-pro,universal-2`. AssemblyAI model list.
- **`--assemblyai-poll-interval SEC`** — Default: `5`. Poll interval.
- **`--assemblyai-timeout SEC`** — Default: `7200`. Total AssemblyAI wait
  timeout.

`--asr-backend groq` replaces local faster-whisper with Groq Whisper but still needs local diarization unless another diarization backend is selected. Groq does not currently provide speaker diarization in this script.

`--asr-backend assemblyai --diarization-backend assemblyai` uses AssemblyAI for both transcript and speaker labels. `--only-cloud` selects this pair automatically.

Local faster-whisper options:

- **`--asr-beam N`** — Default: `5`.
- **`--asr-vad true|false`** — Default: `true`.
- **`--asr-compute-gpu TYPE`** — Default: `int8_float16`.
- **`--asr-compute-cpu TYPE`** — Default: `int8`.

## Cache Layout

Intermediate files are stored under:

```text
.autodub_local/<input_stem>/
```

Typical files:

```text
<stem>.mono16k.wav
<stem>.mono16k.meta.json
<stem>.transcript.json
<stem>.diarization.json
<stem>.utterances.json
<stem>.translated.<target>.json
<stem>.speakers.<target>.<tts_engine>.json
<stem>.<target>.<tts_engine>.manifest.json
<stem>.<target>.<tts_engine>.wav
tts_segments_<tts_engine>/
reference_clips/
```

Translated utterances use the generic field:

```json
{
  "text_translated": "..."
}
```

Local NLLB translations can also include:

```json
{
  "translation_units": [
    {
      "source": "...",
      "text_translated": "..."
    }
  ]
}
```

JSON caches include a `config` block. If relevant inputs or parameters change, the affected cache is regenerated.

Speaker profile caches store rough pitch, voice class, selected TTS voice, source speaking rate, and per-speaker TTS speed. TTS manifests store the final voice, speed/rate, generated duration, timing stretch decision, and LLM adaptation metadata for each utterance.

## Testing Releases

Release testing is documented in [TESTING.md](TESTING.md). At minimum, every
script change should pass:

```bash
/bin/bash -n autodub_local.sh
/bin/bash ./autodub_local.sh --help
```

When the embedded Python worker changes, extract and compile it as described in
`TESTING.md`. Before a public GitHub release, regenerate and listen to both
reference outputs on the maintainer's machine.

Agent/developer workflow notes are documented in [AGENTS.md](AGENTS.md).

## Contributing

Please open GitHub issues for bugs, quality regressions, or requests for
enhancement. Pull requests are not the preferred workflow for this repository,
because the maintainer wants to test every change on the target machine before
publication.

If an issue needs to be very precise, include a small code snippet or suggested
patch fragment. This is optional; a clear description of the problem or requested
behavior is usually enough.

## Use, Licenses, And Service Terms

This project is released as CC0, but generated dubs are also affected by the
rights in the input media, the selected model licenses, the selected online
service terms, and any consent requirements for voices or speakers. This section
is a technical summary, not legal advice.

High-level guidance:

- `--tts-engine kokoro` is the most license-friendly preset TTS path currently
  available in the script: Kokoro code/weights are Apache-2.0.
- `--tts-engine xtts` uses XTTS v2 voice cloning. The XTTS v2 model license is
  non-commercial and also restricts use of model outputs.
- `--translation-method local` uses NLLB-200, whose model card lists CC-BY-NC
  and research/sentence-level intended use. Do not assume this path is suitable
  for commercial production.
- `--translation-method google` uses `deep-translator` and an unofficial online
  route to Google Translate. The Python package is permissively licensed, but
  service availability and permitted use are separate questions.
- `--tts-engine edge` uses the LGPLv3 `edge-tts` package and Microsoft's online
  Edge TTS service. For commercial or production TTS, Microsoft recommends using
  the official Azure AI Speech platform instead of relying on unofficial Edge
  service access.
- `--only-cloud` sends audio/text to AssemblyAI, Google Translate, Groq, and
  Microsoft Edge TTS. Use fully local mode when privacy or offline processing is
  more important than runtime.

Component references:

- **`autodub-local`** — This script.
  [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- **FFmpeg** — Audio/video extraction, stretching, and muxing. LGPL v2.1+ by
  default; GPL v2+ if built with GPL components.
  [License reference](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **OpenAI Whisper weights** — Local ASR through faster-whisper. MIT.
  [Project reference](https://github.com/openai/whisper).
- **faster-whisper** — Local ASR runtime. MIT.
  [Project reference](https://github.com/SYSTRAN/faster-whisper).
- **pyannote.audio** — Local diarization toolkit. MIT.
  [Project reference](https://github.com/pyannote/pyannote-audio).
- **`pyannote/speaker-diarization-3.1`** — Local diarization model. Gated
  Hugging Face model; verify the current model card and terms before
  redistribution or commercial use.
  [Model reference](https://huggingface.co/pyannote/speaker-diarization-3.1).
- **NLLB-200 distilled 600M** — Local translation. CC-BY-NC; intended for
  research and sentence-level machine translation, not production or document
  translation.
  [Model reference](https://huggingface.co/facebook/nllb-200-distilled-600M).
- **deep-translator** — Google Translate wrapper. MIT package; online provider
  terms are separate.
  [Project reference](https://github.com/nidhaloff/deep-translator).
- **Ollama** — Local LLM runner. MIT runner; model licenses vary.
  [Project reference](https://github.com/ollama/ollama).
- **Qwen3** — Default local LLM family. Apache-2.0 open-weight models according
  to Qwen. [Project reference](https://github.com/QwenLM/Qwen3).
- **GroqCloud** — Cloud Whisper and cloud LLM adaptation. Governed by Groq
  service terms, rate limits, data settings, and model licenses.
  [Service agreement](https://console.groq.com/docs/legal/services-agreement).
- **Groq Whisper Large v3** — Cloud ASR. OpenAI Whisper model served by Groq;
  Groq recommends chunking large audio.
  [Speech-to-text reference](https://console.groq.com/docs/speech-to-text).
- **GPT OSS 120B** — Default Groq LLM. Apache-2.0 open weights plus Groq terms.
  [Model reference](https://huggingface.co/openai/gpt-oss-120b).
- **AssemblyAI** — Cloud transcription and speaker labels. Service terms and
  free credits; new accounts currently receive free credits for STT-related
  services.
  [Billing and pricing reference](https://www.assemblyai.com/docs/billing-and-pricing).
- **Kokoro** — Local preset TTS. Apache-2.0 code and weights.
  [Model reference](https://huggingface.co/hexgrad/Kokoro-82M).
- **Coqui TTS toolkit** — XTTS runtime. MPL-2.0 toolkit; pretrained model
  licenses vary. [Project reference](https://github.com/coqui-ai/TTS).
- **XTTS v2 model** — Local voice cloning. Coqui Public Model License;
  non-commercial use of the model and its outputs.
  [License reference](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt).
- **edge-tts** — Python client for Edge TTS. LGPLv3 package, with one MIT file
  noted in its license.
  [License reference](https://github.com/rany2/edge-tts/blob/master/LICENSE).
- **Microsoft Edge TTS service** — Online preset TTS. Unofficial service access;
  Microsoft points commercial users to Azure AI Speech.
  [Service reference](https://learn.microsoft.com/en-us/answers/questions/2088770/are-opensource-edge-tts-free-for-commercial-use).

Cloud quota notes as of the last README update:

- Groq publishes per-model free-plan rate limits and says exact current limits
  are visible in the account console.
- AssemblyAI documentation states that new accounts receive free credits for
  pre-recorded STT, real-time STT, Voice Agent API, Speech Understanding, and
  Guardrails; LLM Gateway is not included in that free tier.
- Free quotas and terms can change. Check provider dashboards before planning
  multi-hour dubbing jobs.

## Limitations

- No lip-sync
- No subtitle generation
- Diarization can assign the wrong speaker or the wrong speaker count
- Cloud services require internet access, API keys, and available free-tier quota or billing
- `--only-cloud` avoids local AI model execution for ASR/diarization/LLM/TTS, but audio and text are sent to external services
- Groq Whisper provides transcription but not speaker diarization in this script
- AssemblyAI speaker labels can segment speakers differently from pyannote, so output timing and speaker ids may differ from local mode
- Pitch-based voice class assignment is a rough heuristic
- Per-speaker speed adjustments help preserve pacing, but they can still affect timing and may leave pauses when TTS is much shorter than the original slot
- XTTS voice cloning quality depends strongly on source audio and reference clip quality
- Kokoro TTS is local and lightweight, but it does not preserve the original speaker identity
- Edge TTS is online and unofficial through `edge-tts`; it can change, fail, or hit service limits
- `--no-gpu` controls this script's local processes and requests `num_gpu=0` from Ollama, but an externally managed Ollama server can still have its own runtime environment
- Google Translate uses an online unofficial route through `deep-translator` and may hit rate limits
- Local NLLB is slower but offline after model download
- Local LLM adaptation can shorten text, but it can still produce awkward output and should be reviewed for important material

For technical, scientific, legal, medical, or public publishing workflows, manually review the translated JSON and add a disclaimer that the dub was generated automatically.
