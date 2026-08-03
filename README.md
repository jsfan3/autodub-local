# autodub_local

Local, no-cloud video dubbing for MP4 recordings using open-source components.

By [Francesco Galgani](https://www.informatica-libera.net/), license [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

This project replaces the audio track of one or more MP4 files with a translated, synthesized dub while keeping the original video stream intact. It is designed for long recordings such as webinars, livestreams, interviews, and meetings where perfect lip-sync is not required.

## Quick start: Test with the included sample video

**Highly recommended for first-time users**: Before processing your own videos, test the pipeline with the included sample video `test.mp4` to verify that everything works correctly on your system.

## What it does

The pipeline is:

1. Extract mono audio from each MP4
2. Transcribe speech with `faster-whisper`
3. Run speaker diarization with `pyannote.audio`
4. Rebuild larger speaker turns (“utterances”)
5. Translate the utterances with **NLLB (local)** or **Google Translate (online)**
6. Generate target-language speech with XTTS v2 using per-speaker reference clips
7. Assemble the dubbed audio timeline
8. Mux the new audio into the original MP4 without re-encoding the video stream

The script supports checkpoint/resume, so it can be interrupted and restarted without redoing every completed step.

**Notes on Google Translate:**
- Requires internet connection
- Uses the [`deep-translator` library](https://github.com/nidhaloff/deep-translator)
- May have rate limits for large batches
- Automatically installed when selected
- Translation results are cached in the `.translated.json` file like NLLB

## Tested environment

This script has been tested on:

- **OS:** Linux Mint 22
- **Python:** 3.12.3
- **Laptop:** ASUS ROG GL703VD
- **CPU:** Intel Core i7-7700HQ
- **RAM:** 16 GB
- **GPU:** NVIDIA GeForce GTX 1050 Mobile, 4 GB VRAM

I did my best to get the script to work on macOS Sequoia (Intel), but I wasn't able to.

### Notes on GPU behavior on the tested machine

On the tested system, the script works correctly, but only part of the pipeline runs on GPU.

On this hardware, `faster-whisper` can use CUDA, while the PyTorch-based stages (`pyannote.audio` diarization and XTTS) fall back to CPU because the GTX 1050 (`sm_61`) is not supported by the installed PyTorch/CUDA build.

On newer NVIDIA GPUs supported by the installed PyTorch build, **the script is designed to run the entire pipeline on GPU. CPU is used only as a fallback when GPU execution is not available or not compatible**.

## Compatibility notes

The current script is primarily aimed at:

- Linux Mint / Ubuntu / Debian-like systems
- systems with `apt-get`
- local execution with enough disk space for models and temporary audio files (for example, dubbing a two-hour video requires about 20 GiB for downloaded software and temporary audio files)

It may work on other Linux distributions, but the automatic system dependency installation assumes `apt-get`.

## Runtime expectations

This is **not** a quick script.

A first run can take a very long time because it may need to download:

- PyTorch and CUDA runtime packages
- Whisper model weights
- NLLB model weights
- XTTS model weights
- Python dependencies

On older hardware or CPU-heavy fallback paths, a single long video can require **many hours**.

For multi-hour videos, especially on hardware similar to the tested machine, it is normal for the full pipeline to take a very long time.

## Requirements

The script will try to install missing system packages automatically:

- `ffmpeg`
- `git-lfs`
- `python3-venv`
- `python3-pip`

This may require `sudo`.

An internet connection is typically required on the first run to download models and Python packages.

## First-time setup

### 1. Hugging Face token for pyannote diarization

A **Hugging Face READ token** is required for local pyannote diarization.

Before the first run:

1. Accept the model terms for:
   - `pyannote/speaker-diarization-community-1`
2. Create a Hugging Face **READ** token

The script will prompt for the token if it is not already available and will store it in:

```text
./.hf_token
```

with restricted permissions.

### 2. XTTS license confirmation

XTTS v2 may prompt for license acceptance on first use.

The current toolchain may show a Coqui Public Model License prompt interactively. In practice, this usually means typing `y` once on the first run.

## Basic usage

Place the script in the same directory as the videos to process, then run:

```bash
chmod +x autodub_local.sh
./autodub_local.sh
```

### Input files

The script automatically looks for input videos in its own directory.

If no hard-coded preferred filenames are found, it processes **all `*.mp4` files** in that directory.

At the moment, the preferred filenames are defined inside the script itself and are not configurable through an environment variable.

### Output naming

Output files are created next to the original MP4 files with this pattern:

```text
<original_stem>_<TARGET_LANG_UPPER>_dub.mp4
```

Example:

```text
MyVideo_IT_dub.mp4
```

## Configuration

The script is configured with **environment variables**, not CLI flags.

You do **not** need to edit the script to process different MP4 files as long as they are in the same directory as the script.

### Common environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `TRANSLATION_METHOD` | `local` | Translation method: `local` (NLLB) or `google` (Google Translate) |
| `SOURCE_LANG` | `ru` | Source language for Whisper |
| `TARGET_LANG` | `it` | Target language for TTS/output naming |
| `NLLB_SRC_LANG` | `rus_Cyrl` | Source language code for NLLB |
| `NLLB_TGT_LANG` | `ita_Latn` | Target language code for NLLB |
| `WHISPER_MODEL` | `medium` | Whisper model size |
| `NUM_SPEAKERS` | `2` | Expected number of speakers for diarization |
| `MAX_REF_CLIPS` | `3` | Reference clips per speaker for XTTS |
| `AAC_BITRATE` | `192k` | AAC bitrate for the muxed output |
| `LOG_LEVEL` | `INFO` | Python logging verbosity |

### XTTS tuning variables

| Variable | Default |
|---|---:|
| `XTTS_MAX_CHARS` | `180` |
| `XTTS_CHAR_LIMIT_MARGIN` | `20` |
| `XTTS_SPEED` | `1.0` |
| `XTTS_TEMPERATURE` | `0.65` |
| `XTTS_REPETITION_PENALTY` | `2.0` |
| `XTTS_INTER_CHUNK_SILENCE_MS` | `120` |

### ASR tuning variables

| Variable | Default |
|---|---:|
| `ASR_BEAM` | `5` |
| `ASR_VAD` | `true` |
| `ASR_COMPUTE_GPU` | `int8_float16` |
| `ASR_COMPUTE_CPU` | `int8` |

### Examples

Default Russian → Italian run (local NLLB):

```bash
./autodub_local.sh
```

Explicit Russian → Italian run with local NLLB:

```bash
TRANSLATION_METHOD=local \
SOURCE_LANG=ru \
TARGET_LANG=it \
NLLB_SRC_LANG=rus_Cyrl \
NLLB_TGT_LANG=ita_Latn \
NUM_SPEAKERS=2 \
./autodub_local.sh
```

English → Italian with local NLLB:

```bash
TRANSLATION_METHOD=local \
SOURCE_LANG=en \
TARGET_LANG=it \
NLLB_SRC_LANG=eng_Latn \
NLLB_TGT_LANG=ita_Latn \
NUM_SPEAKERS=2 \
./autodub_local.sh
```

English → Italian with Google Translate:

```bash
TRANSLATION_METHOD=google \
SOURCE_LANG=en \
TARGET_LANG=it \
NUM_SPEAKERS=2 \
./autodub_local.sh
```

## Supported languages

### Whisper / NLLB

The script includes an internal mapping from common Whisper language codes to NLLB language codes. It supports a broad set of language pairs through NLLB.

### XTTS target languages

The current script maps `TARGET_LANG` to XTTS for these target languages:

- `it`, `en`, `es`, `fr`, `de`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh`, `ja`, `hu`, `ko`, `hi`

If the selected `TARGET_LANG` is not supported by XTTS in the script, the run will fail at the TTS stage.

## Checkpoint/resume behavior

The script stores intermediate results under:

```text
./.autodub_local/
```

Each video gets its own working directory containing intermediate files such as:

- extracted mono audio
- transcript JSON
- diarization JSON
- utterances JSON
- translated JSON
- TTS segment WAV files
- manifest JSON
- dubbed WAV

If the script is interrupted, rerunning it should reuse completed steps whenever possible.

## Logs

Logs are written to:

```text
./.autodub_local/logs/
```

Each run creates a timestamped log file.

## Local model/cache layout

The script keeps most downloaded assets under the project-local working directory:

```text
./.autodub_local/models/
```

This includes Hugging Face caches and the XTTS cache used by the current script version.

## Manual translation review and editing

Yes — the translation can be reviewed and edited manually before TTS.

For each processed video, the translated utterances are stored in:

```text
./.autodub_local/<video_stem>/<video_stem>.translated.json
```

This file contains:

- source text
- translated text (`text_it` for Italian in the tested workflow)
- speaker labels
- timestamps

### Recommended manual review workflow

1. Run the script until translation is complete
2. Stop the script
3. Open and edit the translated JSON manually
4. Remove only the TTS/audio outputs for that video
5. Restart the script

### If the translation was changed after TTS had already started

Delete these for the specific video:

```text
./.autodub_local/<video_stem>/tts_segments/
./.autodub_local/<video_stem>/<video_stem>.manifest.json
./.autodub_local/<video_stem>/<video_stem>.it.wav
./<video_stem>_<TARGET_LANG_UPPER>_dub.mp4
```

Keep these, because they are expensive to rebuild and usually still valid:

```text
./.autodub_local/<video_stem>/<video_stem>.mono16k.wav
./.autodub_local/<video_stem>/<video_stem>.transcript.json
./.autodub_local/<video_stem>/<video_stem>.diarization.json
./.autodub_local/<video_stem>/<video_stem>.utterances.json
./.autodub_local/<video_stem>/<video_stem>.translated.json
```

## What “good enough” means here

This project is intended for **practical local dubbing**, not for certified translation.

For technical, scientific, legal, or medical recordings:

- the transcript may contain ASR errors
- the translation may contain omissions or distortions
- the TTS can sound unnatural on difficult segments
- manual review is strongly recommended before publishing

## Limitations

- No lip-sync
- No subtitle generation in the current workflow
- Quality depends heavily on source audio quality and ASR accuracy
- Speaker diarization can still make mistakes
- Long segments are more fragile than short segments
- First-run setup is heavy and can download many gigabytes
- Old NVIDIA GPUs may only accelerate part of the pipeline

## Publishing note

If the output is published publicly, it is strongly recommended to add a clear disclaimer saying that:

- the dub was generated automatically
- it may contain transcription/translation/synthesis errors
- the original recording should be considered authoritative in case of doubt

## Directory overview

```text
project/
├── autodub_local.sh
├── your_video.mp4
└── .autodub_local/
    ├── logs/
    ├── models/
    ├── venv/
    ├── cache/
    ├── xdg_data/
    └── your_video/
        ├── your_video.mono16k.wav
        ├── your_video.transcript.json
        ├── your_video.diarization.json
        ├── your_video.utterances.json
        ├── your_video.translated.json
        ├── tts_segments/
        ├── your_video.manifest.json
        └── your_video.it.wav
```

## Known Issue: Very Short Speech Interventions (< 4 seconds)

The voice cloning engine (Coqui XTTS) requires a sufficient audio sample to accurately replicate a person's voice. By default, this script is configured to extract reference voice clips **only from segments where a speaker talks continuously for at least 4 seconds** (`if dur >= 4.0:`).

If your video features a speaker who only says a very brief phrase (e.g., a 1 or 2-second interruption like "Yes" or "Thank you"), the script will skip extracting a voice profile for them. Consequently, when the pipeline reaches the dubbing phase and tries to synthesize audio for that speaker, **the script will crash** with the following error:

```text
RuntimeError: Speaker 'SPEAKER_XX' is not present in the XTTS cache
```

### How to Fix It (2 Options)

When the script finishes the translation phase, it will pause and ask you:
`Continue with dubbing now? [Y/n]:`

**Do not press `Y` immediately.** Instead, open the generated files in the `.autodub_local/<video_name>/` folder and choose one of the following workarounds:

**Option 1: Ignore the short intervention (Recommended for unimportant lines)**
If the brief interruption is not important (e.g., someone just saying "Yeah"), you can simply delete it from the translation queue.
1. Open the `<video_name>.translated.json` file.
2. Find the JSON block `{ ... }` corresponding to the short utterance and delete it. Make sure the remaining JSON structure remains valid (correct commas).
3. Save the file, go back to your terminal, and press `Y`.
4. The script will proceed to dub only the main speakers, and the brief interruption will simply be muted in the final video.

**Option 2: Force the voice cloning for the short intervention**
If the brief phrase is important and must be dubbed, you need to "trick" the script into extracting enough audio for that speaker.
1. Open the `<video_name>.diarization.json` file.
2. Find the segment corresponding to the short phrase. It will look something like this: `{"start": 120.0, "end": 121.5, "speaker": "SPEAKER_01"}`.
3. Modify the `"end"` value by adding at least 4 seconds (e.g., change `121.5` to `125.5`).
4. Save the file.
5. *(Important)* If a file named `<video_name>.utterances.json` already exists in that folder, delete it so the script will regenerate it using the new extended timeline.
6. Go back to your terminal and press `Y`.
7. The script will now extract the 1.5 seconds of speech plus the following 3.5 seconds of silence/background noise. This provides enough data for XTTS to clone the voice. Note that because the sample is very short, the resulting cloned voice may sound slightly artificial.

