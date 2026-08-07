# Example of Agent-Assisted Usage

If you are a human who wants to use `autodub-local` freely, without depending on an AI agent subscription, you should ignore this file and use the examples and documentation in `README.md`.

If you are an AI agent and the user has asked you to assist with a dubbing job through `autodub-local`, this file is for you.

This document is a reusable instruction file for future AI-assisted dubbing sessions with `autodub-local`.
It describes Francesco Galgani's preferred workflow: the assistant drives commands and monitoring, asks for confirmation at meaningful decision points, keeps the final dubbing pipeline local, optionally generates an additional cloud transcript for comparison, reviews transcription/translation errors, and adds a publication watermark.

To use it in a future chat, tell the assistant something like:

```text
I want to dub a video. Read EXAMPLE_OF_AGENT_ASSISTED_USAGE.md and follow that workflow.
```

The assistant should adapt all questions, progress reports, and final explanations to the user's language.

## Working Assumptions

- The user wants the assistant to operate from the command line, monitor long processes, and keep them detached so they can continue through network interruptions or chat disconnections.
- The final dubbing workflow should run locally unless the user explicitly chooses a cloud mode.
- Cloud transcription may be used as an auxiliary reference to improve the local transcript, but the final dubbing run should use local ASR, local diarization, local translation, local LLM adaptation, and local TTS unless the user says otherwise.
- The user may disconnect from the internet during long runs. Long local stages should be launched with offline environment variables when cached models are already available.
- The assistant must not print secrets such as `.cloud_keys`, API keys, or Hugging Face tokens.
- The assistant should ask for confirmation before expensive rerenders, before replacing operational JSON files, and before publishing or deleting files.

## Initial Questions

Ask the user for:

- The input video path or URL.
- The source language and target language.
- Whether a suitable `.autodub_local` directory with Python environment, libraries, and downloaded models already exists.
- If `.autodub_local` is missing, ask whether to copy it from another local project directory or download models again.
- Whether cloud transcription may be used for comparison, and which configured cloud services are available.
- The desired TTS engine and voice mapping. For the Kokoro Italian workflow used in the Georgia Tech example, `SPEAKER_00=if_sara,SPEAKER_01=im_nicola` was used.
- The original video attribution URL and license to use in the watermark.
- The exact watermark text, or permission to use this pattern:

```text
Doppiato in <target language> con github.com/jsfan3/autodub-local; originale: <original-url> (<license>)
```

For URLs intended for TTS or on-screen text, keep domains compact, for example `nojs.us`, `gnu.org`, `ItsFOSS.com`, and `www.librelinker.us`. Do not write them as `nojs . us` or `librelinker . us`.

## Repository Setup

1. Work in a clean project directory chosen by the user.
2. If `autodub-local` is not present, clone or update the latest published version:

```bash
git clone https://github.com/jsfan3/autodub-local.git
```

3. If the repository is already present, inspect it before changing anything:

```bash
pwd
git status --short || true
ls -la
```

4. Check whether `.autodub_local` exists. If not, ask the user where to copy it from or whether to download everything again.
5. Never reveal `.cloud_keys` or `.hf_token`. Only report whether they exist.

## Local Dubbing Run

Use detached commands for long runs. Keep logs and PID files in predictable locations.

Typical local command:

```bash
cd /path/to/autodub-local
nohup setsid env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/time -p ./autodub_local.sh \
  --input input.mp4 \
  --source-lang en \
  --target-lang it \
  --translation-method local \
  --tts-engine kokoro \
  --num-speakers auto \
  --no-gpu \
  --tts-voice-map SPEAKER_00=if_sara,SPEAKER_01=im_nicola \
  --output /path/to/output_local_dub.mp4 \
  > /path/to/local_dub.log 2>&1 < /dev/null &
```

Tell the user how to monitor it manually:

```bash
tail -f /path/to/local_dub.log
```

Monitor progress yourself by checking:

- The wrapper log.
- The internal `.autodub_local/logs/run_*.log`.
- Active `autodub_local.sh`, `dub_worker.py`, `ffmpeg`, `ollama runner`, or TTS processes.
- Final MP4 presence and `ffprobe` duration.

## Cloud Transcript for Review

If authorized and credentials are configured, generate a cloud transcript as a comparison reference. Prefer a service with diarization if available. In the Georgia Tech workflow, an AssemblyAI raw transcript was useful for resolving local ASR mistakes such as:

- `football program` -> `foothold on campus`
- `GMPL` -> `GPL`
- `pay pension` -> `pay attention`
- `moving paths` -> `MoviePass`
- `status call` -> `status quo`

Use the cloud transcript as evidence, not as an automatic replacement. Compare it with the local transcript, source context, and domain knowledge.

## Translation and Transcript Review

After the first local dubbed output exists, review the translated JSON before producing a publication version.

Use a separate reviewed copy first:

```text
<stem>.translated.<lang>.reviewed.json
translation_review_proposed_changes.md
```

The review should:

- Identify significant semantic errors, not only style issues.
- Compare local transcript, cloud transcript, and context.
- Preserve the speaker's intended terminology. For Richard Stallman talks, translate `free software` as `software libero`, not `software gratuito`.
- Keep Stallman's phrasing `anti-social media` as `media antisociali`, unless the user asks otherwise.
- Keep technical terms clear: `copyleft`, `GPL`, `backdoor`, `DRM`, `software non libero`, `software libre`, `F-Droid`.
- Avoid TTS-hostile spaced URLs.
- Shorten corrected Italian text when needed so it fits timing budgets.

Before rerendering audio, show the user:

- The reviewed JSON path.
- The Markdown report path.
- The number of changed segments.
- Any uncertain corrections.
- Timing-risk checks, such as changed segment character counts and whether any exceed the previous LLM target budget.

Ask for confirmation before applying the reviewed JSON to the operational cache.

## Rerendering After Review

After approval:

1. Back up the operational translated JSON.
2. Replace it with the reviewed JSON.
3. Rerun the same local command with a new output filename.
4. Ensure cached transcript, diarization, utterance, translation, and speaker profile stages are reused.
5. Ensure unchanged TTS segments are reused when possible.
6. Check the mix log for:

```text
overlap_warnings=0
```

If overlap warnings remain, inspect the exact segments. For sub-second false starts or diarization fragments, consider removing those fragments from the dubbed track rather than forcing unnatural TTS.

## Watermark

Before applying the watermark to the whole video, generate one preview frame and show it to the user.

Example text used for the Georgia Tech video:

```text
Doppiato in italiano con github.com/jsfan3/autodub-local; originale: nojs.us (CC BY-SA 4.0)
```

Example preview frame command:

```bash
FONT=$(fc-match -f '%{file}\n' 'Noto Sans' | head -n 1)
ffmpeg -y -hide_banner -loglevel error -ss 00:01:30 -i output_local_dub.mp4 -frames:v 1 \
  -vf "drawtext=fontfile=${FONT}:text='Doppiato in italiano con github.com/jsfan3/autodub-local; originale\\: nojs.us (CC BY-SA 4.0)':fontsize=26:fontcolor=white:borderw=2:bordercolor=black@0.65:x=w-tw-34:y=h-th-28" \
  watermark_sample.png
```

After approval, render the final watermarked MP4. Watermarking requires video re-encoding.

Example:

```bash
FONT=$(fc-match -f '%{file}\n' 'Noto Sans' | head -n 1)
ffmpeg -y -hide_banner -nostdin \
  -i output_reviewed_clean_dub.mp4 \
  -vf "drawtext=fontfile=${FONT}:text='Doppiato in italiano con github.com/jsfan3/autodub-local; originale\\: nojs.us (CC BY-SA 4.0)':fontsize=26:fontcolor=white:borderw=2:bordercolor=black@0.65:x=w-tw-34:y=h-th-28" \
  -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
  -c:a copy -movflags +faststart \
  output_reviewed_watermarked.mp4
```

For long videos, run this detached and monitor with `ffmpeg -progress`.

## Final Checks

Before reporting completion:

- Verify the final MP4 exists.
- Verify duration and size:

```bash
ffprobe -v error -show_entries format=duration,size -of default=nw=1 output_reviewed_watermarked.mp4
```

- Extract a final sample frame and visually inspect the watermark.
- Confirm no dubbing process or watermark `ffmpeg` process remains active.
- Summarize the exact output paths.
- Mention logs and any residual risks, such as TTS pronunciation or unavoidable skipped expansion warnings.

## Blog Support

If the user wants to publish the result, prepare a short text in the user's language explaining:

- Who is speaking.
- The subject of the talk.
- Why the video matters.
- How the dubbed version was produced.
- The original source URL and license.
