# Reusable Agent-Assisted Dubbing Workflow

Version: 2026-08-17

If you are a human using `autodub-local` without an AI assistant, use
`README.md` instead. This document is an instruction file for an AI agent that
has been asked to manage a dubbing job in this repository.

The user can start a future job by providing this file and a source video. The
agent must treat the preferences below as the default workflow, adapting file
names, languages, speaker count, voices, attribution, and technical parameters
to the new source.

Communicate with the user in the user's language. Drive the work from initial
inspection through the verified publication MP4; do not stop after analysis
when the user has authorized execution.

## Core Preferences

- Keep the main dubbing pipeline local unless the user explicitly authorizes a
  cloud service for a particular stage.
- Design every long stage to survive loss of Internet access and disconnection
  of the chat.
- Use cloud transcription only as an independent comparison source. Never
  replace the local transcript automatically with cloud output.
- Review both the source transcript and the complete translation. A local LLM
  adaptation pass is not a substitute for the agent's own review.
- Validate the actual voice used for every speaker, not merely the command-line
  voice map.
- Produce natural spoken continuity. Do not preserve long source timestamps
  when they create silence that breaks an Italian sentence.
- When shortening dubbed silence, cut the same duration from the video. Audio
  and video must remain on one shared edit timeline.
- Remove technical setup, connection tests, dead air, or irrelevant pre-roll
  before the substantive talk, unless the user asks to preserve it.
- Add an attribution watermark to the publication output. Include a license
  only when it is known; do not write that the license is unknown.
- Preserve source files, approved intermediates, backups, and known-good
  outputs. A revised render gets a new file name unless the user explicitly
  requests replacement.
- Never expose API keys, `.cloud_keys`, Hugging Face tokens, or other secrets.
- Before any local or mixed workflow, explicitly ask whether the user wants to
  force every supported local ML stage onto the CPU with `--no-gpu`, or allows
  the script to try compatible GPU acceleration. Do not infer this preference
  from the computer's owner or hardware.
- Explain that CPU-only execution can be substantially slower but may be useful
  for thermal, power, or compatibility reasons. If GPU use is allowed, verify
  the actual runtime and model compatibility before relying on it. Never change
  CPU/GPU clocks, voltages, or thermal policy without explicit authorization.

## Discover Before Asking

Inspect the workspace, video, caches, documentation, and available commands
before asking questions. Do not ask the user for information that can be found
locally.

Determine or ask for the following when it is not already supplied:

- Source video path and required output name.
- Source and target languages.
- Number of real speakers, their identities or roles when known, and their
  genders or vocal characteristics.
- TTS engine and one distinct voice per speaker.
- Whether original music or other non-speech audio must be retained. For a
  speech-only talk or video call, default to a clean dubbed voice track.
- Whether an auxiliary cloud transcript is authorized and which configured
  service may be used.
- For every local or mixed workflow, whether local ML stages must be CPU-only
  or may try compatible GPU acceleration. Ask this explicitly even when the
  hardware can be inspected.
- Original publication URL for attribution.
- License, only if known.
- Any required trims, exclusions, terminology, pronunciation, or naming rules.
- Whether the user wants the normal approval gates or authorizes the agent to
  apply sensible corrections directly.

The default watermark pattern is:

```text
Doppiato in <target language> con github.com/jsfan3/autodub-local; originale: <original-url>
```

Append `(<license>)` only when the user supplies or confirms the license.

## Workspace and Preflight

1. Work in the directory selected by the user.
2. Locate the repository before running project commands. If the current
   directory is not already an `autodub-local` checkout and no checkout is
   available in the selected workspace, obtain it from its official source:

```bash
git clone https://github.com/jsfan3/autodub-local.git
cd autodub-local
```

A plain `git clone` is intentional: do not make a shallow or partial clone. If
an existing checkout is available, reuse it. Inspect its status and remote
before fetching or updating it, and never overwrite local changes merely to
match the remote.

3. Once inside the repository, inspect before editing:

```bash
pwd
git remote get-url origin || true
git status --short || true
rg --files | sed -n '1,160p'
/bin/bash ./autodub-local.sh --help
```

4. Inspect the source with `ffprobe`: duration, streams, codecs, dimensions,
   frame rate, audio sample rate, and channel count.
5. Check free disk space and estimate the space required for extracted PCM,
   TTS segments, clean masters, previews, and final renders.
6. Inspect `.autodub-local` before downloading or reinstalling anything. Reuse
   a valid environment and cached models.
7. Report only whether credential files or tokens exist, never their contents.
8. Before an offline stage, verify that every required model and Python package
   is already available. Download missing assets while Internet access exists.
9. Inspect GPU and runtime compatibility so the user's CPU/GPU choice is
   informed. When GPU use is allowed, confirm that the installed runtime and
   selected models can actually use it; otherwise explain the incompatibility
   and use a known-working CPU path. When CPU-only execution is selected, add
   `--no-gpu` to every applicable script invocation.

Record exact paths and SHA-256 hashes for operational JSON files before and
after every approved review.

## Required Artifacts

Use predictable, per-video names. Keep at least:

- A workflow/status Markdown file.
- One log, PID file, and machine-readable status file per long stage.
- Local transcript, diarization, utterance, and translated JSON caches.
- A separate cloud-reference cache when cloud ASR is used.
- Proposed and approved ASR review reports and JSON files.
- Proposed and approved translation review reports and JSON files.
- Backups named with the previous SHA-256 before replacing operational caches.
- TTS speaker profiles and manifest.
- A natural-pause edit decision list (EDL) and boundary report.
- Watermark preview and final QA report.

Status files should begin with exactly one of:

```text
RUNNING
SUCCESS
FAILED
```

Include timestamps, stage name, important input hashes, output paths, and the
failure reason when applicable.

## Stage 0: Short Pilot

Before processing a long video, normally create a fully automatic pilot of one
or two representative minutes. The excerpt should contain every important
speaker when possible; it need not be the literal first two minutes when those
contain only technical setup.

The pilot must test the same key choices intended for the full job:

- ASR and diarization.
- Explicit expected speaker count.
- Strict speaker-to-voice mapping.
- Translation method and local LLM adaptation.
- TTS voices and pacing.
- Basic mix/mux behavior.
- Active-voice edge detection and natural-pause video/audio cuts.

Verify that distinct speakers receive distinct intended voices by inspecting
speaker profiles and the TTS manifest. Do not infer success merely because two
voices were passed on the command line.

Let the user assess pronunciation, translation style, voice suitability, and
general direction. Run `silencedetect` on the pilot and do not present a raw
timestamp-anchored pilot with multi-second gaps as representative of the final
workflow. When the user approves the approach, remove disposable pilot outputs
if requested, but do not remove shared models or caches needed by the full run.

## Stage 1: Local ASR and Diarization

Run local ASR, local diarization, and utterance reconstruction as a separate,
checkpointed stage. Do not launch translation or TTS until the source text and
speaker structure are ready for review.

- Set the known speaker count explicitly. Do not leave it on `auto` when the
  user has stated how many people speak.
- Validate that the resulting speaker set contains exactly the expected labels.
- Inspect empty text, invalid intervals, mixed-speaker utterances, implausible
  short fragments, and large timestamp gaps.
- Identify which diarization label corresponds to each real person.
- Store counts, label mapping, hashes, and validation results in the workflow
  report.

For cached local models, force offline behavior where supported:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /bin/bash ./autodub-local.sh <adapted arguments>
```

Use only options confirmed by the current `--help` and repository code.

## Stage 2: Auxiliary Cloud Transcript

Run this stage only when authorized or when unresolved local ASR doubts justify
it. Keep it isolated from the operational local cache.

- Prefer a cloud service capable of diarization.
- Make upload, polling, retry, and rate-limit handling resumable.
- Write cloud job identifiers and status without exposing credentials.
- Align cloud words or segments to the local utterances.
- Determine the cloud-to-local speaker-label mapping from evidence.
- Treat cloud output as another fallible witness. Resolve disagreements using
  audio, source-language grammar, subject matter, names, and context.

The cloud transcript must not silently overwrite the local transcript.

## Stage 3: ASR Review

Review the entire local ASR result, using the cloud transcript when available.
Check more than obvious spelling mistakes:

- Names, institutions, products, technical and medical terminology.
- Negation, numbers, units, abbreviations, and quoted questions.
- Speaker assignments and turn boundaries.
- Sentences accidentally combining two speakers.
- Repeated or hallucinated material.
- Non-speech noises and spurious interjections.
- Timestamp changes required after splitting, merging, or deleting fragments.

Create a proposed JSON and a complete Markdown report before changing the
operational cache. Report the number of source utterances affected, resulting
utterance count, speaker changes, boundary changes, and unresolved doubts.

Default approval gate: show the proposed corrections and wait. If the user has
explicitly authorized direct application, apply only coherent corrections,
back up the old JSON, validate the new JSON, and record both hashes.

## Stage 4: Local Translation and Adaptation

Run local translation and local LLM timing adaptation as a separate offline
stage, stopping before TTS. Reuse the approved ASR, diarization, and utterance
caches.

Illustrative command shape, to be adapted to the current CLI:

```bash
/bin/bash ./autodub-local.sh \
  --input <input-video> \
  --output <unused-pre-tts-output> \
  --source-lang <source-code> \
  --target-lang <target-code> \
  --translation-method local \
  --tts-engine <engine> \
  --num-speakers <count> \
  --stop-after-translation \
  --llm-adapt auto \
  --tts-voice-map SPEAKER_00=<voice-a>,SPEAKER_01=<voice-b> \
  --tts-voice-map-strict
```

Add or remove speaker mappings to match the actual speaker count exactly.
Add `--no-gpu` when the user selected CPU-only execution; omit it when the user
allowed compatible GPU acceleration.

## Stage 5: Complete Translation Review

The agent must review every translated utterance against the approved source
text. Do not delegate final judgment blindly to a local or external LLM.

Review for:

- Meaning, negation, causality, chronology, quantities, and modality.
- Domain terminology and consistent recurring expressions.
- Names and the pronunciation that TTS will produce.
- Natural target-language syntax and register.
- Spurious fillers, ASR noises, duplicated words, and text that should not be
  spoken.
- TTS-friendly readings of acronyms, formulae, numbers, and abbreviations.
- Overlong text that cannot fit its slot.
- Over-shortened text that would create several seconds of dead air.
- Segment boundaries that split one syntactic phrase across multiple clips.

Preserve what the speakers claim. Do not silently fact-check, soften, or
correct medical, political, legal, or technical opinions unless the user asks
for editorial intervention. Accuracy here means fidelity to the source.

Character budgets are diagnostic, not the sole objective. A line fitting under
its maximum budget can still be too short for the original slot. Record both
over-budget and severe under-duration risks.

Create:

```text
<stem>.translated.<lang>.reviewed.json
<stem>-translation-review.md
<stem>-translation-review-meta.json
```

Report all changed items, uncertainties, total character counts, and timing
checks. Use the same approval rule as the ASR review: propose first by default;
apply directly only when the user has granted that authority. Always back up
the operational translation before replacement.

## Stage 6: TTS, Profiles, and Clean Master

Run TTS only after the reviewed translation is operational and its hash is
verified.

- Use one explicit voice per expected speaker and strict mapping.
- Validate speaker profiles after generation.
- Validate every manifest item: requested speaker, actual speaker, actual
  voice, sample rate, nonempty audio, text hash, and duration.
- Ensure cache reuse is based on compatible configuration and reviewed-text
  hashes.
- Keep the clean PCM dubbed track and clean video master. The publication
  version should be derived from these, not from a previously compressed MP4.

Inspect both timing failure modes:

1. **Overlaps:** TTS continues into the next utterance.
2. **Gaps:** TTS finishes early and leaves unnatural silence before the next
   utterance.

`overlap_warnings=0` is necessary but not sufficient. Calculate for every
boundary:

```text
manifest_gap = next.start - (current.start + current.tts_duration)
```

Also measure actual active speech inside every TTS file. Many engines add
leading and trailing silence, so the pause heard by the user is:

```text
perceived_gap = next_active_voice_start - current_active_voice_end
```

Warnings such as `Skipping unsafe TTS expansion` often predict large gaps.
Count them and audit the affected boundaries. Never declare timing successful
after checking overlaps alone.

## Stage 7: Natural-Pause Timeline

The default publication version should use a compact semantic timeline. Do not
anchor every TTS clip to the original ASR timestamp when that produces long
silence inside a sentence.

Review every boundary and classify it semantically. The following targets are
the preferred defaults for Italian; adjust only when the target language or
context clearly requires it:

| Boundary | Target perceived pause |
|---|---:|
| Unpunctuated continuation inside one sentence | 0.10 s |
| Comma or short clause break | about 0.17 s |
| Colon or semicolon | about 0.27 s |
| Completed sentence | about 0.40 s |
| Question, exclamation, or emphatic sentence | about 0.47 s |
| Change of speaker | about 0.60 s |

Rules:

- If the existing perceived gap is already shorter than the target, do not
  lengthen it automatically.
- Preserve a longer pause only when listening or context shows that it is a
  deliberate rhetorical pause.
- Measure voice activity rather than treating the whole TTS file as speech.
- Do not cut phonemes. Leave a small safety margin around detected activity.
- Resolve negative gaps or real overlaps explicitly; they cannot be repaired
  by removing silence.
- Remove the same intervals from audio and video. A jump cut is preferable to
  several seconds of silent moving lips in a talk or interview.
- For visually complex footage, inspect every problematic cut and use the
  least distracting frame-accurate edit that preserves the shared timeline.
- Quantize edits to the video frame grid. Make audio removals sample-accurate
  and equal in duration to video removals so synchronization cannot drift over
  hundreds of cuts.
- Save the EDL as structured JSON or CSV and generate a human-readable boundary
  report with original gap, perceived gap, class, retained pause, removed
  duration, and adjacent text.

Prefer rendering from the source video plus the reviewed PCM track so video and
audio each undergo only one publication encode.

Before the full render, create a three-to-five-minute preview containing many
boundaries. Listen to it and run `silencedetect`. Compare it with the
uncompacted master. A normal publication candidate should have no unexplained
pause of one second or more; list any deliberate exception.

## Long-Running and Offline Orchestration

Each long stage must be independently executable and monitorable outside the
chat.

Use a wrapper script that:

- Writes its own PID after launch.
- Verifies hashes of approved inputs.
- Writes `RUNNING` before starting.
- Redirects complete output to a stable log.
- Validates artifacts before writing `SUCCESS`.
- Writes `FAILED`, an exit code, and a reason on failure.
- Can resume compatible caches without repeating completed work.

Detach robustly with a platform-appropriate mechanism. On Linux, for example:

```bash
setsid -f /bin/bash ./run_stage.sh
```

On macOS, where `setsid` is not part of the base system, redirect every stream
and use `nohup`:

```bash
nohup /bin/bash ./run_stage.sh \
  >run_stage.console.log 2>&1 </dev/null &
```

Do not assume a background process survived merely because the launch command
returned. Recheck the PID, status file, process tree, and first meaningful log
lines.

Provide a monitor command or script that reports progress periodically and
ends with a clear instruction such as:

```text
STAGE COMPLETED. Reconnect to the Internet and ask the agent to inspect the result.
```

When the user is about to disconnect, explicitly state whether it is safe,
give a realistic remaining-time estimate, and identify what event requires
reconnection. Online stages need retry and resume logic; local stages should
use offline model flags after preflight.

## Trim, Watermark, and Publication Render

Find the real beginning of the substantive content. Exclude initial connection
tests, duplicated audio checks, waiting, and other technical setup by default.
Record the chosen source timestamp.

Create the watermark through a text file when escaping would be fragile. It may
wrap onto two lines at low resolution. Choose font size and margins relative to
the actual dimensions; do not copy a size suitable for a larger video.

Generate and inspect a real preview frame before the full encode. The watermark
must be legible, inside the frame, unobtrusive, and must not cover essential
content.

Typical publication characteristics:

- Source video plus final PCM dubbed audio.
- Shared EDL applied to both streams.
- H.264 video, `yuv420p`.
- AAC audio at a bitrate valid for its sample rate and channel count.
- `+faststart`.
- A hidden or clearly temporary partial output.
- Atomic rename to the requested final name only after validation.
- Detached rendering with `ffmpeg -progress` for a long video.

Never overwrite a known-good earlier render when testing a timing correction.
Use a descriptive suffix such as `_natural-pauses`, `_reviewed`, or another
name agreed with the user.

## Default Approval Gates

Unless the user explicitly authorizes direct action, pause at these points:

1. Pilot output and voice direction.
2. Proposed ASR corrections before operational replacement.
3. Proposed translation corrections before operational replacement.
4. Watermark preview before the full publication encode.

Do not repeatedly ask for approval after the user has clearly granted authority
to apply sensible corrections and finish the render. New instructions from the
user override an earlier default gate.

Always ask before deleting source material, approved reviews, known-good
outputs, or expensive caches.

## Final Quality Assurance

Do not report completion until all applicable checks pass:

- Final path is exactly the agreed output path.
- Earlier output files that had to be preserved still have their previous
  SHA-256 hashes.
- `ffprobe` confirms expected duration, start time, codecs, dimensions, frame
  rate, audio sample rate, channels, and stream count.
- Audio and video durations match within a small frame-level tolerance.
- Frame count matches the EDL prediction.
- A complete decode of all selected audio and video streams finishes with no
  errors.
- The final encoded audio is not silent or clipped. Record mean and peak level.
- `silencedetect` on the final encoded audio reports median, p90, maximum, total
  silence, and count of pauses over one second.
- Every pause over one second is either corrected or documented as deliberate.
- Speaker profiles and TTS manifest prove the intended voice for every speaker.
- Initial, middle, and final frames are visually inspected for watermark,
  framing, corruption, and unexpected black output.
- The beginning contains the talk, not the discarded technical setup.
- The ending is not clipped and does not retain unnecessary padding.
- No dubbing, monitoring, or `ffmpeg` process required by the job remains active.
- Final size and SHA-256 are recorded.

Create a concise final QA Markdown report and update the workflow status file.
Tell the user the exact final path, duration, important pause statistics, and
any residual pronunciation or source-quality risk.

## Cleanup and Preservation

Keep:

- Source video.
- Approved ASR and translation JSON files and their backups.
- Speaker profiles, TTS manifest, and reusable TTS clips.
- Clean PCM track or another lossless reusable master.
- EDL and review reports.
- Final QA report, logs, and status files.
- Publication MP4 and any earlier version the user asked to preserve.

Remove disposable pilots, failed partial renders, and obsolete external-model
drafts only when it is clear they are no longer needed or the user requests
cleanup. Never run a broad cleanup command against shared caches.

## Optional Publication Support

When requested, prepare publication text in the user's language covering:

- Who is speaking.
- The subject and relevance of the talk.
- That the version was dubbed with `autodub-local`.
- The original source URL.
- The original license only when known.
