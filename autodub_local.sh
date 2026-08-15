#!/usr/bin/env bash
# ============================================================
# autodub-local 2.1
# Author: Francesco Galgani
# Repository: https://github.com/jsfan3/autodub-local
# License: CC0 - https://creativecommons.org/publicdomain/zero/1.0/
# ============================================================
set -Eeuo pipefail
IFS=$'\n\t'

PROGRAM_NAME="autodub-local"
PROGRAM_VERSION="2.1"
PROGRAM_AUTHOR="Francesco Galgani"
PROGRAM_REPOSITORY="https://github.com/jsfan3/autodub-local"
PROGRAM_LICENSE="CC0 - https://creativecommons.org/publicdomain/zero/1.0/"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/.autodub_local"
VENV_DIR="${WORK_DIR}/venv"
LOG_DIR="${WORK_DIR}/logs"
TMP_DIR="${WORK_DIR}/tmp"
MODELS_DIR="${WORK_DIR}/models"
HF_HOME_DIR="${MODELS_DIR}/hf"
TTS_PREFIX_DIR="${MODELS_DIR}"
TTS_CACHE_DIR="${MODELS_DIR}/tts"
XDG_DATA_DIR="${WORK_DIR}/xdg_data"
PY_SCRIPT="${WORK_DIR}/dub_worker.py"

info() { echo "[$(date +'%F %T')] [INFO] $*"; }
warn() { echo "[$(date +'%F %T')] [WARN] $*"; }
die() { echo "[ERROR] $*" >&2; exit 2; }

print_banner() {
  cat <<EOF
============================================================
${PROGRAM_NAME} ${PROGRAM_VERSION}
Author: ${PROGRAM_AUTHOR}
Repository: ${PROGRAM_REPOSITORY}
License: ${PROGRAM_LICENSE}
============================================================
EOF
}

usage() {
  cat <<EOF
${PROGRAM_NAME} ${PROGRAM_VERSION}

Video dubbing with local/cloud ASR and diarization, sentence-wise NLLB/Google Translate,
XTTS v2, Kokoro, Microsoft Edge TTS, and optional Ollama/Groq LLM text adaptation.

Usage:
  ./autodub_local.sh clean
  ./autodub_local.sh --input FILE --source-lang CODE --target-lang CODE --translation-method METHOD --tts-engine ENGINE [options]
  ./autodub_local.sh --input FILE --source-lang CODE --target-lang CODE --only-cloud [options]
  ./autodub_local.sh --target-lang CODE --tts-engine ENGINE --list-tts-voices [options]
  ./autodub_local.sh --target-lang CODE --tts-engine ENGINE --sample-tts-voices [options]

Required:
  -i, --input FILE              Input video/audio file. Any ffmpeg-readable container is accepted.
      --source-lang CODE        Source language for Whisper and translation, or "auto".
      --target-lang CODE        Target language code for translation and TTS.
      --translation-method M    local or google. Required unless --only-cloud is used.
      --tts-engine ENGINE       xtts, kokoro, or edge. Required unless --only-cloud is used.

Common options:
  -o, --output FILE             Output MP4 path. Default: <input_stem>_<TARGET_LANG>_<TTS_ENGINE>_dub.mp4
      --only-cloud              Use cloud ASR+diarization, cloud LLM adaptation, Edge TTS, and CPU-only local orchestration.
      clean                     Exclusive maintenance command. Lists and optionally deletes per-input temporary work folders.
      --num-speakers N          Expected number of speakers for diarization, or auto. Default: auto
      --min-speakers N          Minimum speakers when --num-speakers auto.
      --max-speakers N          Maximum speakers when --num-speakers auto.
      --whisper-model NAME      faster-whisper model. Default: medium
      --review-translation      Pause after creating the translation JSON when running interactively.
      --stop-after-translation  Stop after writing the translation JSON.
      --no-gpu                  Force CPU mode for local ML stages and request CPU-only Ollama inference.

LLM adaptation:
      --llm-adapt MODE          auto, always, or never. Default: auto
      --llm-segment MODE        auto, always, or never. Default: auto
      --llm-provider PROVIDER   ollama or groq. Default: ollama
      --llm-model NAME          Ollama model. Default: qwen3:8b-q4_K_M
      --groq-llm-model NAME     Groq chat model. Default: openai/gpt-oss-120b
      --llm-chars-per-second N  Override the per-speaker character budget used for overlong lines.
      --llm-max-retries N       Retry count for over-budget LLM output. Default: 3
      --llm-temperature N       Default: 0.1
      --llm-timeout SECONDS     Ollama request timeout. 0 disables timeout. Default: 0
      --llm-num-predict N       Maximum generated tokens per request. GPT OSS uses at least 1024. Default: 256
      --skip-ollama-install     Fail if Ollama/model are missing instead of installing/pulling.

TTS/audio options:
      --tts-locale LOCALE       TTS locale for Edge, e.g. it-IT. Derived from --target-lang when possible.
      --tts-voice-map MAP       Speaker-to-voice map, e.g. SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
      --tts-voice-map-strict    Error if detected speakers are not exactly covered by --tts-voice-map.
      --tts-voice-female V      Default female voice for target language/engine.
      --tts-voice-male V        Default male voice for target language/engine.
      --tts-voice-child V       Default child voice for target language/engine.
      --tts-speed N             Base TTS speed before per-speaker pacing. Default: 1.0
      --tts-max-chars N         TTS cache chunk limit for non-XTTS engines. Default: 5000
      --list-tts-voices         List voices for --tts-engine and --target-lang, then exit.
      --sample-tts-voices       Generate sample voice audio files, then exit.
      --sample-text TEXT        Text for --sample-tts-voices.
      --sample-output-dir DIR   Output directory for samples. Default: .autodub_local/samples/<engine>_<locale>
      --edge-pitch VALUE        Edge TTS pitch. Default: +0Hz
      --edge-volume VALUE       Edge TTS volume. Default: +0%
      --edge-connect-timeout N  Edge TTS connect timeout in seconds. Default: 20
      --edge-receive-timeout N  Edge TTS receive timeout in seconds. Default: 120
      --edge-max-retries N      Edge TTS retries after a transient failure. Default: 3
      --edge-retry-delay N      Initial Edge TTS retry delay in seconds. Default: 5
      --max-ref-clips N
      --xtts-max-chars N
      --xtts-speed N            Default: 1.0
      --xtts-temperature N      Default: 0.65
      --xtts-repetition-penalty N
      --xtts-inter-chunk-silence-ms N
      --max-tts-compress-ratio N  Default: 1.15
      --max-tts-expand-ratio N    Default: 1.20
      --aac-bitrate RATE

Advanced ASR/segmentation options:
      --asr-backend BACKEND     local, groq, or assemblyai. Default: local
      --diarization-backend B   local or assemblyai. Default: local
      --groq-whisper-model NAME Default: whisper-large-v3
      --groq-prompt TEXT        Optional Groq Whisper prompt/context.
      --groq-chunk-seconds N    Default: 120
      --groq-overlap-seconds N  Default: 1.0
      --groq-timeout SECONDS    Default: 300
      --groq-max-retries N      Default: 5
      --groq-rate-limit MODE    wait or fail. Default: wait
      --assemblyai-speech-model NAME  Model or comma-list. Default: universal-3-5-pro,universal-2
      --assemblyai-poll-interval SEC  Default: 5
      --assemblyai-timeout SEC        Default: 7200
      --asr-beam N
      --asr-vad true|false
      --asr-compute-gpu TYPE
      --asr-compute-cpu TYPE
      --translate-batch N
      --translate-on-gpu 0|1
      --utterance-max-gap SEC
      --utterance-max-duration SEC
      --utterance-max-chars N
      --utterance-repair-max-gap SEC
      --utterance-repair-max-duration SEC
      --utterance-repair-max-chars N
      --nllb-src-lang CODE      Optional NLLB override. Normally derived from --source-lang.
      --nllb-tgt-lang CODE      Optional NLLB override. Normally derived from --target-lang.
      --log-level LEVEL

Examples:
  ./autodub_local.sh --input test.mp4 --source-lang en --target-lang it --translation-method google --tts-engine edge --tts-voice-map SPEAKER_00=it-IT-GiuseppeMultilingualNeural,SPEAKER_01=it-IT-DiegoNeural
  ./autodub_local.sh -i talk.webm --source-lang auto --target-lang fr --translation-method local --tts-engine xtts
  ./autodub_local.sh -i interview.mkv --source-lang en --target-lang es --translation-method local --tts-engine kokoro
  ./autodub_local.sh --target-lang it --tts-engine edge --list-tts-voices
  ./autodub_local.sh --target-lang it --tts-engine edge --sample-tts-voices --sample-text "Questo è un test con free software e Georgia Tech."

Supported XTTS target languages:
  ar, cs, de, en, es, fr, hi, hu, it, ja, ko, nl, pl, pt, ru, tr, zh

Supported Kokoro target languages:
  en, en-us, en-gb, es, fr, hi, it, ja, pt, zh
EOF
}

INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
SOURCE_LANG="${SOURCE_LANG:-}"
TARGET_LANG="${TARGET_LANG:-}"
TRANSLATION_METHOD="${TRANSLATION_METHOD:-}"
TTS_ENGINE="${TTS_ENGINE:-}"
ONLY_CLOUD="${ONLY_CLOUD:-0}"
NLLB_SRC_LANG="${NLLB_SRC_LANG:-auto}"
NLLB_TGT_LANG="${NLLB_TGT_LANG:-auto}"
ASR_BACKEND="${ASR_BACKEND:-local}"
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-local}"
WHISPER_MODEL="${WHISPER_MODEL:-medium}"
GROQ_WHISPER_MODEL="${GROQ_WHISPER_MODEL:-whisper-large-v3}"
GROQ_PROMPT="${GROQ_PROMPT:-}"
GROQ_CHUNK_SECONDS="${GROQ_CHUNK_SECONDS:-120}"
GROQ_OVERLAP_SECONDS="${GROQ_OVERLAP_SECONDS:-1.0}"
GROQ_TIMEOUT="${GROQ_TIMEOUT:-300}"
GROQ_MAX_RETRIES="${GROQ_MAX_RETRIES:-5}"
GROQ_RATE_LIMIT="${GROQ_RATE_LIMIT:-wait}"
ASSEMBLYAI_SPEECH_MODEL="${ASSEMBLYAI_SPEECH_MODEL:-universal-3-5-pro,universal-2}"
ASSEMBLYAI_POLL_INTERVAL="${ASSEMBLYAI_POLL_INTERVAL:-5}"
ASSEMBLYAI_TIMEOUT="${ASSEMBLYAI_TIMEOUT:-7200}"
NUM_SPEAKERS="${NUM_SPEAKERS:-auto}"
MIN_SPEAKERS="${MIN_SPEAKERS:-}"
MAX_SPEAKERS="${MAX_SPEAKERS:-}"
ASR_BEAM="${ASR_BEAM:-5}"
ASR_VAD="${ASR_VAD:-true}"
ASR_COMPUTE_GPU="${ASR_COMPUTE_GPU:-int8_float16}"
ASR_COMPUTE_CPU="${ASR_COMPUTE_CPU:-int8}"
MAX_REF_CLIPS="${MAX_REF_CLIPS:-3}"
XTTS_MAX_CHARS="${XTTS_MAX_CHARS:-180}"
XTTS_CHAR_LIMIT_MARGIN="${XTTS_CHAR_LIMIT_MARGIN:-20}"
XTTS_SPEED="${XTTS_SPEED:-1.0}"
XTTS_TEMPERATURE="${XTTS_TEMPERATURE:-0.65}"
XTTS_REPETITION_PENALTY="${XTTS_REPETITION_PENALTY:-2.0}"
XTTS_INTER_CHUNK_SILENCE_MS="${XTTS_INTER_CHUNK_SILENCE_MS:-120}"
MAX_TTS_COMPRESS_RATIO="${MAX_TTS_COMPRESS_RATIO:-1.15}"
MAX_TTS_EXPAND_RATIO="${MAX_TTS_EXPAND_RATIO:-1.20}"
AAC_BITRATE="${AAC_BITRATE:-192k}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TRANSLATE_BATCH="${TRANSLATE_BATCH:-12}"
TRANSLATE_ON_GPU="${TRANSLATE_ON_GPU:-0}"
NO_GPU="${NO_GPU:-0}"
UTTERANCE_MAX_GAP="${UTTERANCE_MAX_GAP:-0.9}"
UTTERANCE_MAX_DURATION="${UTTERANCE_MAX_DURATION:-18.0}"
UTTERANCE_MAX_CHARS="${UTTERANCE_MAX_CHARS:-420}"
UTTERANCE_REPAIR_MAX_GAP="${UTTERANCE_REPAIR_MAX_GAP:-2.2}"
UTTERANCE_REPAIR_MAX_DURATION="${UTTERANCE_REPAIR_MAX_DURATION:-24.0}"
UTTERANCE_REPAIR_MAX_CHARS="${UTTERANCE_REPAIR_MAX_CHARS:-620}"
REVIEW_TRANSLATION="${REVIEW_TRANSLATION:-0}"
STOP_AFTER_TRANSLATION="${STOP_AFTER_TRANSLATION:-0}"
LLM_ADAPT="${LLM_ADAPT:-auto}"
LLM_SEGMENT="${LLM_SEGMENT:-auto}"
LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
LLM_MODEL="${LLM_MODEL:-qwen3:8b-q4_K_M}"
GROQ_LLM_MODEL="${GROQ_LLM_MODEL:-openai/gpt-oss-120b}"
LLM_CHARS_PER_SECOND="${LLM_CHARS_PER_SECOND:-}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-3}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"
LLM_TIMEOUT="${LLM_TIMEOUT:-0}"
LLM_NUM_PREDICT="${LLM_NUM_PREDICT:-256}"
OLLAMA_INSTALL="${OLLAMA_INSTALL:-auto}"
TTS_LOCALE="${TTS_LOCALE:-}"
TTS_VOICE_MAP="${TTS_VOICE_MAP:-}"
TTS_VOICE_MAP_STRICT="${TTS_VOICE_MAP_STRICT:-0}"
TTS_VOICE_FEMALE="${TTS_VOICE_FEMALE:-}"
TTS_VOICE_MALE="${TTS_VOICE_MALE:-}"
TTS_VOICE_CHILD="${TTS_VOICE_CHILD:-}"
TTS_SPEED="${TTS_SPEED:-1.0}"
TTS_MAX_CHARS="${TTS_MAX_CHARS:-5000}"
LIST_TTS_VOICES="${LIST_TTS_VOICES:-0}"
SAMPLE_TTS_VOICES="${SAMPLE_TTS_VOICES:-0}"
SAMPLE_TEXT="${SAMPLE_TEXT:-}"
SAMPLE_OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-}"
EDGE_TTS_PITCH="${EDGE_TTS_PITCH:-+0Hz}"
EDGE_TTS_VOLUME="${EDGE_TTS_VOLUME:-+0%}"
EDGE_TTS_CONNECT_TIMEOUT="${EDGE_TTS_CONNECT_TIMEOUT:-20}"
EDGE_TTS_RECEIVE_TIMEOUT="${EDGE_TTS_RECEIVE_TIMEOUT:-120}"
EDGE_TTS_MAX_RETRIES="${EDGE_TTS_MAX_RETRIES:-3}"
EDGE_TTS_RETRY_DELAY="${EDGE_TTS_RETRY_DELAY:-5}"

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      -i|--input)
        [[ $# -ge 2 ]] || die "--input requires a value"
        INPUT_FILE="$2"
        shift 2
        ;;
      -o|--output)
        [[ $# -ge 2 ]] || die "--output requires a value"
        OUTPUT_FILE="$2"
        shift 2
        ;;
      --source-lang)
        [[ $# -ge 2 ]] || die "--source-lang requires a value"
        SOURCE_LANG="$2"
        shift 2
        ;;
      --target-lang)
        [[ $# -ge 2 ]] || die "--target-lang requires a value"
        TARGET_LANG="$2"
        shift 2
        ;;
      --translation-method)
        [[ $# -ge 2 ]] || die "--translation-method requires a value"
        TRANSLATION_METHOD="$2"
        shift 2
        ;;
      --tts-engine)
        [[ $# -ge 2 ]] || die "--tts-engine requires a value"
        TTS_ENGINE="$2"
        shift 2
        ;;
      --only-cloud)
        ONLY_CLOUD="1"
        shift
        ;;
      --num-speakers)
        [[ $# -ge 2 ]] || die "--num-speakers requires a value"
        NUM_SPEAKERS="$2"
        shift 2
        ;;
      --min-speakers)
        [[ $# -ge 2 ]] || die "--min-speakers requires a value"
        MIN_SPEAKERS="$2"
        shift 2
        ;;
      --max-speakers)
        [[ $# -ge 2 ]] || die "--max-speakers requires a value"
        MAX_SPEAKERS="$2"
        shift 2
        ;;
      --whisper-model)
        [[ $# -ge 2 ]] || die "--whisper-model requires a value"
        WHISPER_MODEL="$2"
        shift 2
        ;;
      --review-translation)
        REVIEW_TRANSLATION="1"
        shift
        ;;
      --stop-after-translation)
        STOP_AFTER_TRANSLATION="1"
        shift
        ;;
      --no-gpu)
        NO_GPU="1"
        shift
        ;;
      --llm-adapt)
        [[ $# -ge 2 ]] || die "--llm-adapt requires a value"
        LLM_ADAPT="$2"
        shift 2
        ;;
      --llm-segment)
        [[ $# -ge 2 ]] || die "--llm-segment requires a value"
        LLM_SEGMENT="$2"
        shift 2
        ;;
      --llm-provider)
        [[ $# -ge 2 ]] || die "--llm-provider requires a value"
        LLM_PROVIDER="$2"
        shift 2
        ;;
      --llm-model)
        [[ $# -ge 2 ]] || die "--llm-model requires a value"
        LLM_MODEL="$2"
        shift 2
        ;;
      --groq-llm-model)
        [[ $# -ge 2 ]] || die "--groq-llm-model requires a value"
        GROQ_LLM_MODEL="$2"
        shift 2
        ;;
      --llm-chars-per-second)
        [[ $# -ge 2 ]] || die "--llm-chars-per-second requires a value"
        LLM_CHARS_PER_SECOND="$2"
        shift 2
        ;;
      --llm-max-retries)
        [[ $# -ge 2 ]] || die "--llm-max-retries requires a value"
        LLM_MAX_RETRIES="$2"
        shift 2
        ;;
      --llm-temperature)
        [[ $# -ge 2 ]] || die "--llm-temperature requires a value"
        LLM_TEMPERATURE="$2"
        shift 2
        ;;
      --llm-timeout)
        [[ $# -ge 2 ]] || die "--llm-timeout requires a value"
        LLM_TIMEOUT="$2"
        shift 2
        ;;
      --llm-num-predict)
        [[ $# -ge 2 ]] || die "--llm-num-predict requires a value"
        LLM_NUM_PREDICT="$2"
        shift 2
        ;;
      --skip-ollama-install)
        OLLAMA_INSTALL="never"
        shift
        ;;
      --tts-locale)
        [[ $# -ge 2 ]] || die "--tts-locale requires a value"
        TTS_LOCALE="$2"
        shift 2
        ;;
      --tts-voice-map)
        [[ $# -ge 2 ]] || die "--tts-voice-map requires a value"
        TTS_VOICE_MAP="$2"
        shift 2
        ;;
      --tts-voice-map-strict)
        TTS_VOICE_MAP_STRICT="1"
        shift
        ;;
      --tts-voice-female)
        [[ $# -ge 2 ]] || die "--tts-voice-female requires a value"
        TTS_VOICE_FEMALE="$2"
        shift 2
        ;;
      --tts-voice-male)
        [[ $# -ge 2 ]] || die "--tts-voice-male requires a value"
        TTS_VOICE_MALE="$2"
        shift 2
        ;;
      --tts-voice-child)
        [[ $# -ge 2 ]] || die "--tts-voice-child requires a value"
        TTS_VOICE_CHILD="$2"
        shift 2
        ;;
      --tts-speed)
        [[ $# -ge 2 ]] || die "--tts-speed requires a value"
        TTS_SPEED="$2"
        shift 2
        ;;
      --tts-max-chars)
        [[ $# -ge 2 ]] || die "--tts-max-chars requires a value"
        TTS_MAX_CHARS="$2"
        shift 2
        ;;
      --list-tts-voices)
        LIST_TTS_VOICES="1"
        shift
        ;;
      --sample-tts-voices)
        SAMPLE_TTS_VOICES="1"
        shift
        ;;
      --sample-text)
        [[ $# -ge 2 ]] || die "--sample-text requires a value"
        SAMPLE_TEXT="$2"
        shift 2
        ;;
      --sample-output-dir)
        [[ $# -ge 2 ]] || die "--sample-output-dir requires a value"
        SAMPLE_OUTPUT_DIR="$2"
        shift 2
        ;;
      --edge-pitch)
        [[ $# -ge 2 ]] || die "--edge-pitch requires a value"
        EDGE_TTS_PITCH="$2"
        shift 2
        ;;
      --edge-volume)
        [[ $# -ge 2 ]] || die "--edge-volume requires a value"
        EDGE_TTS_VOLUME="$2"
        shift 2
        ;;
      --edge-connect-timeout)
        [[ $# -ge 2 ]] || die "--edge-connect-timeout requires a value"
        EDGE_TTS_CONNECT_TIMEOUT="$2"
        shift 2
        ;;
      --edge-receive-timeout)
        [[ $# -ge 2 ]] || die "--edge-receive-timeout requires a value"
        EDGE_TTS_RECEIVE_TIMEOUT="$2"
        shift 2
        ;;
      --edge-max-retries)
        [[ $# -ge 2 ]] || die "--edge-max-retries requires a value"
        EDGE_TTS_MAX_RETRIES="$2"
        shift 2
        ;;
      --edge-retry-delay)
        [[ $# -ge 2 ]] || die "--edge-retry-delay requires a value"
        EDGE_TTS_RETRY_DELAY="$2"
        shift 2
        ;;
      --max-ref-clips)
        [[ $# -ge 2 ]] || die "--max-ref-clips requires a value"
        MAX_REF_CLIPS="$2"
        shift 2
        ;;
      --xtts-max-chars)
        [[ $# -ge 2 ]] || die "--xtts-max-chars requires a value"
        XTTS_MAX_CHARS="$2"
        shift 2
        ;;
      --xtts-speed)
        [[ $# -ge 2 ]] || die "--xtts-speed requires a value"
        XTTS_SPEED="$2"
        shift 2
        ;;
      --xtts-temperature)
        [[ $# -ge 2 ]] || die "--xtts-temperature requires a value"
        XTTS_TEMPERATURE="$2"
        shift 2
        ;;
      --xtts-repetition-penalty)
        [[ $# -ge 2 ]] || die "--xtts-repetition-penalty requires a value"
        XTTS_REPETITION_PENALTY="$2"
        shift 2
        ;;
      --xtts-inter-chunk-silence-ms)
        [[ $# -ge 2 ]] || die "--xtts-inter-chunk-silence-ms requires a value"
        XTTS_INTER_CHUNK_SILENCE_MS="$2"
        shift 2
        ;;
      --max-tts-compress-ratio)
        [[ $# -ge 2 ]] || die "--max-tts-compress-ratio requires a value"
        MAX_TTS_COMPRESS_RATIO="$2"
        shift 2
        ;;
      --max-tts-expand-ratio)
        [[ $# -ge 2 ]] || die "--max-tts-expand-ratio requires a value"
        MAX_TTS_EXPAND_RATIO="$2"
        shift 2
        ;;
      --aac-bitrate)
        [[ $# -ge 2 ]] || die "--aac-bitrate requires a value"
        AAC_BITRATE="$2"
        shift 2
        ;;
      --asr-backend)
        [[ $# -ge 2 ]] || die "--asr-backend requires a value"
        ASR_BACKEND="$2"
        shift 2
        ;;
      --diarization-backend)
        [[ $# -ge 2 ]] || die "--diarization-backend requires a value"
        DIARIZATION_BACKEND="$2"
        shift 2
        ;;
      --groq-whisper-model)
        [[ $# -ge 2 ]] || die "--groq-whisper-model requires a value"
        GROQ_WHISPER_MODEL="$2"
        shift 2
        ;;
      --groq-prompt)
        [[ $# -ge 2 ]] || die "--groq-prompt requires a value"
        GROQ_PROMPT="$2"
        shift 2
        ;;
      --groq-chunk-seconds)
        [[ $# -ge 2 ]] || die "--groq-chunk-seconds requires a value"
        GROQ_CHUNK_SECONDS="$2"
        shift 2
        ;;
      --groq-overlap-seconds)
        [[ $# -ge 2 ]] || die "--groq-overlap-seconds requires a value"
        GROQ_OVERLAP_SECONDS="$2"
        shift 2
        ;;
      --groq-timeout)
        [[ $# -ge 2 ]] || die "--groq-timeout requires a value"
        GROQ_TIMEOUT="$2"
        shift 2
        ;;
      --groq-max-retries)
        [[ $# -ge 2 ]] || die "--groq-max-retries requires a value"
        GROQ_MAX_RETRIES="$2"
        shift 2
        ;;
      --groq-rate-limit)
        [[ $# -ge 2 ]] || die "--groq-rate-limit requires a value"
        GROQ_RATE_LIMIT="$2"
        shift 2
        ;;
      --assemblyai-speech-model)
        [[ $# -ge 2 ]] || die "--assemblyai-speech-model requires a value"
        ASSEMBLYAI_SPEECH_MODEL="$2"
        shift 2
        ;;
      --assemblyai-poll-interval)
        [[ $# -ge 2 ]] || die "--assemblyai-poll-interval requires a value"
        ASSEMBLYAI_POLL_INTERVAL="$2"
        shift 2
        ;;
      --assemblyai-timeout)
        [[ $# -ge 2 ]] || die "--assemblyai-timeout requires a value"
        ASSEMBLYAI_TIMEOUT="$2"
        shift 2
        ;;
      --asr-beam)
        [[ $# -ge 2 ]] || die "--asr-beam requires a value"
        ASR_BEAM="$2"
        shift 2
        ;;
      --asr-vad)
        [[ $# -ge 2 ]] || die "--asr-vad requires a value"
        ASR_VAD="$2"
        shift 2
        ;;
      --asr-compute-gpu)
        [[ $# -ge 2 ]] || die "--asr-compute-gpu requires a value"
        ASR_COMPUTE_GPU="$2"
        shift 2
        ;;
      --asr-compute-cpu)
        [[ $# -ge 2 ]] || die "--asr-compute-cpu requires a value"
        ASR_COMPUTE_CPU="$2"
        shift 2
        ;;
      --translate-batch)
        [[ $# -ge 2 ]] || die "--translate-batch requires a value"
        TRANSLATE_BATCH="$2"
        shift 2
        ;;
      --translate-on-gpu)
        [[ $# -ge 2 ]] || die "--translate-on-gpu requires a value"
        TRANSLATE_ON_GPU="$2"
        shift 2
        ;;
      --utterance-max-gap)
        [[ $# -ge 2 ]] || die "--utterance-max-gap requires a value"
        UTTERANCE_MAX_GAP="$2"
        shift 2
        ;;
      --utterance-max-duration)
        [[ $# -ge 2 ]] || die "--utterance-max-duration requires a value"
        UTTERANCE_MAX_DURATION="$2"
        shift 2
        ;;
      --utterance-max-chars)
        [[ $# -ge 2 ]] || die "--utterance-max-chars requires a value"
        UTTERANCE_MAX_CHARS="$2"
        shift 2
        ;;
      --utterance-repair-max-gap)
        [[ $# -ge 2 ]] || die "--utterance-repair-max-gap requires a value"
        UTTERANCE_REPAIR_MAX_GAP="$2"
        shift 2
        ;;
      --utterance-repair-max-duration)
        [[ $# -ge 2 ]] || die "--utterance-repair-max-duration requires a value"
        UTTERANCE_REPAIR_MAX_DURATION="$2"
        shift 2
        ;;
      --utterance-repair-max-chars)
        [[ $# -ge 2 ]] || die "--utterance-repair-max-chars requires a value"
        UTTERANCE_REPAIR_MAX_CHARS="$2"
        shift 2
        ;;
      --nllb-src-lang)
        [[ $# -ge 2 ]] || die "--nllb-src-lang requires a value"
        NLLB_SRC_LANG="$2"
        shift 2
        ;;
      --nllb-tgt-lang)
        [[ $# -ge 2 ]] || die "--nllb-tgt-lang requires a value"
        NLLB_TGT_LANG="$2"
        shift 2
        ;;
      --log-level)
        [[ $# -ge 2 ]] || die "--log-level requires a value"
        LOG_LEVEL="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      -*)
        die "Unknown option: $1"
        ;;
      *)
        if [[ -z "$INPUT_FILE" ]]; then
          INPUT_FILE="$1"
          shift
        else
          die "Unexpected positional argument: $1"
        fi
        ;;
    esac
  done
  [[ $# -eq 0 ]] || die "Unexpected positional arguments: $*"
}

is_reserved_work_dir() {
  local name
  name="$(basename "$1")"
  case "$name" in
    __pycache__|cache|logs|models|tmp|venv|xdg_data)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

dub_work_status() {
  local dir="$1"
  if find "$dir" -maxdepth 1 -type f -name '*.manifest.json' -print -quit | grep -q .; then
    printf 'completed'
  elif find "$dir" -maxdepth 1 -type f -name '*.translated.*.json' -print -quit | grep -q .; then
    printf 'translated'
  elif find "$dir" -maxdepth 1 -type f -name '*.utterances.json' -print -quit | grep -q .; then
    printf 'utterances'
  elif find "$dir" -maxdepth 1 -type f -name '*.transcript.json' -print -quit | grep -q .; then
    printf 'transcribed'
  else
    printf 'partial'
  fi
}

clean_mode() {
  print_banner
  echo "Temporary dubbing work folders under:"
  echo "  ${WORK_DIR}"
  echo

  if [[ ! -d "$WORK_DIR" ]]; then
    echo "No work directory exists yet."
    return 0
  fi

  local dirs=()
  local dir
  shopt -s nullglob
  for dir in "$WORK_DIR"/*; do
    [[ -d "$dir" ]] || continue
    if is_reserved_work_dir "$dir"; then
      continue
    fi
    dirs+=("$dir")
  done
  shopt -u nullglob

  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No per-input temporary dubbing folders were found."
    return 0
  fi

  local i size mtime status
  for i in "${!dirs[@]}"; do
    dir="${dirs[$i]}"
    size="$(du -sh -- "$dir" 2>/dev/null | awk '{print $1}')"
    mtime="$(stat -c '%y' -- "$dir" 2>/dev/null | cut -d'.' -f1)"
    status="$(dub_work_status "$dir")"
    printf '%3d) %-28s status=%-11s size=%8s modified=%s\n' "$((i + 1))" "$(basename "$dir")" "$status" "${size:-?}" "${mtime:-?}"
  done

  echo
  if [[ ! -t 0 ]]; then
    echo "Run this command in an interactive terminal to delete selected folders."
    return 0
  fi

  local choice
  read -r -p "Delete which folders? Enter 'all', comma-separated numbers, or press Enter to cancel: " choice
  choice="${choice//[[:space:]]/}"
  if [[ -z "$choice" ]]; then
    echo "Cancelled."
    return 0
  fi

  local selected=()
  if [[ "${choice,,}" == "all" ]]; then
    selected=("${dirs[@]}")
  else
    local part idx
    local parts=()
    IFS=',' read -r -a parts <<< "$choice"
    for part in "${parts[@]}"; do
      [[ "$part" =~ ^[0-9]+$ ]] || die "Invalid clean selection: ${part}"
      idx=$((part - 1))
      [[ "$idx" -ge 0 && "$idx" -lt "${#dirs[@]}" ]] || die "Clean selection out of range: ${part}"
      selected+=("${dirs[$idx]}")
    done
  fi

  echo
  echo "Selected folders:"
  for dir in "${selected[@]}"; do
    echo "  ${dir}"
  done
  echo

  local confirm resolved
  read -r -p "Type DELETE to permanently remove them: " confirm
  if [[ "$confirm" != "DELETE" ]]; then
    echo "Cancelled."
    return 0
  fi

  for dir in "${selected[@]}"; do
    resolved="$(realpath -m -- "$dir")"
    case "$resolved" in
      "$WORK_DIR"/*)
        if is_reserved_work_dir "$resolved"; then
          die "Refusing to delete reserved work directory: ${resolved}"
        fi
        rm -rf -- "$resolved"
        echo "Deleted: ${resolved}"
        ;;
      *)
        die "Refusing to delete path outside ${WORK_DIR}: ${resolved}"
        ;;
    esac
  done
}

if [[ $# -gt 0 ]]; then
  for raw_arg in "$@"; do
    if [[ "$raw_arg" == "clean" || "$raw_arg" == "--clean" ]]; then
      if [[ $# -ne 1 || ( "$1" != "clean" && "$1" != "--clean" ) ]]; then
        die "clean must be used alone: ./autodub_local.sh clean"
      fi
      clean_mode
      exit 0
    fi
  done
fi

parse_args "$@"

SOURCE_LANG="${SOURCE_LANG,,}"
TARGET_LANG="${TARGET_LANG,,}"
TRANSLATION_METHOD="${TRANSLATION_METHOD,,}"
TTS_ENGINE="${TTS_ENGINE,,}"
LLM_ADAPT="${LLM_ADAPT,,}"
LLM_SEGMENT="${LLM_SEGMENT,,}"
LLM_PROVIDER="${LLM_PROVIDER,,}"
ASR_BACKEND="${ASR_BACKEND,,}"
DIARIZATION_BACKEND="${DIARIZATION_BACKEND,,}"
GROQ_RATE_LIMIT="${GROQ_RATE_LIMIT,,}"
ASR_VAD="${ASR_VAD,,}"
NO_GPU="${NO_GPU,,}"
ONLY_CLOUD="${ONLY_CLOUD,,}"

case "$ONLY_CLOUD" in
  0|false|no) ONLY_CLOUD="0" ;;
  1|true|yes)
    ONLY_CLOUD="1"
    ASR_BACKEND="assemblyai"
    DIARIZATION_BACKEND="assemblyai"
    TRANSLATION_METHOD="google"
    TTS_ENGINE="edge"
    LLM_PROVIDER="groq"
    NO_GPU="1"
    ;;
  *) die "--only-cloud/ONLY_CLOUD must be 0 or 1" ;;
esac

[[ -n "$TARGET_LANG" ]] || die "Missing required --target-lang CODE"
[[ -n "$TTS_ENGINE" ]] || die "Missing required --tts-engine xtts|kokoro|edge"

TTS_CATALOG_ACTION=0
if [[ "$LIST_TTS_VOICES" == "1" || "$SAMPLE_TTS_VOICES" == "1" ]]; then
  TTS_CATALOG_ACTION=1
fi

if [[ "$TTS_CATALOG_ACTION" -eq 0 ]]; then
  [[ -n "$INPUT_FILE" ]] || die "Missing required --input FILE"
  [[ -n "$SOURCE_LANG" ]] || die "Missing required --source-lang CODE"
  [[ -n "$TRANSLATION_METHOD" ]] || die "Missing required --translation-method local|google"
  [[ -f "$INPUT_FILE" ]] || die "Input file does not exist: $INPUT_FILE"
fi

if [[ -n "$TRANSLATION_METHOD" ]]; then
  case "$TRANSLATION_METHOD" in
    local|google) ;;
    *) die "--translation-method must be local or google" ;;
  esac
fi

case "$TTS_ENGINE" in
  xtts|kokoro|edge) ;;
  *) die "--tts-engine must be xtts, kokoro, or edge" ;;
esac

case "$LLM_ADAPT" in
  auto|always|never) ;;
  *) die "--llm-adapt must be auto, always, or never" ;;
esac

case "$LLM_SEGMENT" in
  auto|always|never) ;;
  *) die "--llm-segment must be auto, always, or never" ;;
esac

case "$LLM_PROVIDER" in
  ollama|groq) ;;
  *) die "--llm-provider must be ollama or groq" ;;
esac

case "$ASR_BACKEND" in
  local|groq|assemblyai) ;;
  *) die "--asr-backend must be local, groq, or assemblyai" ;;
esac

case "$DIARIZATION_BACKEND" in
  local|assemblyai) ;;
  *) die "--diarization-backend must be local or assemblyai" ;;
esac

case "$GROQ_RATE_LIMIT" in
  wait|fail) ;;
  *) die "--groq-rate-limit must be wait or fail" ;;
esac

if [[ "$DIARIZATION_BACKEND" == "assemblyai" && "$ASR_BACKEND" != "assemblyai" ]]; then
  die "--diarization-backend assemblyai requires --asr-backend assemblyai"
fi

if [[ "$ASR_BACKEND" == "assemblyai" && "$DIARIZATION_BACKEND" != "assemblyai" ]]; then
  die "--asr-backend assemblyai requires --diarization-backend assemblyai"
fi

case "$ASR_VAD" in
  true|false) ;;
  *) die "--asr-vad must be true or false" ;;
esac

case "$TRANSLATE_ON_GPU" in
  0|1) ;;
  *) die "--translate-on-gpu must be 0 or 1" ;;
esac

case "$NO_GPU" in
  0|false|no) NO_GPU="0" ;;
  1|true|yes) NO_GPU="1" ;;
  *) die "--no-gpu/NO_GPU must be 0 or 1" ;;
esac

if [[ "$NUM_SPEAKERS" != "auto" && ! "$NUM_SPEAKERS" =~ ^[0-9]+$ ]]; then
  die "--num-speakers must be auto or a positive integer"
fi
if [[ "$NUM_SPEAKERS" != "auto" && "$NUM_SPEAKERS" -lt 1 ]]; then
  die "--num-speakers must be auto or a positive integer"
fi
if [[ -n "$MIN_SPEAKERS" && ! "$MIN_SPEAKERS" =~ ^[0-9]+$ ]]; then
  die "--min-speakers must be a positive integer"
fi
if [[ -n "$MAX_SPEAKERS" && ! "$MAX_SPEAKERS" =~ ^[0-9]+$ ]]; then
  die "--max-speakers must be a positive integer"
fi
if [[ -n "$MIN_SPEAKERS" && "$MIN_SPEAKERS" -lt 1 ]]; then
  die "--min-speakers must be a positive integer"
fi
if [[ -n "$MAX_SPEAKERS" && "$MAX_SPEAKERS" -lt 1 ]]; then
  die "--max-speakers must be a positive integer"
fi
if [[ -n "$MIN_SPEAKERS" && -n "$MAX_SPEAKERS" && "$MIN_SPEAKERS" -gt "$MAX_SPEAKERS" ]]; then
  die "--min-speakers cannot be greater than --max-speakers"
fi

if [[ -n "$INPUT_FILE" ]]; then
  INPUT_FILE="$(readlink -f "$INPUT_FILE")"
fi
if [[ -n "$OUTPUT_FILE" ]]; then
  OUTPUT_FILE="$(realpath -m "$OUTPUT_FILE")"
fi
if [[ -n "$SAMPLE_OUTPUT_DIR" ]]; then
  SAMPLE_OUTPUT_DIR="$(realpath -m "$SAMPLE_OUTPUT_DIR")"
fi
export TRANSLATION_METHOD TTS_ENGINE

if [[ "$NO_GPU" == "1" ]]; then
  TRANSLATE_ON_GPU="0"
  export AUTODUB_NO_GPU="1"
  export CUDA_VISIBLE_DEVICES="-1"
  export NVIDIA_VISIBLE_DEVICES="none"
  export HIP_VISIBLE_DEVICES="-1"
  export ROCR_VISIBLE_DEVICES="-1"
  export GPU_DEVICE_ORDINAL=""
  export GGML_CUDA_VISIBLE_DEVICES="-1"
  export GGML_VK_VISIBLE_DEVICES="-1"
  export OLLAMA_VULKAN="0"
  export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cpu}"
fi

mkdir -p "$WORK_DIR" "$LOG_DIR" "$TMP_DIR" "$MODELS_DIR" "$HF_HOME_DIR" "$TTS_PREFIX_DIR" "$TTS_CACHE_DIR" "$XDG_DATA_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local ec=$?
  echo
  echo "[ERROR] The script stopped with exit code ${ec}."
  echo "[ERROR] Full log: ${LOG_FILE}"
  echo "[ERROR] Review the log above for the failing step and stack trace."
  exit "$ec"
}
trap on_error ERR

auto_apt_install() {
  local missing=()
  local packages=()
  for bin in "$@"; do
    command -v "$bin" >/dev/null 2>&1 || missing+=("$bin")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "Required commands are missing: ${missing[*]}"
    warn "Install the required system packages and run the script again."
    return 1
  fi
  info "Missing system dependencies: ${missing[*]}"
  info "Attempting installation with apt-get. sudo may prompt for a password..."
  local bin
  for bin in "${missing[@]}"; do
    case "$bin" in
      ffmpeg) packages+=("ffmpeg") ;;
      python3) packages+=("python3" "python3-venv" "python3-pip") ;;
      curl) packages+=("curl") ;;
      espeak-ng) packages+=("espeak-ng") ;;
      git-lfs) packages+=("git-lfs") ;;
      *) packages+=("$bin") ;;
    esac
  done
  sudo apt-get update
  sudo apt-get install -y "${packages[@]}"
}

ensure_python_venv_support() {
  python3 -m venv --help >/dev/null 2>&1 && return 0
  if ! command -v apt-get >/dev/null 2>&1; then
    die "python3 venv support is missing. Install python3-venv and python3-pip, then run the script again."
  fi
  info "python3 venv support is missing. Installing python3-venv and python3-pip..."
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip
}

python_imports_ok() {
  python - <<'PY' >/dev/null 2>&1
import os
mods = [
    'soundfile', 'numpy', 'librosa', 'requests'
]
if os.environ.get("ASR_BACKEND", "local") == "local":
    mods += ['torch', 'torchaudio', 'faster_whisper']
if os.environ.get("DIARIZATION_BACKEND", "local") == "local":
    mods += ['torch', 'torchaudio', 'pyannote.audio', 'huggingface_hub']
if os.environ.get("TRANSLATION_METHOD") == "local":
    mods += ['torch', 'transformers', 'sentencepiece', 'accelerate']
if os.environ.get("TRANSLATION_METHOD") == "google":
    mods += ['deep_translator']
if os.environ.get("TTS_ENGINE") == "xtts":
    mods += ['torch', 'TTS']
if os.environ.get("TTS_ENGINE") == "kokoro":
    mods += ['torch', 'kokoro']
    target = os.environ.get("TARGET_LANG", "").strip().lower()
    if target == "ja":
        mods += ['misaki.ja']
    elif target == "zh":
        mods += ['misaki.zh']
if os.environ.get("TTS_ENGINE") == "edge":
    mods += ['edge_tts']
for m in mods:
    __import__(m)
print('ok')
PY
}

ollama_api_ok() {
  command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
}

load_cloud_keys() {
  local file="${SCRIPT_DIR}/.cloud_keys"
  [[ -f "$file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      GROQ_API_KEY|ASSEMBLYAI_API_KEY|DEEPGRAM_API_KEY|GEMINI_API_KEY)
        if [[ -z "${!key:-}" && -n "$value" ]]; then
          export "$key=$value"
        fi
        ;;
    esac
  done < "$file"
}

save_cloud_key() {
  local key="$1"
  local value="$2"
  local file="${SCRIPT_DIR}/.cloud_keys"
  local tmp="${file}.tmp"
  umask 077
  touch "$file"
  chmod 600 "$file"
  {
    grep -v -E "^${key}=" "$file" 2>/dev/null || true
    printf '%s=%s\n' "$key" "$value"
  } > "$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$file"
}

ensure_cloud_key() {
  local key="$1"
  local label="$2"
  local url="$3"
  if [[ -n "${!key:-}" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    die "${key} is required for ${label}. Create it at ${url} and export ${key}=..."
  fi
  echo
  echo "${label} requires ${key}."
  echo "Create or copy it from:"
  echo "  ${url}"
  echo
  local value save_answer
  read -r -s -p "Paste ${key} here: " value
  echo
  [[ -n "$value" ]] || die "No ${key} was provided."
  export "$key=$value"
  read -r -p "Save ${key} to ${SCRIPT_DIR}/.cloud_keys for future runs? [y/N] " save_answer
  case "${save_answer,,}" in
    y|yes|s|si|sì)
      save_cloud_key "$key" "$value"
      info "${key} saved to ${SCRIPT_DIR}/.cloud_keys with restricted permissions."
      ;;
    *)
      info "${key} will only be used for this run."
      ;;
  esac
}

ensure_ollama_ready() {
  [[ "$LLM_ADAPT" != "never" || "$LLM_SEGMENT" == "always" ]] || return 0
  [[ "$LLM_PROVIDER" == "ollama" ]] || return 0

  if ! command -v ollama >/dev/null 2>&1; then
    if [[ "$OLLAMA_INSTALL" == "never" ]]; then
      die "Ollama is required by the selected LLM options, but it is not installed."
    fi
    info "Ollama is not installed. Installing it with the official installer..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  if ! ollama_api_ok; then
    info "Starting a local Ollama server..."
    (OLLAMA_HOST=127.0.0.1:11434 ollama serve >> "${LOG_DIR}/ollama_${RUN_ID}.log" 2>&1 &)
    for _ in $(seq 1 30); do
      if ollama_api_ok; then
        break
      fi
      sleep 1
    done
  fi

  if ! ollama_api_ok; then
    die "Ollama is installed but the local API is not reachable at 127.0.0.1:11434."
  fi

  if ! ollama show "$LLM_MODEL" >/dev/null 2>&1; then
    info "Pulling Ollama model: ${LLM_MODEL}"
    ollama pull "$LLM_MODEL"
  else
    info "Ollama model already available: ${LLM_MODEL}"
  fi
}

print_banner
info "Log: ${LOG_FILE}"
if [[ -n "$INPUT_FILE" ]]; then
  info "Input: ${INPUT_FILE}"
fi
if [[ -n "$SOURCE_LANG" ]]; then
  info "Source language: ${SOURCE_LANG}"
fi
info "Target language: ${TARGET_LANG}"
if [[ -n "$TRANSLATION_METHOD" ]]; then
  info "Translation method: ${TRANSLATION_METHOD}"
fi
info "TTS engine: ${TTS_ENGINE}"
if [[ "$TTS_CATALOG_ACTION" -eq 0 ]]; then
  info "ASR backend: ${ASR_BACKEND}"
  info "Diarization backend: ${DIARIZATION_BACKEND}"
  if [[ "$ONLY_CLOUD" == "1" ]]; then
    info "Cloud mode: enabled by --only-cloud"
  fi
fi
if [[ "$NO_GPU" == "1" ]]; then
  info "GPU usage: disabled by --no-gpu"
  if ollama_api_ok; then
    warn "--no-gpu will request num_gpu=0 for Ollama API calls, but an already-running external Ollama server may have been started with its own GPU environment."
  fi
fi
if [[ "$TTS_CATALOG_ACTION" -eq 0 ]]; then
  if [[ "$LLM_ADAPT" == "never" ]]; then
    info "LLM adaptation: never"
  else
    if [[ "$LLM_PROVIDER" == "groq" ]]; then
      info "LLM adaptation: ${LLM_ADAPT} (groq:${GROQ_LLM_MODEL})"
    else
      info "LLM adaptation: ${LLM_ADAPT} (ollama:${LLM_MODEL})"
    fi
  fi
  info "LLM segmentation: ${LLM_SEGMENT}"
fi
system_deps=(ffmpeg python3 curl)
if [[ "$TTS_ENGINE" == "kokoro" ]]; then
  system_deps+=(espeak-ng)
fi
auto_apt_install "${system_deps[@]}"
ensure_python_venv_support

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"
export TORCH_HOME="${MODELS_DIR}/torch"
export XDG_CACHE_HOME="${WORK_DIR}/cache"
export XDG_DATA_HOME="${XDG_DATA_DIR}"
export TTS_HOME="${TTS_PREFIX_DIR}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TTS_HOME" "$TTS_CACHE_DIR"


copy_dir_contents() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "$dst_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src_dir/" "$dst_dir/"
  else
    cp -a "$src_dir/." "$dst_dir/"
  fi
}

migrate_existing_tts_cache() {
  local legacy_xtts="${HOME}/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
  local wrong_nested_xtts="${MODELS_DIR}/tts/tts/tts_models--multilingual--multi-dataset--xtts_v2"
  local local_xtts="${TTS_CACHE_DIR}/tts_models--multilingual--multi-dataset--xtts_v2"

  if [[ -d "$wrong_nested_xtts" ]]; then
    info "Found an XTTS cache in an old nested project path. Normalizing it into the project cache..."
    copy_dir_contents "$wrong_nested_xtts" "$local_xtts"
  fi

  if [[ -d "$legacy_xtts" && ! -d "$local_xtts" ]]; then
    info "Found an existing XTTS cache in the default user location. Migrating it into the project cache..."
    copy_dir_contents "$legacy_xtts" "$local_xtts"
  fi

  if [[ -d "$wrong_nested_xtts" && "$wrong_nested_xtts" != "$local_xtts" ]]; then
    rm -rf "$wrong_nested_xtts"
    local wrong_parent
    wrong_parent="$(dirname "$wrong_nested_xtts")"
    rmdir "$wrong_parent" 2>/dev/null || true
  fi
}

migrate_existing_tts_cache

if [[ "$TTS_CATALOG_ACTION" -eq 0 ]]; then
  load_cloud_keys

  if [[ "$ASR_BACKEND" == "groq" || ( "$LLM_PROVIDER" == "groq" && ( "$LLM_ADAPT" != "never" || "$LLM_SEGMENT" == "always" || ( "$LLM_SEGMENT" == "auto" && "$ONLY_CLOUD" == "1" ) ) ) ]]; then
    ensure_cloud_key "GROQ_API_KEY" "GroqCloud" "https://console.groq.com/keys"
  fi
  if [[ "$ASR_BACKEND" == "assemblyai" || "$DIARIZATION_BACKEND" == "assemblyai" ]]; then
    ensure_cloud_key "ASSEMBLYAI_API_KEY" "AssemblyAI" "https://www.assemblyai.com/dashboard/signup"
  fi

  if [[ "$DIARIZATION_BACKEND" == "local" ]]; then
    if [[ -f "${SCRIPT_DIR}/.hf_token" && -z "${HF_TOKEN:-}" ]]; then
      HF_TOKEN="$(<"${SCRIPT_DIR}/.hf_token")"
      export HF_TOKEN
    fi

    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo
      echo "A Hugging Face READ token is required for local pyannote diarization."
      echo "Before continuing, accept the model terms once at:"
      echo "  1) https://huggingface.co/pyannote/speaker-diarization-3.1"
      echo "  2) https://huggingface.co/pyannote/segmentation-3.0"
      echo "  3) https://huggingface.co/settings/tokens  (create a READ token)"
      echo
      read -r -s -p "Paste the Hugging Face READ token here: " HF_TOKEN
      echo
      export HF_TOKEN
      if [[ -n "$HF_TOKEN" ]]; then
        umask 077
        printf '%s' "$HF_TOKEN" > "${SCRIPT_DIR}/.hf_token"
        info "Token saved to ${SCRIPT_DIR}/.hf_token with restricted permissions."
      fi
    fi

    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "[ERROR] No HF token was provided."
      exit 1
    fi
  fi

  ensure_ollama_ready
fi

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating the local Python virtual environment..."
  if command -v python3.12 &>/dev/null; then
    python3.12 -m venv "$VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade pip setuptools wheel

GPU_HINT=0
if [[ "$NO_GPU" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_HINT=1
fi

# Install Python packages according to translation backend and TTS engine.
if python_imports_ok; then
  info "Python packages are already available in the virtual environment. Reinstallation skipped."
else
  if [[ "$GPU_HINT" -eq 1 ]]; then
    info "NVIDIA GPU detected. Installing CUDA-enabled PyTorch and CUDA runtime packages for faster-whisper."
    if [[ "$ASR_BACKEND" == "local" || "$DIARIZATION_BACKEND" == "local" || "$TRANSLATION_METHOD" == "local" || "$TTS_ENGINE" == "xtts" || "$TTS_ENGINE" == "kokoro" ]]; then
      pip install --index-url https://download.pytorch.org/whl/cu128 "torch<2.9" "torchaudio<2.9"
      pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
    fi
  else
    if [[ "$ASR_BACKEND" == "local" || "$DIARIZATION_BACKEND" == "local" || "$TRANSLATION_METHOD" == "local" || "$TTS_ENGINE" == "xtts" || "$TTS_ENGINE" == "kokoro" ]]; then
      info "No NVIDIA GPU detected. Installing CPU-only PyTorch."
      pip install --index-url https://download.pytorch.org/whl/cpu "torch<2.9" "torchaudio<2.9"
    fi
  fi

  packages=(
    "soundfile>=0.12.1"
    "numpy>=1.26,<2.0"
    "librosa"
    "numba<0.61"
    "requests>=2.32.0"
  )
  if [[ "$ASR_BACKEND" == "local" ]]; then
    packages+=("faster-whisper==1.2.1")
  fi
  if [[ "$DIARIZATION_BACKEND" == "local" ]]; then
    packages+=("pyannote-audio==3.3.2" "huggingface-hub>=0.34")
  fi
  if [[ "$TRANSLATION_METHOD" == "local" ]]; then
    packages+=("transformers>=4.57,<5.0" "sentencepiece>=0.2.0" "accelerate>=0.25,<1.0")
  elif [[ "$TRANSLATION_METHOD" == "google" ]]; then
    packages+=("deep-translator")
  fi
  if [[ "$TTS_ENGINE" == "xtts" ]]; then
    packages+=("coqui-tts==0.27.5")
  elif [[ "$TTS_ENGINE" == "kokoro" ]]; then
    packages+=("kokoro>=0.9.4")
    if [[ "$TARGET_LANG" == "ja" ]]; then
      packages+=("misaki[ja]>=0.9.4")
    elif [[ "$TARGET_LANG" == "zh" ]]; then
      packages+=("misaki[zh]>=0.9.4")
    fi
  elif [[ "$TTS_ENGINE" == "edge" ]]; then
    packages+=("edge-tts>=7.2.8")
  fi
  info "Installing or updating Python packages for translation=${TRANSLATION_METHOD:-none} tts=${TTS_ENGINE}..."
  pip install "${packages[@]}"
fi

if [[ "$GPU_HINT" -eq 1 ]]; then
  export LD_LIBRARY_PATH="$(python - <<'PY'
import os
paths=[]
for mod in [
    ('nvidia.cublas.lib',),
    ('nvidia.cudnn.lib',),
    ('nvidia.cuda_nvrtc.lib',),
]:
    name = mod[0]
    try:
        m = __import__(name, fromlist=['*'])
        paths.append(os.path.dirname(m.__file__))
    except Exception:
        pass
print(':'.join(paths))
PY
):${LD_LIBRARY_PATH:-}"
fi

cat > "$PY_SCRIPT" <<'PY'
#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import math
import logging
import hashlib
import subprocess
import re
import shutil
import inspect
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import soundfile as sf
import librosa


LOG = logging.getLogger("autodub")

# Lazy imports for optional modules (loaded only when needed)
_TorchModule = None
_TorchLoadPatched = False
_WhisperModelClass = None
_PyannotePipelineClass = None
_TTS_API = None
_TransformerTokenizer = None
_TransformerModel = None
_KokoroPipeline = None
_EdgeTTSModule = None


def _get_torch_module():
    global _TorchModule, _TorchLoadPatched
    if _TorchModule is None:
        import torch
        _TorchModule = torch
    if not _TorchLoadPatched:
        original_torch_load = _TorchModule.load

        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        _TorchModule.load = _patched_torch_load
        _TorchLoadPatched = True
    return _TorchModule


def _get_whisper_model_class():
    global _WhisperModelClass
    if _WhisperModelClass is None:
        from faster_whisper import WhisperModel
        _WhisperModelClass = WhisperModel
    return _WhisperModelClass


def _get_pyannote_pipeline_class():
    global _PyannotePipelineClass
    if _PyannotePipelineClass is None:
        from pyannote.audio import Pipeline as PyannotePipeline
        _PyannotePipelineClass = PyannotePipeline
    return _PyannotePipelineClass


def _get_tts_api():
    """Lazy load TTS.api only when XTTS is actually used."""
    global _TTS_API
    if _TTS_API is None:
        from TTS.api import TTS as _TTS_API
    return _TTS_API


def _get_transformer_modules():
    """Lazy load transformers modules only when NLLB translation is actually used."""
    global _TransformerTokenizer, _TransformerModel
    if _TransformerTokenizer is None or _TransformerModel is None:
        from transformers import AutoTokenizer as _TransformerTokenizer
        from transformers import AutoModelForSeq2SeqLM as _TransformerModel
    return _TransformerTokenizer, _TransformerModel


def _get_kokoro_pipeline():
    """Lazy load Kokoro only when Kokoro TTS is used."""
    global _KokoroPipeline
    if _KokoroPipeline is None:
        from kokoro import KPipeline as _KokoroPipeline
    return _KokoroPipeline


def _get_edge_tts_module():
    """Lazy load edge-tts only when Microsoft Edge TTS is used."""
    global _EdgeTTSModule
    if _EdgeTTSModule is None:
        import edge_tts as _EdgeTTSModule
    return _EdgeTTSModule

LANG_MAP = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn",
    "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn", "fa": "pes_Arab",
    "fi": "fin_Latn", "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr",
    "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "ko": "kor_Hang", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mr": "mar_Deva",
    "ms": "zsm_Latn", "nl": "nld_Latn", "no": "nob_Latn", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn",
    "zh": "zho_Hans",
}

XTTS_LANG_MAP = {
    "it": "it", "en": "en", "es": "es", "fr": "fr", "de": "de", "pt": "pt",
    "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi",
}

KOKORO_LANG_MAP = {
    "en": "a", "en-us": "a", "en-gb": "b", "es": "e", "fr": "f",
    "hi": "h", "it": "i", "ja": "j", "pt": "p", "zh": "z",
}

KOKORO_VOICE_CATALOG = {
    "en": [
        ("af_heart", "Female"), ("af_alloy", "Female"), ("af_aoede", "Female"), ("af_bella", "Female"),
        ("af_jessica", "Female"), ("af_kore", "Female"), ("af_nicole", "Female"), ("af_nova", "Female"),
        ("af_river", "Female"), ("af_sarah", "Female"), ("af_sky", "Female"),
        ("am_adam", "Male"), ("am_echo", "Male"), ("am_eric", "Male"), ("am_fenrir", "Male"),
        ("am_liam", "Male"), ("am_michael", "Male"), ("am_onyx", "Male"), ("am_puck", "Male"),
        ("am_santa", "Male"),
    ],
    "en-us": [
        ("af_heart", "Female"), ("af_alloy", "Female"), ("af_aoede", "Female"), ("af_bella", "Female"),
        ("af_jessica", "Female"), ("af_kore", "Female"), ("af_nicole", "Female"), ("af_nova", "Female"),
        ("af_river", "Female"), ("af_sarah", "Female"), ("af_sky", "Female"),
        ("am_adam", "Male"), ("am_echo", "Male"), ("am_eric", "Male"), ("am_fenrir", "Male"),
        ("am_liam", "Male"), ("am_michael", "Male"), ("am_onyx", "Male"), ("am_puck", "Male"),
        ("am_santa", "Male"),
    ],
    "en-gb": [
        ("bf_alice", "Female"), ("bf_emma", "Female"), ("bf_isabella", "Female"), ("bf_lily", "Female"),
        ("bm_daniel", "Male"), ("bm_fable", "Male"), ("bm_george", "Male"), ("bm_lewis", "Male"),
    ],
    "ja": [
        ("jf_alpha", "Female"), ("jf_gongitsune", "Female"), ("jf_nezumi", "Female"), ("jf_tebukuro", "Female"),
        ("jm_kumo", "Male"),
    ],
    "zh": [
        ("zf_xiaobei", "Female"), ("zf_xiaoni", "Female"), ("zf_xiaoxiao", "Female"), ("zf_xiaoyi", "Female"),
        ("zm_yunjian", "Male"), ("zm_yunxi", "Male"), ("zm_yunxia", "Male"), ("zm_yunyang", "Male"),
    ],
    "es": [("ef_dora", "Female"), ("em_alex", "Male"), ("em_santa", "Male")],
    "fr": [("ff_siwis", "Female")],
    "hi": [("hf_alpha", "Female"), ("hf_beta", "Female"), ("hm_omega", "Male"), ("hm_psi", "Male")],
    "it": [("if_sara", "Female"), ("im_nicola", "Male")],
    "pt": [("pf_dora", "Female"), ("pm_alex", "Male"), ("pm_santa", "Male")],
}

EDGE_DEFAULT_LOCALES = {
    "ar": "ar-SA", "bg": "bg-BG", "ca": "ca-ES", "cs": "cs-CZ", "da": "da-DK",
    "de": "de-DE", "el": "el-GR", "en": "en-US", "es": "es-ES", "fi": "fi-FI",
    "fr": "fr-FR", "he": "he-IL", "hi": "hi-IN", "hu": "hu-HU", "id": "id-ID",
    "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "nl": "nl-NL", "no": "nb-NO",
    "pl": "pl-PL", "pt": "pt-BR", "ro": "ro-RO", "ru": "ru-RU", "sk": "sk-SK",
    "sv": "sv-SE", "th": "th-TH", "tr": "tr-TR", "uk": "uk-UA", "vi": "vi-VN",
    "zh": "zh-CN",
}

TTS_PROFILE_VERSION = 3
TTS_SYNTHESIS_VERSION = 5
TTS_SPEAKER_PACING_VERSION = 1

LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "ka": "Georgian", "kk": "Kazakh", "ko": "Korean", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi",
    "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "zh": "Chinese",
}

GOOGLE_LANG_ALIASES = {
    "eng": "en", "ita": "it", "fra": "fr", "deu": "de", "spa": "es",
    "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh-cn",
    "nld": "nl", "pol": "pl", "tur": "tr", "ara": "ar", "arb": "ar",
    "hin": "hi", "zh": "zh-cn",
}

CJK_LANGS = {"ja", "ko", "zh"}
UTTERANCE_BOUNDARY_VERSION = 2
LLM_SEGMENT_PROMPT_VERSION = 1
LLM_PROMPT_VERSION = 3
LLM_BUDGET_VERSION = 5
NLLB_TRANSLATION_UNIT_VERSION = 1
GOOGLE_TRANSLATION_UNIT_VERSION = 2


def run(cmd: List[str], **kwargs):
    LOG.debug("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip()
    return float(out)


def ffprobe_audio_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=duration",
        "-select_streams", "a:0",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip().splitlines()
    vals = [float(x) for x in out if x.strip()]
    if vals:
        return vals[0]
    return ffprobe_duration(path)


def extract_audio(video: Path, wav_path: Path):
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ])


def extract_audio_segment(audio_path: Path, start: float, dur: float, out_path: Path):
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(audio_path), "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out_path)
    ])


def extract_audio_segment_flac(audio_path: Path, start: float, dur: float, out_path: Path):
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(audio_path), "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", "16000", "-c:a", "flac", str(out_path)
    ])


def convert_audio_to_flac(audio_path: Path, out_path: Path):
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path), "-ac", "1", "-ar", "16000", "-c:a", "flac", str(out_path)
    ])


def torch_cuda_usable(min_major: int = 7) -> bool:
    if os.environ.get("AUTODUB_NO_GPU", "0") == "1":
        return False
    try:
        torch = _get_torch_module()
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        if major < min_major:
            LOG.warning(
                "CUDA GPU is present but not compatible with the current PyTorch build: compute capability %s.%s < %s.0; using CPU for torch-based components",
                major, minor, min_major
            )
            return False
        # minimal probe to catch broken CUDA runtime setups
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as exc:
        LOG.warning("PyTorch CUDA is not usable, falling back to CPU: %s", exc)
        return False


def detect_torch_device() -> str:
    if os.environ.get("AUTODUB_NO_GPU", "0") == "1":
        return "cpu"
    return "cuda" if torch_cuda_usable() else "cpu"


def preload_audio_dict(audio_path: Path) -> Dict[str, Any]:
    torch = _get_torch_module()
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    return {"waveform": waveform, "sample_rate": int(sr)}


def load_pyannote_pipeline(model_id: str, hf_token: str):
    PyannotePipeline = _get_pyannote_pipeline_class()
    kwargs = {}
    try:
        params = inspect.signature(PyannotePipeline.from_pretrained).parameters
        if "token" in params:
            kwargs["token"] = hf_token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = hf_token
    except Exception:
        kwargs["token"] = hf_token

    try:
        return PyannotePipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "token" in kwargs:
            return PyannotePipeline.from_pretrained(model_id, use_auth_token=hf_token)
        return PyannotePipeline.from_pretrained(model_id, token=hf_token)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalized_payload_items(payload: Any) -> List[Dict]:
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unrecognized JSON payload format")


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_json(data: Any) -> str:
    return hashlib.sha1(stable_json(data).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def cache_config_matches(payload: Any, expected_config: Dict[str, Any]) -> bool:
    return isinstance(payload, dict) and payload.get("config") == expected_config


def utterances_fingerprint(utterances: List[Dict]) -> str:
    slim = [
        {
            "start": round(float(u.get("start", 0.0)), 3),
            "end": round(float(u.get("end", 0.0)), 3),
            "speaker": str(u.get("speaker", "")),
            "text": str(u.get("text", "")),
        }
        for u in utterances
    ]
    return sha1_json(slim)


def translation_text(utt: Dict) -> str:
    return (utt.get("text_translated") or "").strip()


def language_name(code: str) -> str:
    code = (code or "").split("_")[0].lower()
    return LANGUAGE_NAMES.get(code, code or "the target language")


def seg_start(seg):
    return float(seg["start"] if isinstance(seg, dict) else seg.start)


def seg_end(seg):
    return float(seg["end"] if isinstance(seg, dict) else seg.end)


def seg_text(seg):
    return (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")) or ""


def seg_words(seg):
    if isinstance(seg, dict):
        return seg.get("words", []) or []
    return getattr(seg, "words", None) or []


def seg_speaker(seg) -> str:
    if isinstance(seg, dict):
        return str(seg.get("speaker", "") or "")
    return str(getattr(seg, "speaker", "") or "")


def word_text(word: Any) -> str:
    if isinstance(word, dict):
        return (word.get("word") or word.get("text") or "").strip()
    return str(getattr(word, "word", getattr(word, "text", "")) or "").strip()


def word_start(word: Any, default: float = 0.0) -> float:
    value = word.get("start", default) if isinstance(word, dict) else getattr(word, "start", default)
    return float(value if value is not None else default)


def word_end(word: Any, default: float = 0.0) -> float:
    value = word.get("end", default) if isinstance(word, dict) else getattr(word, "end", default)
    return float(value if value is not None else default)


def word_speaker(word: Any) -> str:
    if isinstance(word, dict):
        return str(word.get("speaker", "") or "")
    return str(getattr(word, "speaker", "") or "")


def _get_requests_module():
    import requests
    return requests


def retry_after_seconds(response) -> float:
    raw = response.headers.get("Retry-After", "") if response is not None else ""
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


def cloud_request(method: str, url: str, *, service: str, timeout: Optional[float], max_retries: int, rate_limit_mode: str, **kwargs):
    requests = _get_requests_module()
    last_exc = None
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, max_retries + 1):
        response = None
        try:
            response = requests.request(method, url, timeout=timeout if timeout and timeout > 0 else None, **kwargs)
            if response.status_code < 400:
                return response
            if response.status_code == 429 and rate_limit_mode == "fail":
                raise RuntimeError(f"{service} rate limit reached: HTTP 429 {response.text[:500]}")
            if response.status_code in retryable_statuses and attempt < max_retries:
                wait = retry_after_seconds(response) or min(60.0, 2.0 ** attempt)
                LOG.warning("%s request returned HTTP %s; retrying in %.1fs", service, response.status_code, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{service} request failed: HTTP {response.status_code} {response.text[:1000]}")
        except Exception as exc:
            last_exc = exc
            if response is not None and response.status_code not in retryable_statuses:
                raise
            if attempt >= max_retries:
                break
            wait = min(60.0, 2.0 ** attempt)
            LOG.warning("%s request failed (%s); retrying in %.1fs", service, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"{service} request failed after {max_retries} attempts: {last_exc}")


def speaker_mapper():
    labels: Dict[str, str] = {}

    def map_label(label: Any) -> str:
        raw = str(label if label is not None else "").strip() or "unknown"
        if raw not in labels:
            labels[raw] = f"SPEAKER_{len(labels):02d}"
        return labels[raw]

    return map_label


def transcript_segments_to_diarization(asr_segments: List[Dict]) -> List[Dict]:
    diar = []
    for seg in asr_segments:
        speaker = seg_speaker(seg)
        if not speaker:
            continue
        start = seg_start(seg)
        end = seg_end(seg)
        if end > start:
            diar.append({"start": start, "end": end, "speaker": speaker})
    return diar


def serialize_transcript_segments(asr_segments) -> List[Dict[str, Any]]:
    items = []
    for seg in asr_segments:
        item = {
            "start": float(seg_start(seg)),
            "end": float(seg_end(seg)),
            "text": seg_text(seg),
            "words": [
                {
                    "start": float(word_start(w, seg_start(seg)) or 0.0),
                    "end": float(word_end(w, seg_end(seg)) or 0.0),
                    "word": word_text(w),
                    **({"speaker": word_speaker(w)} if word_speaker(w) else {}),
                }
                for w in seg_words(seg)
            ],
        }
        speaker = seg_speaker(seg)
        if speaker:
            item["speaker"] = speaker
        items.append(item)
    return items


def transcribe_audio(audio_path: Path, source_lang: str, model_name: str):
    torch = _get_torch_module()
    WhisperModel = _get_whisper_model_class()
    preferred = detect_torch_device()
    attempts = []
    if preferred == "cuda":
        attempts.append(("cuda", os.environ.get("ASR_COMPUTE_GPU", "int8_float16")))
    attempts.append(("cpu", os.environ.get("ASR_COMPUTE_CPU", "int8")))

    last_exc = None
    for device, compute_type in attempts:
        try:
            LOG.info("ASR with faster-whisper: model=%s device=%s compute=%s", model_name, device, compute_type)
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            kwargs = {
                "beam_size": int(os.environ.get("ASR_BEAM", "5")),
                "word_timestamps": True,
                "vad_filter": os.environ.get("ASR_VAD", "true").lower() == "true",
                "condition_on_previous_text": False,
            }
            if source_lang != "auto":
                kwargs["language"] = source_lang
            segments, info = model.transcribe(str(audio_path), **kwargs)
            segs = list(segments)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            LOG.info("ASR completed: detected_language=%s prob=%.3f segments=%d", info.language, info.language_probability, len(segs))
            return segs, info.language
        except Exception as exc:
            last_exc = exc
            LOG.exception("ASR failed on device=%s: %s", device, exc)
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    raise RuntimeError(f"ASR failed on all devices: {last_exc}")


def groq_transcribe_file(chunk_path: Path, source_lang: str) -> Dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for --asr-backend groq")
    model = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3").strip()
    timeout = float(os.environ.get("GROQ_TIMEOUT", "300"))
    max_retries = int(os.environ.get("GROQ_MAX_RETRIES", "5"))
    rate_limit_mode = os.environ.get("GROQ_RATE_LIMIT", "wait").strip().lower()
    prompt = os.environ.get("GROQ_PROMPT", "").strip()
    data = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("temperature", "0"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]
    if prompt:
        data.append(("prompt", prompt))
    if source_lang != "auto":
        data.append(("language", source_lang))
    headers = {"Authorization": f"Bearer {api_key}"}
    requests = _get_requests_module()
    last_exc = None
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, max_retries + 1):
        response = None
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (chunk_path.name, f, "audio/flac")}
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=timeout if timeout and timeout > 0 else None,
                )
            if response.status_code < 400:
                return response.json()
            if response.status_code == 429 and rate_limit_mode == "fail":
                raise RuntimeError(f"Groq Whisper rate limit reached: HTTP 429 {response.text[:500]}")
            if response.status_code in retryable_statuses and attempt < max_retries:
                wait = retry_after_seconds(response) or min(60.0, 2.0 ** attempt)
                LOG.warning("Groq Whisper returned HTTP %s; retrying in %.1fs", response.status_code, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq Whisper failed: HTTP {response.status_code} {response.text[:1000]}")
        except Exception as exc:
            last_exc = exc
            if response is not None and response.status_code not in retryable_statuses:
                raise
            if attempt >= max_retries:
                break
            wait = min(60.0, 2.0 ** attempt)
            LOG.warning("Groq Whisper request failed (%s); retrying in %.1fs", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Groq Whisper failed after {max_retries} attempts: {last_exc}")


def normalize_groq_words(words: List[Dict[str, Any]], offset: float, accept_start: float, accept_end: float) -> List[Dict[str, Any]]:
    result = []
    for word in words or []:
        text = (word.get("word") or word.get("text") or "").strip()
        start = offset + float(word.get("start", 0.0) or 0.0)
        end = offset + float(word.get("end", start) or start)
        mid = (start + end) / 2.0
        if mid < accept_start or mid >= accept_end:
            continue
        result.append({"start": start, "end": max(end, start + 0.01), "word": text})
    return result


def normalize_groq_segments(payload: Dict[str, Any], offset: float, accept_start: float, accept_end: float) -> List[Dict[str, Any]]:
    words = normalize_groq_words(payload.get("words") or [], offset, accept_start, accept_end)
    segments = []
    for seg in payload.get("segments") or []:
        start = offset + float(seg.get("start", 0.0) or 0.0)
        end = offset + float(seg.get("end", start) or start)
        mid = (start + end) / 2.0
        if mid < accept_start or mid >= accept_end:
            continue
        seg_words = [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
        segments.append({
            "start": start,
            "end": max(end, start + 0.01),
            "text": (seg.get("text") or "").strip(),
            "words": seg_words,
        })
    if not segments and words:
        segments.append({
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),
            "words": words,
        })
    return segments


def transcribe_audio_groq(audio_path: Path, source_lang: str) -> Tuple[List[Dict], str]:
    duration = ffprobe_audio_duration(audio_path)
    chunk_seconds = float(os.environ.get("GROQ_CHUNK_SECONDS", "120"))
    overlap = float(os.environ.get("GROQ_OVERLAP_SECONDS", "1.0"))
    if chunk_seconds <= 1:
        raise RuntimeError("--groq-chunk-seconds must be greater than 1")
    if overlap < 0 or overlap >= chunk_seconds / 2:
        raise RuntimeError("--groq-overlap-seconds must be >= 0 and less than half of --groq-chunk-seconds")

    chunks_key = hashlib.sha1(stable_json({
        "audio": file_fingerprint(audio_path),
        "source_lang": source_lang,
        "model": os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3"),
        "prompt": os.environ.get("GROQ_PROMPT", ""),
        "chunk_seconds": chunk_seconds,
        "overlap": overlap,
    }).encode("utf-8")).hexdigest()[:12]
    chunks_dir = audio_path.parent / f"groq_asr_chunks_{chunks_key}"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_segments: List[Dict] = []
    detected_lang = source_lang if source_lang != "auto" else "auto"
    logical_start = 0.0
    chunk_index = 0
    while logical_start < duration:
        logical_end = min(duration, logical_start + chunk_seconds)
        extract_start = max(0.0, logical_start - (overlap if logical_start > 0 else 0.0))
        extract_end = min(duration, logical_end + (overlap if logical_end < duration else 0.0))
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}_{extract_start:.3f}_{extract_end:.3f}.flac"
        cache_path = chunk_path.with_suffix(".json")
        if cache_path.exists() and cache_path.stat().st_size > 0:
            payload = load_json(cache_path)
            LOG.info("Groq ASR chunk %d cache reused", chunk_index + 1)
        else:
            extract_audio_segment_flac(audio_path, extract_start, extract_end - extract_start, chunk_path)
            LOG.info("Groq ASR chunk %d: %.2fs -> %.2fs", chunk_index + 1, extract_start, extract_end)
            payload = groq_transcribe_file(chunk_path, source_lang)
            save_json(cache_path, payload)
        if payload.get("language") and detected_lang == "auto":
            detected_lang = str(payload.get("language")).lower()
        all_segments.extend(normalize_groq_segments(payload, extract_start, logical_start, logical_end if logical_end < duration else duration + 0.001))
        logical_start = logical_end
        chunk_index += 1

    all_segments = sorted(all_segments, key=lambda s: (s["start"], s["end"]))
    LOG.info("Groq ASR completed: detected_language=%s segments=%d", detected_lang, len(all_segments))
    return all_segments, detected_lang


def assemblyai_upload(audio_path: Path) -> str:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is required for AssemblyAI transcription/diarization")
    timeout = float(os.environ.get("ASSEMBLYAI_TIMEOUT", "7200"))
    headers = {"Authorization": api_key}
    requests = _get_requests_module()
    last_exc = None
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, 6):
        response = None
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    data=f,
                    timeout=timeout if timeout and timeout > 0 else None,
                )
            if response.status_code < 400:
                upload_url = response.json().get("upload_url")
                if not upload_url:
                    raise RuntimeError("AssemblyAI upload response did not include upload_url")
                return upload_url
            if response.status_code in retryable_statuses and attempt < 5:
                wait = retry_after_seconds(response) or min(60.0, 2.0 ** attempt)
                LOG.warning("AssemblyAI upload returned HTTP %s; retrying in %.1fs", response.status_code, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"AssemblyAI upload failed: HTTP {response.status_code} {response.text[:1000]}")
        except Exception as exc:
            last_exc = exc
            if response is not None and response.status_code not in retryable_statuses:
                raise
            if attempt >= 5:
                break
            wait = min(60.0, 2.0 ** attempt)
            LOG.warning("AssemblyAI upload failed (%s); retrying in %.1fs", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"AssemblyAI upload failed after 5 attempts: {last_exc}")


def assemblyai_submit(upload_url: str, source_lang: str, num_speakers: Optional[int], min_speakers: Optional[int], max_speakers: Optional[int]) -> str:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    payload: Dict[str, Any] = {
        "audio_url": upload_url,
        "speaker_labels": True,
        "punctuate": True,
        "format_text": True,
    }
    speech_model = os.environ.get("ASSEMBLYAI_SPEECH_MODEL", "").strip()
    if speech_model:
        payload["speech_models"] = [item.strip() for item in speech_model.split(",") if item.strip()]
    if source_lang == "auto":
        payload["language_detection"] = True
        payload["language_confidence_threshold"] = 0.7
    else:
        payload["language_code"] = source_lang
    if num_speakers is not None:
        payload["speakers_expected"] = int(num_speakers)
    elif min_speakers is not None or max_speakers is not None:
        lower = int(min_speakers or 1)
        upper = int(max_speakers or max(lower, 10))
        payload["speaker_options"] = {
            "min_speakers_expected": lower,
            "max_speakers_expected": upper,
            "advanced_speaker_segmentation": True,
        }
    response = cloud_request(
        "POST",
        "https://api.assemblyai.com/v2/transcript",
        service="AssemblyAI submit",
        timeout=60,
        max_retries=5,
        rate_limit_mode="wait",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json=payload,
    )
    transcript_id = response.json().get("id")
    if not transcript_id:
        raise RuntimeError(f"AssemblyAI submit response did not include id: {response.text[:1000]}")
    return transcript_id


def assemblyai_poll(transcript_id: str) -> Dict[str, Any]:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    timeout_total = float(os.environ.get("ASSEMBLYAI_TIMEOUT", "7200"))
    interval = float(os.environ.get("ASSEMBLYAI_POLL_INTERVAL", "5"))
    deadline = time.monotonic() + timeout_total
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        response = cloud_request(
            "GET",
            url,
            service="AssemblyAI poll",
            timeout=60,
            max_retries=5,
            rate_limit_mode="wait",
            headers={"Authorization": api_key},
        )
        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {payload.get('error')}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"AssemblyAI transcription timed out after {timeout_total:.0f}s")
        LOG.info("AssemblyAI status: %s; polling again in %.1fs", status, interval)
        time.sleep(interval)


def assemblyai_payload_to_segments(payload: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], str]:
    map_speaker = speaker_mapper()
    utterances = payload.get("utterances") or []
    asr_segments: List[Dict] = []
    diar_segments: List[Dict] = []
    for utt in utterances:
        speaker = map_speaker(utt.get("speaker"))
        start = float(utt.get("start", 0) or 0) / 1000.0
        end = float(utt.get("end", 0) or 0) / 1000.0
        words = []
        for word in utt.get("words") or []:
            words.append({
                "start": float(word.get("start", 0) or 0) / 1000.0,
                "end": float(word.get("end", 0) or 0) / 1000.0,
                "word": (word.get("text") or word.get("word") or "").strip(),
                "speaker": speaker,
            })
        if end <= start and words:
            start, end = words[0]["start"], words[-1]["end"]
        if end <= start:
            continue
        text = (utt.get("text") or " ".join(w["word"] for w in words)).strip()
        asr_segments.append({"start": start, "end": end, "text": text, "speaker": speaker, "words": words})
        diar_segments.append({"start": start, "end": end, "speaker": speaker})

    if not asr_segments:
        map_speaker = speaker_mapper()
        words = []
        for word in payload.get("words") or []:
            speaker = map_speaker(word.get("speaker"))
            words.append({
                "start": float(word.get("start", 0) or 0) / 1000.0,
                "end": float(word.get("end", 0) or 0) / 1000.0,
                "word": (word.get("text") or "").strip(),
                "speaker": speaker,
            })
        if words:
            speaker = words[0]["speaker"]
            asr_segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": payload.get("text", ""), "speaker": speaker, "words": words})
            diar_segments.append({"start": words[0]["start"], "end": words[-1]["end"], "speaker": speaker})

    detected = (payload.get("language_code") or payload.get("language") or os.environ.get("SOURCE_LANG", "auto")).lower()
    return asr_segments, diar_segments, detected


def transcribe_and_diarize_assemblyai(audio_path: Path, source_lang: str, num_speakers: Optional[int], min_speakers: Optional[int], max_speakers: Optional[int]) -> Tuple[List[Dict], List[Dict], str]:
    flac_path = audio_path.with_suffix(".assemblyai.flac")
    if not flac_path.exists() or flac_path.stat().st_size == 0:
        convert_audio_to_flac(audio_path, flac_path)
    raw_cache_key = hashlib.sha1(stable_json({
        "audio": file_fingerprint(flac_path),
        "source_lang": source_lang,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "speech_model": os.environ.get("ASSEMBLYAI_SPEECH_MODEL", ""),
    }).encode("utf-8")).hexdigest()[:12]
    raw_cache = audio_path.with_suffix(f".assemblyai.{raw_cache_key}.raw.json")
    if raw_cache.exists() and raw_cache.stat().st_size > 0:
        payload = load_json(raw_cache)
        LOG.info("AssemblyAI raw transcript cache reused.")
    else:
        LOG.info("Uploading audio to AssemblyAI: %s", flac_path.name)
        upload_url = assemblyai_upload(flac_path)
        transcript_id = assemblyai_submit(upload_url, source_lang, num_speakers, min_speakers, max_speakers)
        LOG.info("AssemblyAI transcript id: %s", transcript_id)
        payload = assemblyai_poll(transcript_id)
        save_json(raw_cache, payload)
    asr_segments, diar_segments, detected_lang = assemblyai_payload_to_segments(payload)
    LOG.info("AssemblyAI completed: detected_language=%s transcript_segments=%d diarization_segments=%d", detected_lang, len(asr_segments), len(diar_segments))
    return asr_segments, diar_segments, detected_lang


def diarize_audio(
    audio_path: Path,
    hf_token: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    torch = _get_torch_module()
    # pyannote 4.x can use an internal audio decoder (torchcodec) that may fail in some environments;
    # by passing preloaded audio in memory, that code path is avoided entirely.
    audio_dict = preload_audio_dict(audio_path)
    preferred = detect_torch_device()
    attempts = [preferred] if preferred == "cpu" else ["cuda", "cpu"]
    last_exc = None
    for device in attempts:
        try:
            LOG.info(
                "pyannote diarization: device=%s num_speakers=%s min_speakers=%s max_speakers=%s",
                device, num_speakers if num_speakers is not None else "auto", min_speakers, max_speakers,
            )
            pipeline = load_pyannote_pipeline("pyannote/speaker-diarization-3.1", hf_token)
            if device == "cuda":
                pipeline.to(torch.device("cuda"))
                diar_input = {"waveform": audio_dict["waveform"].to(torch.device("cuda")), "sample_rate": audio_dict["sample_rate"]}
            else:
                diar_input = audio_dict
            diar_kwargs = {}
            if num_speakers is not None:
                diar_kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    diar_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diar_kwargs["max_speakers"] = max_speakers
            diar = pipeline(diar_input, **diar_kwargs)
            exclusive = getattr(diar, "exclusive_speaker_diarization", diar)
            segments = []
            for turn, _, speaker in exclusive.itertracks(yield_label=True):
                segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
            if not segments:
                raise RuntimeError("pyannote returned zero diarization segments")
            del pipeline
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            LOG.info("Diarization completed: %d segments", len(segments))
            return segments
        except Exception as exc:
            last_exc = exc
            LOG.exception("Diarization failed on device=%s: %s", device, exc)
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    msg = str(last_exc)
    if "gated" in msg.lower() or "403" in msg or "401" in msg:
        raise RuntimeError(
            "Access to pyannote models was denied. Accept the terms manually at: "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and then use a valid READ token."
        ) from last_exc
    raise RuntimeError(f"Diarization failed: {last_exc}")


def speaker_for_time(t: float, diar_segments: List[Dict]) -> str:
    for seg in diar_segments:
        if seg["start"] <= t < seg["end"]:
            return seg["speaker"]
    nearest = min(diar_segments, key=lambda s: min(abs(s["start"] - t), abs(s["end"] - t)))
    return nearest["speaker"]


def build_utterances(asr_segments, diar_segments: List[Dict]) -> List[Dict]:
    utterances = []
    current = None
    max_gap = float(os.environ.get("UTTERANCE_MAX_GAP", "0.9"))
    max_duration = float(os.environ.get("UTTERANCE_MAX_DURATION", "18.0"))
    max_chars = int(os.environ.get("UTTERANCE_MAX_CHARS", "420"))

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = " ".join(current["text"].split())
            utterances.append(current)
        current = None

    def should_merge(spk: str, start: float, end: float, text: str) -> bool:
        if current is None:
            return False
        if spk != current["speaker"]:
            return False
        gap = start - current["end"]
        if gap > max_gap:
            return False
        merged_duration = max(current["end"], end) - current["start"]
        if merged_duration > max_duration:
            return False
        merged_chars = len((current["text"] + " " + text).strip())
        if merged_chars > max_chars:
            return False
        return True

    def append_piece(start: float, end: float, speaker: str, text: str):
        nonlocal current
        text = text.strip()
        if not text:
            return
        if current is None:
            current = {"start": start, "end": end, "speaker": speaker, "text": text}
        elif should_merge(speaker, start, end, text):
            current["end"] = max(current["end"], end)
            current["text"] += " " + text
        else:
            flush()
            current = {"start": start, "end": end, "speaker": speaker, "text": text}

    for seg in asr_segments:
        words = seg_words(seg)
        if words:
            for w in words:
                text = ((w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")) or "").strip()
                if not text:
                    continue
                ws_raw = w.get("start", seg_start(seg)) if isinstance(w, dict) else getattr(w, "start", seg_start(seg))
                we_raw = w.get("end", seg_end(seg)) if isinstance(w, dict) else getattr(w, "end", seg_end(seg))
                ws = float(ws_raw if ws_raw is not None else seg_start(seg))
                we = float(we_raw if we_raw is not None else max(ws + 0.01, seg_end(seg)))
                spk = speaker_for_time((ws + we) / 2.0, diar_segments)
                append_piece(ws, we, spk, text)
        else:
            text = seg_text(seg).strip()
            if not text:
                continue
            s0, s1 = seg_start(seg), seg_end(seg)
            spk = speaker_for_time((s0 + s1) / 2.0, diar_segments)
            append_piece(s0, s1, spk, text)
    flush()
    initial_count = len(utterances)
    utterances = repair_utterance_boundaries(utterances)
    utterances = segment_utterances_with_llm_if_enabled(utterances)
    LOG.info(
        "Reconstructed utterances: %d -> %d (max_gap=%.2fs max_duration=%.2fs max_chars=%d)",
        initial_count, len(utterances), max_gap, max_duration, max_chars,
    )
    return utterances


def terminal_text(text: str) -> str:
    text = (text or "").strip()
    return text.rstrip(" \t\r\n\"'”’)]}»›")


def has_strong_terminal_punctuation(text: str) -> bool:
    text = terminal_text(text)
    return bool(text) and text[-1] in ".!?。！？؟।॥"


def merge_utterance_items(items: List[Dict], method: str) -> Dict:
    first = items[0]
    last = items[-1]
    merged = dict(first)
    merged["start"] = float(first["start"])
    merged["end"] = float(last["end"])
    merged["speaker"] = first["speaker"]
    merged["text"] = " ".join(str(item.get("text", "")).strip() for item in items if str(item.get("text", "")).strip())
    if len(items) > 1:
        merged["utterance_boundary"] = {
            "method": method,
            "merged_count": len(items),
            "source_starts": [round(float(item.get("start", 0.0)), 3) for item in items],
            "source_ends": [round(float(item.get("end", 0.0)), 3) for item in items],
        }
    return merged


def repair_utterance_boundaries(utterances: List[Dict]) -> List[Dict]:
    if len(utterances) < 2:
        return utterances
    repair_gap = float(os.environ.get("UTTERANCE_REPAIR_MAX_GAP", "2.2"))
    repair_max_duration = float(os.environ.get("UTTERANCE_REPAIR_MAX_DURATION", "24.0"))
    repair_max_chars = int(os.environ.get("UTTERANCE_REPAIR_MAX_CHARS", "620"))
    repaired: List[Dict] = []
    merges = 0
    idx = 0
    ordered = sorted(utterances, key=lambda x: float(x.get("start", 0.0)))
    while idx < len(ordered):
        group = [ordered[idx]]
        idx += 1
        while idx < len(ordered):
            prev = group[-1]
            nxt = ordered[idx]
            if str(prev.get("speaker", "")) != str(nxt.get("speaker", "")):
                break
            gap = float(nxt.get("start", 0.0)) - float(prev.get("end", 0.0))
            if gap < -0.05 or gap > repair_gap:
                break
            if has_strong_terminal_punctuation(str(prev.get("text", ""))):
                break
            candidate = group + [nxt]
            duration = float(candidate[-1].get("end", 0.0)) - float(candidate[0].get("start", 0.0))
            chars = len(" ".join(str(item.get("text", "")).strip() for item in candidate))
            if duration > repair_max_duration or chars > repair_max_chars:
                break
            group.append(nxt)
            idx += 1
        if len(group) > 1:
            merges += len(group) - 1
        repaired.append(merge_utterance_items(group, "heuristic_boundary_repair"))
    if merges:
        LOG.info(
            "Utterance boundary repair merged %d adjacent split(s): %d -> %d",
            merges, len(utterances), len(repaired),
        )
    return repaired


def llm_segmentation_should_run(utterances: List[Dict]) -> bool:
    mode = os.environ.get("LLM_SEGMENT", "auto").strip().lower()
    if mode == "never" or len(utterances) < 2:
        return False
    if mode == "always":
        return True
    return os.environ.get("ONLY_CLOUD", "0") == "1" and os.environ.get("LLM_PROVIDER", "ollama").strip().lower() == "groq"


def llm_adapter_for_segmentation(source_lang: str):
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "groq":
        model = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b").strip()
        return GroqLLMAdapter(model, source_lang)
    model = os.environ.get("LLM_MODEL", "qwen3:8b-q4_K_M").strip()
    return OllamaAdapter(model, source_lang)


def build_llm_segmentation_prompt(utterances: List[Dict], source_lang: str) -> str:
    repair_gap = float(os.environ.get("UTTERANCE_REPAIR_MAX_GAP", "2.2"))
    repair_max_duration = float(os.environ.get("UTTERANCE_REPAIR_MAX_DURATION", "24.0"))
    repair_max_chars = int(os.environ.get("UTTERANCE_REPAIR_MAX_CHARS", "620"))
    items = []
    prev_end = None
    for idx, utt in enumerate(utterances, start=1):
        start = float(utt.get("start", 0.0))
        end = float(utt.get("end", 0.0))
        items.append({
            "id": idx,
            "speaker": str(utt.get("speaker", "")),
            "start": round(start, 3),
            "end": round(end, 3),
            "gap_from_previous": None if prev_end is None else round(start - prev_end, 3),
            "text": str(utt.get("text", "")),
        })
        prev_end = end
    payload = {
        "source_language_code": source_lang,
        "max_small_gap_seconds": repair_gap,
        "preferred_max_group_duration_seconds": repair_max_duration,
        "preferred_max_group_chars": repair_max_chars,
        "items": items,
    }
    return f"""
You segment ASR output for a dubbing pipeline.
The input text may be in any language. Use punctuation, timing, same-speaker continuity, and natural sentence boundaries.

Rules:
- Group only adjacent item ids.
- Never reorder, drop, duplicate, or edit text.
- Never group different speakers.
- Prefer complete sentences or complete thoughts.
- Merge a same-speaker boundary when the previous item appears unfinished and the gap is small.
- Keep groups under the preferred duration and character limits when possible.
- Return only valid JSON with this exact shape: {{"groups":[[1,2],[3],[4,5]]}}

Input JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def validate_llm_segmentation_groups(groups: Any, utterances: List[Dict]) -> List[List[int]]:
    if not isinstance(groups, list):
        raise ValueError("groups is not a list")
    repair_max_duration = float(os.environ.get("UTTERANCE_REPAIR_MAX_DURATION", "24.0"))
    repair_max_chars = int(os.environ.get("UTTERANCE_REPAIR_MAX_CHARS", "620"))
    normalized: List[List[int]] = []
    expected = 1
    for raw_group in groups:
        if isinstance(raw_group, dict):
            raw_group = raw_group.get("ids")
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError("each group must be a non-empty list")
        ids = [int(value) for value in raw_group]
        if ids != list(range(expected, expected + len(ids))):
            raise ValueError("groups must contain contiguous ordered ids")
        group_items = [utterances[item_id - 1] for item_id in ids]
        speakers = {str(item.get("speaker", "")) for item in group_items}
        if len(speakers) != 1:
            raise ValueError("a group crosses speakers")
        if len(group_items) > 1:
            duration = float(group_items[-1].get("end", 0.0)) - float(group_items[0].get("start", 0.0))
            chars = len(" ".join(str(item.get("text", "")).strip() for item in group_items))
            if duration > repair_max_duration or chars > repair_max_chars:
                raise ValueError("a group exceeds repair limits")
        normalized.append(ids)
        expected += len(ids)
    if expected != len(utterances) + 1:
        raise ValueError("groups do not cover every item exactly once")
    return normalized


def apply_llm_segmentation_groups(utterances: List[Dict], groups: List[List[int]]) -> List[Dict]:
    segmented = []
    for ids in groups:
        items = [utterances[item_id - 1] for item_id in ids]
        segmented.append(merge_utterance_items(items, "llm_boundary_segmentation"))
    return segmented


def segment_utterances_with_llm_if_enabled(utterances: List[Dict]) -> List[Dict]:
    if not llm_segmentation_should_run(utterances):
        return utterances
    source_lang = os.environ.get("SOURCE_LANG", "auto").strip().lower()
    try:
        adapter = llm_adapter_for_segmentation(source_lang)
        prompt = build_llm_segmentation_prompt(utterances, source_lang)
        parsed = adapter._generate(prompt)
        groups = validate_llm_segmentation_groups(parsed.get("groups"), utterances)
        segmented = apply_llm_segmentation_groups(utterances, groups)
        if len(segmented) != len(utterances):
            LOG.info("LLM segmentation changed utterances: %d -> %d", len(utterances), len(segmented))
        else:
            LOG.info("LLM segmentation kept utterance count unchanged: %d", len(utterances))
        return segmented
    except Exception as exc:
        LOG.warning("LLM segmentation failed; keeping heuristic utterance boundaries: %s", exc)
        return utterances


def extract_reference_clips(audio_path: Path, diar_segments: List[Dict], work_dir: Path, max_clips: int = 3) -> Dict[str, List[str]]:
    refs_root = work_dir / "reference_clips"
    raw_dir = refs_root / "raw"
    clean_dir = refs_root / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    by_spk = defaultdict(list)
    fallback_by_spk = defaultdict(list)
    for seg in diar_segments:
        dur = seg["end"] - seg["start"]
        if dur >= 4.0:
            by_spk[seg["speaker"]].append(seg)
        if dur >= 0.75:
            fallback_by_spk[seg["speaker"]].append(seg)
    if not by_spk and fallback_by_spk:
        LOG.warning("No reference clip is at least 4 seconds long. Falling back to shorter clips.")
        by_spk = fallback_by_spk
    result = {}
    for spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda x: (x["end"] - x["start"]), reverse=True)[:max_clips]
        clips = []
        for idx, seg in enumerate(segs, start=1):
            raw_out = raw_dir / f"{spk}_{idx}.wav"
            clean_out = clean_dir / f"{spk}_{idx}.wav"
            if not raw_out.exists() or raw_out.stat().st_size == 0:
                start = seg["start"]
                dur = min(seg["end"] - seg["start"], 12.0)
                extract_audio_segment(audio_path, start, dur, raw_out)
            if (
                not clean_out.exists()
                or clean_out.stat().st_size == 0
                or clean_out.stat().st_mtime_ns < raw_out.stat().st_mtime_ns
            ):
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(raw_out),
                    "-filter:a", "highpass=f=80,afftdn=nr=12",
                    "-ar", "24000", "-ac", "1", str(clean_out)
                ]
                run(cmd)
            clips.append(str(clean_out))
        if clips:
            result[spk] = clips
    if not result:
        raise RuntimeError("Could not derive reference voice clips")
    LOG.info("Clean reference clips available for %d speakers", len(result))
    return result


def parse_tts_voice_map(raw: str) -> Dict[str, str]:
    result = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise RuntimeError(f"Invalid TTS voice map item '{part}'. Expected SPEAKER_00=voice_name.")
        speaker, voice = part.split("=", 1)
        speaker = speaker.strip()
        voice = voice.strip()
        if not re.fullmatch(r"SPEAKER_\d{2,}", speaker):
            raise RuntimeError(f"Invalid speaker id in TTS voice map: '{speaker}'. Expected SPEAKER_00 style ids.")
        if not voice:
            raise RuntimeError(f"Missing voice name for {speaker} in TTS voice map.")
        result[speaker] = voice
    return result


def normalize_locale(locale: str) -> str:
    locale = (locale or "").replace("_", "-").strip()
    if not locale:
        return ""
    parts = [p for p in locale.split("-") if p]
    if len(parts) == 1:
        return parts[0].lower()
    normalized = [parts[0].lower()]
    normalized.extend(p.upper() if len(p) == 2 else p.title() for p in parts[1:])
    return "-".join(normalized)


def edge_locale_for_target(target_lang: str, voices: Optional[List[Dict[str, Any]]] = None) -> str:
    override = normalize_locale(os.environ.get("TTS_LOCALE", ""))
    if override:
        return override
    target = normalize_locale(target_lang)
    if "-" in target:
        return target
    if target in EDGE_DEFAULT_LOCALES:
        return EDGE_DEFAULT_LOCALES[target]
    voices = voices if voices is not None else edge_all_voices()
    prefix = target.lower() + "-"
    locales = sorted({normalize_locale(v.get("Locale", "")) for v in voices if str(v.get("Locale", "")).lower().startswith(prefix)})
    if locales:
        return locales[0]
    return target


_EDGE_ALL_VOICES_CACHE: Optional[List[Dict[str, Any]]] = None


def edge_voice_cache_file() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or ".autodub_local/cache") / "edge_tts_voices.json"


def load_cached_edge_voices() -> Optional[List[Dict[str, Any]]]:
    path = edge_voice_cache_file()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        voices = data.get("voices") if isinstance(data, dict) else data
        if isinstance(voices, list):
            return voices
    except Exception as exc:
        LOG.warning("Ignoring unreadable Edge TTS voice catalog cache %s: %s", path, exc)
    return None


def save_cached_edge_voices(voices: List[Dict[str, Any]]) -> None:
    path = edge_voice_cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"voices": voices}, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as exc:
        LOG.warning("Could not write Edge TTS voice catalog cache %s: %s", path, exc)


def edge_all_voices() -> List[Dict[str, Any]]:
    global _EDGE_ALL_VOICES_CACHE
    if _EDGE_ALL_VOICES_CACHE is None:
        edge_tts = _get_edge_tts_module()
        try:
            _EDGE_ALL_VOICES_CACHE = asyncio.run(edge_tts.list_voices())
            save_cached_edge_voices(_EDGE_ALL_VOICES_CACHE)
        except Exception as exc:
            cached = load_cached_edge_voices()
            if cached is None:
                raise RuntimeError(
                    "Could not download the Microsoft Edge TTS voice catalog, and no cached catalog is available. "
                    "Check the network connection or run --list-tts-voices once when the service is reachable."
                ) from exc
            LOG.warning("Could not refresh Microsoft Edge TTS voice catalog (%s). Using cached catalog.", exc)
            _EDGE_ALL_VOICES_CACHE = cached
    return _EDGE_ALL_VOICES_CACHE


def edge_voice_catalog_for_target(target_lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    voices = edge_all_voices()
    requested_locale = normalize_locale(os.environ.get("TTS_LOCALE", ""))
    locale = edge_locale_for_target(target_lang, voices)
    exact = [
        {
            "name": v.get("ShortName", ""),
            "gender": v.get("Gender", ""),
            "locale": normalize_locale(v.get("Locale", "")),
            "friendly_name": v.get("FriendlyName", ""),
            "engine": "edge",
        }
        for v in voices
        if normalize_locale(v.get("Locale", "")).lower() == locale.lower()
    ]
    if exact:
        return locale, sorted(exact, key=lambda v: v["name"])
    if requested_locale:
        raise RuntimeError(f"No Microsoft Edge TTS voices found for locale '{requested_locale}'.")
    target = normalize_locale(target_lang).split("-")[0].lower()
    prefix = target + "-"
    fallback = [
        {
            "name": v.get("ShortName", ""),
            "gender": v.get("Gender", ""),
            "locale": normalize_locale(v.get("Locale", "")),
            "friendly_name": v.get("FriendlyName", ""),
            "engine": "edge",
        }
        for v in voices
        if normalize_locale(v.get("Locale", "")).lower().startswith(prefix)
    ]
    if not fallback:
        raise RuntimeError(f"No Microsoft Edge TTS voices found for target language '{target_lang}'.")
    locale = sorted({v["locale"] for v in fallback})[0]
    selected = [v for v in fallback if v["locale"] == locale]
    return locale, sorted(selected, key=lambda v: v["name"])


def kokoro_voice_catalog_for_target(target_lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    target = normalize_locale(target_lang).lower()
    voices = KOKORO_VOICE_CATALOG.get(target)
    if voices is None and target == "en":
        voices = KOKORO_VOICE_CATALOG.get("en")
    if voices is None:
        supported = ", ".join(sorted(KOKORO_VOICE_CATALOG))
        raise RuntimeError(f"Target language '{target_lang}' is not supported by Kokoro TTS. Supported targets: {supported}")
    return target, [
        {
            "name": name,
            "gender": gender,
            "locale": target,
            "friendly_name": name,
            "engine": "kokoro",
        }
        for name, gender in voices
    ]


def tts_voice_catalog(tts_engine: str, target_lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    if tts_engine == "edge":
        return edge_voice_catalog_for_target(target_lang)
    if tts_engine == "kokoro":
        return kokoro_voice_catalog_for_target(target_lang)
    return target_lang, []


def voice_names(catalog: List[Dict[str, Any]]) -> set:
    return {str(v.get("name", "")) for v in catalog if v.get("name")}


def validate_tts_voice_options(tts_engine: str, target_lang: str, num_speakers_raw: str = "auto") -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
    manual_map = parse_tts_voice_map(os.environ.get("TTS_VOICE_MAP", ""))
    locale, catalog = tts_voice_catalog(tts_engine, target_lang)
    if tts_engine == "xtts":
        if manual_map:
            raise RuntimeError("--tts-voice-map is only valid with --tts-engine kokoro or edge.")
        return locale, catalog, manual_map

    supported = voice_names(catalog)
    for speaker, voice in manual_map.items():
        if voice not in supported:
            raise RuntimeError(f"Voice '{voice}' mapped for {speaker} is not available for {tts_engine} target '{target_lang}'/locale '{locale}'.")
    for env_name, label in [
        ("TTS_VOICE_FEMALE", "female"),
        ("TTS_VOICE_MALE", "male"),
        ("TTS_VOICE_CHILD", "child"),
    ]:
        voice = os.environ.get(env_name, "").strip()
        if voice and voice not in supported:
            raise RuntimeError(f"--tts-voice-{label} voice '{voice}' is not available for {tts_engine} target '{target_lang}'/locale '{locale}'.")

    if num_speakers_raw != "auto":
        expected = {f"SPEAKER_{i:02d}" for i in range(int(num_speakers_raw))}
        extra = set(manual_map) - expected
        missing = expected - set(manual_map)
        if os.environ.get("TTS_VOICE_MAP_STRICT", "0") == "1" and (extra or missing):
            raise RuntimeError(
                "--tts-voice-map-strict requires exactly one mapped voice per expected speaker "
                f"({len(expected)} expected, {len(manual_map)} mapped)."
            )
        if extra:
            LOG.warning("TTS voice map includes speakers outside --num-speakers %s and they will be ignored: %s", num_speakers_raw, ", ".join(sorted(extra)))
    return locale, catalog, manual_map


def catalog_by_voice_class(catalog: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    female = [v for v in catalog if str(v.get("gender", "")).lower() == "female"]
    male = [v for v in catalog if str(v.get("gender", "")).lower() == "male"]
    all_voices = list(catalog)
    return {
        "female": female or all_voices,
        "male": male or all_voices,
        "child": female or all_voices,
    }


def apply_voice_class_overrides(by_class: Dict[str, List[Dict[str, Any]]], catalog: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_name = {v["name"]: v for v in catalog}
    overrides = {
        "female": os.environ.get("TTS_VOICE_FEMALE", "").strip(),
        "male": os.environ.get("TTS_VOICE_MALE", "").strip(),
        "child": os.environ.get("TTS_VOICE_CHILD", "").strip(),
    }
    result = {k: list(v) for k, v in by_class.items()}
    for key, value in overrides.items():
        if value and value in by_name:
            result[key] = [by_name[value]]
    return result


def classify_voice_from_pitch(median_f0: Optional[float]) -> str:
    if median_f0 is None or not math.isfinite(median_f0):
        return "female"
    if median_f0 >= 280.0:
        return "child"
    if median_f0 >= 165.0:
        return "female"
    return "male"


def speaker_source_char_rates(utterances: List[Dict]) -> Dict[str, float]:
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"chars": 0.0, "duration": 0.0})
    for utt in utterances or []:
        speaker = str(utt.get("speaker", ""))
        if not speaker:
            continue
        duration = max(0.0, float(utt.get("end", 0.0)) - float(utt.get("start", 0.0)))
        if duration <= 0:
            continue
        text = str(utt.get("text", "") or "").strip()
        totals[speaker]["chars"] += float(len(text))
        totals[speaker]["duration"] += duration
    rates = {}
    for speaker, values in totals.items():
        if values["duration"] > 0:
            rates[speaker] = values["chars"] / values["duration"]
    return rates


def dynamic_chars_per_second_for_utterance(utt: Dict, target_lang: str, speaker_rates: Dict[str, float]) -> float:
    configured = os.environ.get("LLM_CHARS_PER_SECOND", "").strip()
    if configured:
        return float(configured)
    base = target_chars_per_second(target_lang)
    if target_lang in CJK_LANGS:
        return base
    speaker = str(utt.get("speaker", ""))
    source_rate = speaker_rates.get(speaker)
    if source_rate is None or not math.isfinite(source_rate) or source_rate <= 0:
        return base
    # Preserve each speaker's relative pace, but keep the budget inside a range
    # that Kokoro can usually synthesize without large timing overflows.
    dynamic = source_rate * 0.95
    return max(base, min(16.0, dynamic))


def text_budget_for_utterance_with_rates(utt: Dict, target_lang: str, speaker_rates: Dict[str, float]) -> int:
    duration = max(0.1, float(utt.get("end", 0.0)) - float(utt.get("start", 0.0)))
    cps = dynamic_chars_per_second_for_utterance(utt, target_lang, speaker_rates)
    return max(24, int(round(duration * cps)))


def minimum_adapted_chars(machine_translation: str, budget: int) -> int:
    machine_len = len(machine_translation or "")
    if machine_len <= 0:
        return 0
    if machine_len <= budget:
        return 0
    return min(budget - 1, max(24, int(round(budget * 0.78))))


def speaker_speed_profiles(utterances: List[Dict]) -> Dict[str, float]:
    rates = speaker_source_char_rates(utterances)
    if not rates:
        return {}
    values = np.asarray([v for v in rates.values() if math.isfinite(v) and v > 0], dtype=np.float32)
    if values.size == 0:
        return {}
    median = float(np.median(values))
    if median <= 0:
        return {}
    base_speed = float(os.environ.get("TTS_SPEED", "1.0"))
    result = {}
    for speaker, rate in rates.items():
        relative = math.sqrt(max(0.25, rate / median))
        result[speaker] = max(0.92, min(1.10, base_speed * relative))
    return result


def estimate_speaker_pitch(audio_path: Path, diar_segments: List[Dict], max_seconds_per_speaker: float = 30.0) -> Dict[str, Optional[float]]:
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    by_spk: Dict[str, List[np.ndarray]] = defaultdict(list)
    used: Dict[str, float] = defaultdict(float)
    for seg in sorted(diar_segments, key=lambda s: (s["speaker"], -(s["end"] - s["start"]))):
        speaker = str(seg["speaker"])
        if used[speaker] >= max_seconds_per_speaker:
            continue
        start = max(0, int(float(seg["start"]) * sr))
        end = min(len(audio), int(float(seg["end"]) * sr))
        if end <= start:
            continue
        clip = audio[start:end]
        dur = len(clip) / sr
        if dur < 0.5:
            continue
        by_spk[speaker].append(clip)
        used[speaker] += dur

    result: Dict[str, Optional[float]] = {}
    for speaker, clips in by_spk.items():
        samples = np.concatenate(clips) if clips else np.array([], dtype=np.float32)
        if samples.size < sr // 2:
            result[speaker] = None
            continue
        try:
            f0 = librosa.yin(samples, fmin=70, fmax=500, sr=sr, frame_length=2048)
            f0 = np.asarray(f0, dtype=np.float32)
            f0 = f0[np.isfinite(f0)]
            f0 = f0[(f0 >= 70) & (f0 <= 500)]
            result[speaker] = float(np.median(f0)) if f0.size else None
        except Exception as exc:
            LOG.warning("Pitch estimation failed for %s: %s", speaker, exc)
            result[speaker] = None
    for seg in diar_segments:
        result.setdefault(str(seg["speaker"]), None)
    return result


def build_speaker_profiles(audio_path: Path, diar_segments: List[Dict], target_lang: str, utterances: List[Dict], tts_engine: str) -> Dict[str, Dict[str, Any]]:
    if tts_engine not in {"kokoro", "edge"}:
        return {}
    tts_locale, catalog, manual_map = validate_tts_voice_options(
        tts_engine,
        target_lang,
        os.environ.get("NUM_SPEAKERS", "auto").strip().lower(),
    )
    detected_speakers = sorted({str(seg["speaker"]) for seg in diar_segments})
    mapped_speakers = set(manual_map)
    missing = set(detected_speakers) - mapped_speakers
    extra = mapped_speakers - set(detected_speakers)
    strict = os.environ.get("TTS_VOICE_MAP_STRICT", "0") == "1"
    if strict and (missing or extra):
        raise RuntimeError(
            "--tts-voice-map-strict requires the mapped speakers to match diarization exactly. "
            f"Missing: {', '.join(sorted(missing)) or 'none'}; extra: {', '.join(sorted(extra)) or 'none'}."
        )
    if extra:
        LOG.warning("TTS voice map includes speakers not found by diarization and they will be ignored: %s", ", ".join(sorted(extra)))

    by_class = apply_voice_class_overrides(catalog_by_voice_class(catalog), catalog)
    class_counters: Dict[str, int] = defaultdict(int)
    pitch_by_speaker = estimate_speaker_pitch(audio_path, diar_segments)
    source_rates = speaker_source_char_rates(utterances)
    tts_speeds = speaker_speed_profiles(utterances)
    profiles = {}
    for speaker in detected_speakers:
        pitch = pitch_by_speaker.get(speaker)
        voice_class = classify_voice_from_pitch(pitch)
        if speaker in manual_map:
            voice = manual_map[speaker]
            voice_source = "manual"
        else:
            candidates = by_class.get(voice_class) or by_class.get("female") or catalog
            if not candidates:
                raise RuntimeError(f"No TTS voices are available for {tts_engine} target '{target_lang}'/locale '{tts_locale}'.")
            pos = class_counters[voice_class] % len(candidates)
            class_counters[voice_class] += 1
            voice = candidates[pos]["name"]
            voice_source = f"pitch_{voice_class}_auto"
        profiles[speaker] = {
            "pitch_hz": pitch,
            "voice_class": voice_class,
            "tts_engine": tts_engine,
            "tts_locale": tts_locale,
            "tts_voice": voice,
            "tts_voice_source": voice_source,
            "source_chars_per_second": source_rates.get(speaker),
            "tts_speed": tts_speeds.get(speaker, float(os.environ.get("TTS_SPEED", "1.0"))),
            "tts_pitch_shift_steps": 0.0,
            "tts_pitch_shift_source": "disabled",
        }
        LOG.info(
            "Speaker profile %s: pitch=%s class=%s tts_engine=%s tts_voice=%s source_cps=%s tts_speed=%.3f",
            speaker,
            f"{pitch:.1f}Hz" if pitch is not None else "unknown",
            voice_class,
            tts_engine,
            voice,
            f"{source_rates.get(speaker):.1f}" if source_rates.get(speaker) is not None else "unknown",
            profiles[speaker]["tts_speed"],
        )
    return profiles


class Translator:
    """NLLB-based local translator."""
    def __init__(self, src_code: str, tgt_code: str, model_name: str = "facebook/nllb-200-distilled-600M"):
        AutoTokenizer, AutoModelForSeq2SeqLM = _get_transformer_modules()
        self.torch = _get_torch_module()
        self.src_code = src_code
        self.tgt_code = tgt_code
        self.device = "cuda" if (os.environ.get("TRANSLATE_ON_GPU", "0") == "1" and torch_cuda_usable()) else "cpu"
        LOG.info("Loading NLLB translator: %s device=%s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_code)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_code)

    def translate_batch(self, texts: List[str]) -> List[str]:
        clean = [t.strip() for t in texts]
        inputs = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos,
                max_length=512,
                num_beams=4,
                repetition_penalty=1.05,
            )
        txt = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in txt]


NLLB_NO_SPLIT_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "dept", "univ", "inc", "ltd", "co",
}


def normalize_source_for_nllb(text: str) -> str:
    """Clean ASR artifacts that make sentence-level NLLB translation less reliable."""
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(\d+)\s*-\s*([A-Za-z])", r"\1-\2", text)
    text = re.sub(r"\bwww\s*\.\s*([A-Za-z0-9-]+)\s*\.\s*([A-Za-z]{2,})\b", r"www.\1.\2", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b((?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?=\s+[A-Z])",
        r"\1.",
        text,
    )
    return text.strip()


def previous_alpha_token(text: str, end_idx: int) -> str:
    left = text[:end_idx].rstrip()
    match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)?)$", left)
    return match.group(1).lower() if match else ""


def is_nllb_sentence_boundary(text: str, idx: int) -> bool:
    ch = text[idx]
    if ch not in ".!?":
        return False
    prev_ch = text[idx - 1] if idx > 0 else ""
    next_ch = text[idx + 1] if idx + 1 < len(text) else ""
    if ch == "." and prev_ch.isalnum() and next_ch.isalnum():
        return False
    if ch == "." and previous_alpha_token(text, idx) in NLLB_NO_SPLIT_ABBREVIATIONS:
        return False
    return idx + 1 == len(text) or next_ch.isspace() or next_ch in "\"')]}”’"


def split_long_nllb_unit(unit: str, max_chars: int = 280) -> List[str]:
    unit = unit.strip()
    if len(unit) <= max_chars:
        return [unit] if unit else []
    parts = re.split(r"([,;:])", unit)
    chunks = []
    current = ""
    for i in range(0, len(parts), 2):
        piece = parts[i].strip()
        if not piece:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        candidate = f"{current} {piece}{punct}".strip() if current else f"{piece}{punct}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = f"{piece}{punct}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    if len(chunks) == 1 and len(chunks[0]) > max_chars:
        text = chunks[0]
        chunks = [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]
    return [chunk for chunk in chunks if chunk]


def split_source_for_nllb(text: str) -> List[str]:
    text = normalize_source_for_nllb(text)
    if not text:
        return []
    units = []
    start = 0
    for idx, _ in enumerate(text):
        if not is_nllb_sentence_boundary(text, idx):
            continue
        unit = text[start:idx + 1].strip()
        if unit:
            units.extend(split_long_nllb_unit(unit))
        start = idx + 1
    tail = text[start:].strip()
    if tail:
        units.extend(split_long_nllb_unit(tail))
    return units or [text]


def join_translated_units(units: List[str]) -> str:
    text = " ".join(unit.strip() for unit in units if unit and unit.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


class GoogleTranslatorWrapper:
    """Google Translate wrapper using deep-translator library."""
    
    def __init__(self, src_lang: str, tgt_lang: str):
        """
        Initialize Google Translator.
        
        Args:
            src_lang: Source language code (e.g., 'en', 'it')
            tgt_lang: Target language code (e.g., 'en', 'it')
        """
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang
            LOG.info("Initialized Google Translator (deep-translator): %s -> %s", src_lang, tgt_lang)
        except ImportError:
            raise RuntimeError("deep-translator library not available. Install with: pip install deep-translator")
    
    def translate_one(self, text: str, attempts: int = 2) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                result = self.translator.translate(text)
                return (result or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(2.0, 0.5 * attempt))
        LOG.warning("Google Translate error for '%s': %s", text[:80], last_error)
        return ""

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate texts using Google Translate."""
        return [self.translate_one(text) for text in texts]


def google_code(lang: str) -> str:
    lang = (lang or "").strip()
    if not lang:
        return lang
    if "_" in lang:
        lang = lang.split("_", 1)[0]
    return GOOGLE_LANG_ALIASES.get(lang, lang)


def resolve_nllb_lang(lang: str, role: str) -> str:
    lang = (lang or "").strip()
    if "_" in lang:
        return lang
    mapped = LANG_MAP.get(lang)
    if mapped:
        return mapped
    raise RuntimeError(f"The {role} language '{lang}' is not mapped to an NLLB language code.")


def resolve_translation_codes(
    method: str,
    src_lang: str,
    detected_lang: str,
    target_lang: str,
    nllb_src_code: Optional[str] = None,
    nllb_tgt_code: Optional[str] = None,
) -> Tuple[str, str]:
    source_for_translation = detected_lang if src_lang == "auto" else src_lang
    if method == "google":
        return google_code(source_for_translation), google_code(target_lang)

    src_code = nllb_src_code if nllb_src_code and nllb_src_code != "auto" else resolve_nllb_lang(source_for_translation, "source")
    tgt_code = nllb_tgt_code if nllb_tgt_code and nllb_tgt_code != "auto" else resolve_nllb_lang(target_lang, "target")
    return src_code, tgt_code


def get_translator(method: str, src_code: str, tgt_code: str):
    """Factory for the selected translation backend."""
    if method == "google":
        return GoogleTranslatorWrapper(src_code, tgt_code)
    return Translator(src_code=src_code, tgt_code=tgt_code)


def translate_utterances_local_sentencewise(
    utterances: List[Dict],
    src_lang: str,
    detected_lang: str,
    target_lang: str,
    nllb_src_code: str = None,
    nllb_tgt_code: str = None,
) -> Tuple[List[Dict], str]:
    src_code, tgt_code = resolve_translation_codes(
        "local",
        src_lang,
        detected_lang,
        target_lang,
        nllb_src_code,
        nllb_tgt_code,
    )
    translator = Translator(src_code=src_code, tgt_code=tgt_code)
    batch_size = int(os.environ.get("TRANSLATE_BATCH", "12"))

    expanded_units = []
    per_utterance_units: List[List[Dict[str, str]]] = [[] for _ in utterances]
    for utt_idx, utt in enumerate(utterances):
        source_units = split_source_for_nllb(str(utt.get("text", "")))
        for unit_idx, source_unit in enumerate(source_units):
            expanded_units.append((utt_idx, unit_idx, source_unit))

    translated_unit_count = 0
    for i in range(0, len(expanded_units), batch_size):
        chunk = expanded_units[i:i + batch_size]
        texts = [item[2] for item in chunk]
        translated_texts = translator.translate_batch(texts)
        for (utt_idx, _unit_idx, source_unit), translated_text in zip(chunk, translated_texts):
            translated_text = (translated_text or "").strip()
            if not translated_text:
                LOG.warning("NLLB produced an empty translation unit; keeping source text: %r", source_unit)
                translated_text = source_unit
            per_utterance_units[utt_idx].append({
                "source": source_unit,
                "text_translated": translated_text,
            })
        translated_unit_count += len(chunk)
        LOG.info("Translation units %d/%d (local sentencewise)", translated_unit_count, len(expanded_units))

    translated = []
    for utt, unit_translations in zip(utterances, per_utterance_units):
        item = dict(utt)
        translated_parts = [unit["text_translated"] for unit in unit_translations]
        item["text_translated"] = join_translated_units(translated_parts)
        item["translation_units"] = unit_translations
        translated.append(item)

    if torch_cuda_usable():
        try:
            torch = _get_torch_module()
            torch.cuda.empty_cache()
        except Exception:
            pass

    return translated, src_code


def translation_identity_key(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip()).lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def google_translation_looks_failed(source: str, translated: str, src_code: str, tgt_code: str) -> bool:
    translated = (translated or "").strip()
    if not translated:
        return True
    if src_code == tgt_code:
        return False
    source_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", source or "")
    if len(source_words) < 3:
        return False
    return translation_identity_key(source) == translation_identity_key(translated)


def translate_google_unit_with_fallback(translator: GoogleTranslatorWrapper, source_unit: str, src_code: str, tgt_code: str) -> str:
    translated_text = translator.translate_one(source_unit)
    if not google_translation_looks_failed(source_unit, translated_text, src_code, tgt_code):
        return translated_text

    fallback_units = split_long_nllb_unit(source_unit, max_chars=120)
    if len(fallback_units) <= 1:
        LOG.warning("Google produced an empty or unchanged translation unit; keeping source text: %r", source_unit)
        return source_unit

    LOG.warning("Google failed a translation unit; retrying it as %d smaller fragments: %r", len(fallback_units), source_unit)
    translated_parts = []
    failed_parts = 0
    for fallback_unit in fallback_units:
        fallback_translation = translator.translate_one(fallback_unit)
        if google_translation_looks_failed(fallback_unit, fallback_translation, src_code, tgt_code):
            failed_parts += 1
            translated_parts.append(fallback_unit)
        else:
            translated_parts.append(fallback_translation)
    translated_text = join_translated_units(translated_parts)
    if failed_parts:
        LOG.warning("Google fallback left %d/%d fragments untranslated for: %r", failed_parts, len(fallback_units), source_unit)
    return translated_text or source_unit


def translate_utterances_google_sentencewise(
    utterances: List[Dict],
    src_lang: str,
    detected_lang: str,
    target_lang: str,
) -> Tuple[List[Dict], str]:
    src_code, tgt_code = resolve_translation_codes("google", src_lang, detected_lang, target_lang)
    translator = GoogleTranslatorWrapper(src_code, tgt_code)
    batch_size = int(os.environ.get("TRANSLATE_BATCH", "12"))

    expanded_units = []
    per_utterance_units: List[List[Dict[str, str]]] = [[] for _ in utterances]
    for utt_idx, utt in enumerate(utterances):
        source_units = split_source_for_nllb(str(utt.get("text", "")))
        for unit_idx, source_unit in enumerate(source_units):
            expanded_units.append((utt_idx, unit_idx, source_unit))

    translated_unit_count = 0
    for i in range(0, len(expanded_units), batch_size):
        chunk = expanded_units[i:i + batch_size]
        for (utt_idx, _unit_idx, source_unit) in chunk:
            translated_text = translate_google_unit_with_fallback(translator, source_unit, src_code, tgt_code)
            per_utterance_units[utt_idx].append({
                "source": source_unit,
                "text_translated": translated_text,
            })
        translated_unit_count += len(chunk)
        LOG.info("Translation units %d/%d (google sentencewise)", translated_unit_count, len(expanded_units))

    translated = []
    for utt, unit_translations in zip(utterances, per_utterance_units):
        item = dict(utt)
        translated_parts = [unit["text_translated"] for unit in unit_translations]
        item["text_translated"] = join_translated_units(translated_parts)
        item["translation_units"] = unit_translations
        translated.append(item)

    return translated, src_code


def translate_utterances(utterances: List[Dict], src_lang: str, detected_lang: str, target_lang: str, 
                         translation_method: str = "local", nllb_src_code: str = None, 
                         nllb_tgt_code: str = None) -> Tuple[List[Dict], str]:
    """
    Translate utterances using the specified method (local NLLB or Google Translate).
    
    Args:
        utterances: List of utterance dictionaries
        src_lang: Source language code
        detected_lang: Detected language from Whisper
        target_lang: Target language code
        translation_method: 'local' for NLLB, 'google' for Google Translate
        nllb_src_code: NLLB source language code (for local method)
        nllb_tgt_code: NLLB target language code (for local method)
    
    Returns:
        Tuple of (translated utterances, source code used)
    """
    if translation_method == "local":
        return translate_utterances_local_sentencewise(
            utterances,
            src_lang,
            detected_lang,
            target_lang,
            nllb_src_code,
            nllb_tgt_code,
        )
    if translation_method == "google":
        return translate_utterances_google_sentencewise(
            utterances,
            src_lang,
            detected_lang,
            target_lang,
        )

    src_code, tgt_code = resolve_translation_codes(
        translation_method,
        src_lang,
        detected_lang,
        target_lang,
        nllb_src_code,
        nllb_tgt_code,
    )
    translator = get_translator(translation_method, src_code, tgt_code)
    
    batch_size = int(os.environ.get("TRANSLATE_BATCH", "12"))
    translated = []
    for i in range(0, len(utterances), batch_size):
        chunk = utterances[i:i + batch_size]
        texts = [u["text"] for u in chunk]
        t_texts = translator.translate_batch(texts)
        for u, t in zip(chunk, t_texts):
            item = dict(u)
            item["text_translated"] = t
            translated.append(item)
        LOG.info("Translation %d/%d (%s)", min(i + batch_size, len(utterances)), len(utterances), translation_method)
    
    if translation_method == "local" and torch_cuda_usable():
        try:
            torch = _get_torch_module()
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    return translated, src_code


def target_chars_per_second(target_lang: str) -> float:
    configured = os.environ.get("LLM_CHARS_PER_SECOND", "").strip()
    if configured:
        return float(configured)
    if target_lang in CJK_LANGS:
        return 5.0
    if target_lang == "ar":
        return 9.0
    return 12.0


def text_budget_for_utterance(utt: Dict, target_lang: str) -> int:
    duration = max(0.1, float(utt.get("end", 0.0)) - float(utt.get("start", 0.0)))
    return max(24, int(round(duration * target_chars_per_second(target_lang))))


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"response is not a JSON object: {text[:120]!r}")


class OllamaAdapter:
    """Local LLM adapter for dubbing-length text reduction."""

    def __init__(self, model: str, target_lang: str):
        self.model = model
        self.target_lang = target_lang
        self.target_language_name = language_name(target_lang)
        raw_timeout = float(os.environ.get("LLM_TIMEOUT", "0"))
        self.timeout = None if raw_timeout <= 0 else raw_timeout
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
        self.num_predict = int(os.environ.get("LLM_NUM_PREDICT", "256"))

    def _generate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096,
                "num_predict": self.num_predict,
            },
        }
        if os.environ.get("AUTODUB_NO_GPU", "0") == "1":
            payload["options"]["num_gpu"] = 0
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return extract_json_object(data.get("response", ""))

    def adapt(self, source_text: str, machine_translation: str, budget: int, min_chars: int = 0) -> Tuple[str, Dict[str, Any]]:
        preferred_line = ""
        if min_chars > 0:
            preferred_line = f"\nPreferred length: {min_chars}-{budget} characters. Use the available space to keep details; do not over-summarize."
        prompt = f"""/no_think
You are a dubbing adaptation editor.
Write natural spoken {self.target_language_name} (language code: {self.target_lang}).
Shorten only as much as needed to fit a dubbing slot.
Preserve the core meaning, names, numbers, technical terms, and claims.
Do not add facts, explanations, notes, quotes, Markdown, chain-of-thought, or alternative versions.
Return only valid JSON with this exact shape:
{{"text":"...", "fits":true}}

Hard maximum length: {budget} characters.{preferred_line}
Source text:
{source_text}

        Machine translation:
{machine_translation}
"""
        best = sanitize_tts_text(machine_translation)
        def candidate_score(value: str) -> float:
            length = len(value or "")
            too_long = max(0, length - budget)
            too_short = max(0, min_chars - length) if min_chars > 0 else 0
            # A small overflow is usually less harmful than an over-compressed
            # dub line, because over-compression creates audible dead air.
            return (too_long * 1.5) + (too_short * 6.0) + (abs(budget - length) * 0.02)

        best_score = candidate_score(best)
        attempts = []
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                if len(best) > budget:
                    reason = f"too long: {len(best)} characters, hard limit {budget}"
                    instruction = "Remove filler, repetition, and secondary detail while preserving every essential meaning unit."
                elif min_chars > 0 and len(best) < min_chars:
                    reason = f"too short: {len(best)} characters, preferred minimum {min_chars}"
                    instruction = "Restore omitted details from the source or machine translation while staying under the hard limit."
                else:
                    reason = "not suitable for the dubbing length target"
                    instruction = "Rewrite it to fit the requested length range."
                prompt = f"""/no_think
The previous candidate is invalid because it is {reason}.
Rewrite it as natural spoken {self.target_language_name}.
Hard maximum: {budget} characters.
{f"Preferred length: {min_chars}-{budget} characters." if min_chars > 0 else ""}
{instruction}
Do not add new information.
Return only valid JSON: {{"text":"...", "fits":true}}

Source text:
{source_text}

Machine translation:
{machine_translation}

Text:
{best}
"""
            try:
                parsed = self._generate(prompt)
                candidate = sanitize_tts_text(str(parsed.get("text", "")).strip())
                if candidate:
                    score = candidate_score(candidate)
                    if score < best_score:
                        best = candidate
                        best_score = score
                attempts.append({
                    "attempt": attempt + 1,
                    "chars": len(candidate),
                    "fits": bool(candidate and len(candidate) <= budget and (min_chars <= 0 or len(candidate) >= min_chars)),
                })
                if candidate and len(candidate) <= budget and (min_chars <= 0 or len(candidate) >= min_chars):
                    return candidate, {
                        "enabled": True,
                        "model": self.model,
                        "target_chars": budget,
                        "preferred_min_chars": min_chars,
                        "fits_char_budget": True,
                        "attempts": attempts,
                    }
            except Exception as exc:
                attempts.append({"attempt": attempt + 1, "error": str(exc)})
                LOG.warning("LLM adaptation attempt failed: %s", exc)

        return best, {
            "enabled": True,
            "model": self.model,
            "target_chars": budget,
            "preferred_min_chars": min_chars,
            "fits_char_budget": len(best) <= budget,
            "fits_preferred_min": min_chars <= 0 or len(best) >= min_chars,
            "attempts": attempts,
        }


class GroqLLMAdapter(OllamaAdapter):
    """Groq-hosted LLM adapter using the OpenAI-compatible chat completions endpoint."""

    def _generate(self, prompt: str) -> Dict[str, Any]:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for --llm-provider groq")
        prompt = re.sub(r"^/no_think\s*", "", prompt.strip())
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise dubbing adaptation editor. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max(self.num_predict, 1024 if self.model == "openai/gpt-oss-120b" else 512),
            "stream": False,
        }
        if self.model == "openai/gpt-oss-120b":
            # Keep reasoning available for the adaptation and segmentation work,
            # but minimize its token use on Groq's free tier.
            payload["reasoning_effort"] = "low"
            payload["reasoning_format"] = "hidden"
            payload["response_format"] = {"type": "json_object"}
        response = cloud_request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            service="Groq LLM",
            timeout=self.timeout,
            max_retries=self.max_retries + 1,
            rate_limit_mode=os.environ.get("GROQ_RATE_LIMIT", "wait").strip().lower(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Groq LLM response did not include choices: {data}")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        return extract_json_object(content)


def adapt_utterances_for_dubbing(translated: List[Dict], target_lang: str) -> List[Dict]:
    mode = os.environ.get("LLM_ADAPT", "auto").strip().lower()
    if mode == "never":
        return translated

    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "groq":
        model = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b").strip()
        adapter = GroqLLMAdapter(model, target_lang)
    else:
        model = os.environ.get("LLM_MODEL", "qwen3:8b-q4_K_M").strip()
        adapter = OllamaAdapter(model, target_lang)
    speaker_rates = speaker_source_char_rates(translated)
    adapted = []
    for idx, utt in enumerate(translated, start=1):
        machine_translation = translation_text(utt)
        item = dict(utt)
        if not machine_translation:
            adapted.append(item)
            continue
        budget = text_budget_for_utterance_with_rates(utt, target_lang, speaker_rates)
        min_chars = minimum_adapted_chars(machine_translation, budget)
        should_adapt = mode == "always" or len(machine_translation) > budget
        if not should_adapt:
            item["text_machine_translated"] = machine_translation
            item["llm_adaptation"] = {
                "enabled": False,
                "reason": "within_char_budget",
                "target_chars": budget,
                "preferred_min_chars": min_chars,
                "speaker_source_chars_per_second": speaker_rates.get(str(utt.get("speaker", ""))),
                "chars": len(machine_translation),
            }
            adapted.append(item)
            continue
        new_text, meta = adapter.adapt(str(utt.get("text", "")), machine_translation, budget, min_chars)
        item["text_machine_translated"] = machine_translation
        item["text_translated"] = new_text
        meta["speaker_source_chars_per_second"] = speaker_rates.get(str(utt.get("speaker", "")))
        item["llm_adaptation"] = meta
        adapted.append(item)
        LOG.info(
            "LLM adaptation %d/%d: %d -> %d chars (budget=%d preferred_min=%d fits=%s)",
            idx, len(translated), len(machine_translation), len(new_text), budget, min_chars, meta.get("fits_char_budget"),
        )
    return adapted




def sanitize_tts_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace("…", "...")
    text = re.sub(r"\.\s*\.\s*\.+", "...", text)
    text = re.sub(r"\.{3,}", ", ", text)
    text = re.sub(r"\.{2}", ", ", text)
    text = re.sub(r"\s*([,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"(^|[\s(])-\s+", r"\1", text)
    text = re.sub(r"\s+[–—-]\s+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"(\.\s*){2,}", ". ", text)
    text = re.sub(r"([!?])\s*\.", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(" ,")


def strip_terminal_tts_punctuation(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"[\s\.,;:!?…]+$", "", text)
    text = re.sub(r"[\s–—-]+$", "", text)
    return text.strip()


def split_text_for_tts(text: str, max_chars: Optional[int] = None) -> List[str]:
    text = sanitize_tts_text(text)
    if not text:
        return []
    limit = max_chars or int(os.environ.get("XTTS_MAX_CHARS", "180"))
    if len(text) <= limit:
        return [text]

    sentence_chunks: List[str] = []
    parts = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in ".!?;:,—–-" or ch == "\n":
            parts.append("".join(current).strip())
            current = []
    if current:
        parts.append("".join(current).strip())
    parts = [p for p in parts if p]
    if not parts:
        parts = [text]

    for part in parts:
        if len(part) <= limit:
            sentence_chunks.append(part)
            continue
        words = part.split()
        buf = []
        buf_len = 0
        for word in words:
            extra = len(word) + (1 if buf else 0)
            if buf and buf_len + extra > limit:
                sentence_chunks.append(" ".join(buf))
                buf = [word]
                buf_len = len(word)
            else:
                buf.append(word)
                buf_len += extra
        if buf:
            sentence_chunks.append(" ".join(buf))

    merged: List[str] = []
    for chunk in sentence_chunks:
        if merged and len(merged[-1]) + 1 + len(chunk) <= limit:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def passthrough_tts_audio(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    return wav, (len(wav) / sr if sr else 0.0)


def prompt_before_dubbing_if_translation_was_just_created(translated_json: Path) -> bool:
    if not sys.stdin.isatty():
        LOG.warning("A new translation JSON was created, but no interactive terminal is available. Continuing automatically: %s", translated_json)
        return True

    print() 
    print("A new translation JSON has been created:")
    print(f"  {translated_json}")
    print("You can stop now, review or edit this JSON manually, and rerun the script later.")
    while True:
        answer = input("Continue with dubbing now? [Y/n]: ").strip().lower()
        if answer in ('', 'y', 'yes'):
            return True
        if answer in ('n', 'no'):
            LOG.info("Stopping after translation so the JSON can be reviewed manually: %s", translated_json)
            return False
        print("Please answer 'y' or 'n'.")


class XTTSCloner:
    def __init__(self, target_lang: str):
        if target_lang not in XTTS_LANG_MAP:
            raise RuntimeError(f"The target language '{target_lang}' is not directly supported by XTTS.")
        self.tts_lang = XTTS_LANG_MAP[target_lang]
        preferred = detect_torch_device()
        self.device = preferred
        TTS_API = _get_tts_api()
        try:
            LOG.info("Loading XTTS v2 on %s", self.device)
            self.api = TTS_API("tts_models/multilingual/multi-dataset/xtts_v2")
            self.api.to(self.device)
        except Exception as exc:
            LOG.exception("XTTS on %s failed, retrying on CPU: %s", self.device, exc)
            self.device = "cpu"
            self.api = TTS_API("tts_models/multilingual/multi-dataset/xtts_v2")
            self.api.to("cpu")
        self.model = self.api.synthesizer.tts_model
        self.latents = {}
        env_limit = int(os.environ.get("XTTS_MAX_CHARS", "180"))
        model_limit = None
        try:
            model_limit = int(self.model.tokenizer.char_limits[self.tts_lang])
        except Exception:
            model_limit = None
        margin = int(os.environ.get("XTTS_CHAR_LIMIT_MARGIN", "20"))
        if model_limit is not None:
            safe_limit = max(80, model_limit - margin)
            self.max_chars = min(env_limit, safe_limit)
            LOG.info("XTTS text chunk limit for %s: model=%d safe=%d configured=%d", self.tts_lang, model_limit, safe_limit, self.max_chars)
        else:
            self.max_chars = env_limit
            LOG.info("XTTS text chunk limit for %s: configured=%d (model limit unavailable)", self.tts_lang, self.max_chars)

    def expected_chunks(self, text: str) -> int:
        return len(split_text_for_tts(text, self.max_chars))

    def build_voice_cache(self, speaker_refs: Dict[str, List[str]]):
        for speaker, refs in speaker_refs.items():
            LOG.info("Computing voice embedding for %s using %d clips", speaker, len(refs))
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=refs)
            self.latents[speaker] = (gpt_cond_latent, speaker_embedding)

    def synthesize(self, text: str, speaker: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        requested_speaker = speaker
        if speaker not in self.latents:
            fallback = next(iter(self.latents), None)
            if fallback is None:
                raise RuntimeError("No speaker voice cache is available for XTTS")
            LOG.warning("Speaker '%s' has no reference voice cache. Falling back to '%s'.", speaker, fallback)
            speaker = fallback
        gpt_cond_latent, speaker_embedding = self.latents[speaker]
        tts_text = sanitize_tts_text(text)
        chunks = split_text_for_tts(tts_text, self.max_chars)
        if not chunks:
            return np.zeros(1, dtype=np.float32), 24000, {
                "tts_text": tts_text,
                "tts_chunks": 0,
                "tts_speaker_requested": requested_speaker,
                "tts_speaker_used": speaker,
            }
        rendered = []
        silence_ms = int(os.environ.get("XTTS_INTER_CHUNK_SILENCE_MS", "120"))
        silence = np.zeros(int(24000 * silence_ms / 1000.0), dtype=np.float32)
        for idx, chunk in enumerate(chunks, start=1):
            chunk = strip_terminal_tts_punctuation(chunk)
            if not chunk:
                continue
            try:
                out = self.model.inference(
                    chunk,
                    self.tts_lang,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=float(os.environ.get("XTTS_TEMPERATURE", "0.65")),
                    repetition_penalty=float(os.environ.get("XTTS_REPETITION_PENALTY", "2.0")),
                    speed=float(os.environ.get("XTTS_SPEED", "1.0")),
                    enable_text_splitting=False,
                )
            except Exception as exc:
                raise RuntimeError(f"XTTS inference failed for chunk {idx}/{len(chunks)}: {exc}") from exc
            wav = np.asarray(out["wav"], dtype=np.float32)
            if wav.size == 0:
                continue
            rendered.append(wav)
            if idx < len(chunks) and silence.size:
                rendered.append(silence)
        if not rendered:
            return np.zeros(1, dtype=np.float32), 24000, {
                "tts_text": tts_text,
                "tts_chunks": len(chunks),
                "tts_speaker_requested": requested_speaker,
                "tts_speaker_used": speaker,
            }
        wav = np.concatenate(rendered).astype(np.float32, copy=False)
        
        # Trim excessive silence that XTTS can append around generated chunks.
        try:
            wav_2d = wav[np.newaxis, :]
            trimmed_wav, _ = librosa.effects.trim(wav_2d, top_db=30)
            wav = trimmed_wav[0]
            LOG.info("Trimmed silence from TTS chunk. New length: %.2f", len(wav) / 24000)
        except Exception as e:
            LOG.warning("Audio trimming failed: %s", e)
        
        final_wav, final_duration = passthrough_tts_audio(wav, 24000)
        return final_wav, 24000, {
            "tts_text": tts_text,
            "tts_chunks": len(chunks),
            "tts_duration": final_duration,
            "tts_speaker_requested": requested_speaker,
            "tts_speaker_used": speaker,
        }


class KokoroTTS:
    def __init__(self, target_lang: str, speaker_profiles: Dict[str, Dict[str, Any]]):
        if target_lang not in KOKORO_LANG_MAP:
            supported = ", ".join(sorted(KOKORO_LANG_MAP))
            raise RuntimeError(f"Target language '{target_lang}' is not supported by Kokoro TTS. Supported targets: {supported}")
        self.target_lang = target_lang
        self.lang_code = KOKORO_LANG_MAP[target_lang]
        self.speaker_profiles = speaker_profiles
        self.max_chars = int(os.environ.get("TTS_MAX_CHARS", "5000"))
        self.speed = float(os.environ.get("TTS_SPEED", "1.0"))
        KPipeline = _get_kokoro_pipeline()
        self.device = detect_torch_device()
        LOG.info(
            "Loading Kokoro TTS: target_lang=%s lang_code=%s device=%s",
            target_lang, self.lang_code, self.device,
        )
        self.pipeline = KPipeline(lang_code=self.lang_code, device=self.device)

    def expected_chunks(self, text: str) -> int:
        return 1

    def synthesize(self, text: str, speaker: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        tts_text = sanitize_tts_text(text)
        profile = self.speaker_profiles.get(speaker) or {}
        voice = profile.get("tts_voice")
        if not voice:
            _, catalog = kokoro_voice_catalog_for_target(self.target_lang)
            female = [v for v in catalog if v.get("gender") == "Female"]
            voice = (female or catalog)[0]["name"]
        speed = float(profile.get("tts_speed", self.speed) or self.speed)
        chunks = []
        try:
            generator = self.pipeline(tts_text, voice=voice, speed=speed)
        except TypeError:
            generator = self.pipeline(tts_text, voice=voice)
        for item in generator:
            audio = None
            if hasattr(item, "audio"):
                audio = item.audio
            elif isinstance(item, tuple):
                audio = item[-1]
            elif isinstance(item, dict):
                audio = item.get("audio") or item.get("wav")
            else:
                audio = item
            if audio is None:
                continue
            try:
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                wav = np.asarray(audio, dtype=np.float32).reshape(-1)
            except Exception as exc:
                raise RuntimeError(f"Could not convert Kokoro audio output for speaker {speaker}: {exc}") from exc
            if wav.size:
                chunks.append(wav)
        if not chunks:
            return np.zeros(1, dtype=np.float32), 24000, {
                "tts_text": tts_text,
                "tts_chunks": 0,
                "tts_speaker_requested": speaker,
                "tts_speaker_used": speaker,
                "tts_engine": "kokoro",
                "tts_voice": voice,
                "tts_speed": speed,
                "tts_pitch_shift_steps": 0.0,
            }
        wav = np.concatenate(chunks).astype(np.float32, copy=False)
        final_wav, final_duration = passthrough_tts_audio(wav, 24000)
        return final_wav, 24000, {
            "tts_text": tts_text,
            "tts_chunks": len(chunks),
            "tts_duration": final_duration,
            "tts_speaker_requested": speaker,
            "tts_speaker_used": speaker,
            "tts_engine": "kokoro",
            "tts_voice": voice,
            "tts_speed": speed,
            "tts_pitch_shift_steps": 0.0,
            "speaker_voice_class": profile.get("voice_class"),
            "speaker_pitch_hz": profile.get("pitch_hz"),
        }


def speed_to_edge_rate(speed: float) -> str:
    percent = int(round((float(speed) - 1.0) * 100.0))
    percent = max(-50, min(100, percent))
    return f"{percent:+d}%"


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("_")
    return value or "voice"


class EdgeTTS:
    def __init__(self, target_lang: str, speaker_profiles: Dict[str, Dict[str, Any]]):
        self.target_lang = target_lang
        self.locale, self.catalog = edge_voice_catalog_for_target(target_lang)
        self.speaker_profiles = speaker_profiles
        self.max_chars = int(os.environ.get("TTS_MAX_CHARS", "5000"))
        self.speed = float(os.environ.get("TTS_SPEED", "1.0"))
        self.pitch = os.environ.get("EDGE_TTS_PITCH", "+0Hz")
        self.volume = os.environ.get("EDGE_TTS_VOLUME", "+0%")
        self.connect_timeout = int(os.environ.get("EDGE_TTS_CONNECT_TIMEOUT", "20"))
        self.receive_timeout = int(os.environ.get("EDGE_TTS_RECEIVE_TIMEOUT", "120"))
        self.max_retries = max(0, int(os.environ.get("EDGE_TTS_MAX_RETRIES", "3")))
        self.retry_delay = max(0.0, float(os.environ.get("EDGE_TTS_RETRY_DELAY", "5")))
        _get_edge_tts_module()
        LOG.info("Using Microsoft Edge TTS: target_lang=%s locale=%s voices=%d", target_lang, self.locale, len(self.catalog))

    def expected_chunks(self, text: str) -> int:
        return len(split_text_for_tts(text, self.max_chars))

    async def _save_mp3(self, text: str, voice: str, rate: str, out_mp3: Path):
        edge_tts = _get_edge_tts_module()
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=self.volume,
            pitch=self.pitch,
            connect_timeout=self.connect_timeout,
            receive_timeout=self.receive_timeout,
        )
        await communicate.save(str(out_mp3))

    def _mp3_to_wav_array(self, mp3_path: Path, wav_path: Path) -> Tuple[np.ndarray, int]:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mp3_path),
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
            str(wav_path),
        ])
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        return np.asarray(wav, dtype=np.float32), int(sr)

    def synthesize(self, text: str, speaker: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        tts_text = sanitize_tts_text(text)
        profile = self.speaker_profiles.get(speaker) or {}
        voice = profile.get("tts_voice")
        if not voice:
            female = [v for v in self.catalog if v.get("gender") == "Female"]
            voice = (female or self.catalog)[0]["name"]
        speed = float(profile.get("tts_speed", self.speed) or self.speed)
        rate = speed_to_edge_rate(speed)
        chunks = split_text_for_tts(tts_text, self.max_chars)
        if not chunks:
            return np.zeros(1, dtype=np.float32), 24000, {
                "tts_text": tts_text,
                "tts_chunks": 0,
                "tts_speaker_requested": speaker,
                "tts_speaker_used": speaker,
                "tts_engine": "edge",
                "tts_voice": voice,
                "tts_speed": speed,
                "edge_rate": rate,
                "edge_pitch": self.pitch,
                "edge_volume": self.volume,
            }

        rendered = []
        with tempfile.TemporaryDirectory(prefix="edge_tts_", dir=os.environ.get("TMP_DIR", None)) as tmp:
            tmp_dir = Path(tmp)
            for idx, chunk in enumerate(chunks, start=1):
                chunk = strip_terminal_tts_punctuation(chunk)
                if not chunk:
                    continue
                mp3_path = tmp_dir / f"chunk_{idx:03d}.mp3"
                wav_path = tmp_dir / f"chunk_{idx:03d}.wav"
                last_exc = None
                for attempt in range(1, self.max_retries + 2):
                    try:
                        mp3_path.unlink(missing_ok=True)
                        wav_path.unlink(missing_ok=True)
                        asyncio.run(self._save_mp3(chunk, voice, rate, mp3_path))
                        wav, sr = self._mp3_to_wav_array(mp3_path, wav_path)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt > self.max_retries:
                            break
                        delay = self.retry_delay * attempt
                        LOG.warning(
                            "Microsoft Edge TTS chunk %d/%d failed on attempt %d/%d voice=%s: %s; retrying in %.1fs",
                            idx, len(chunks), attempt, self.max_retries + 1, voice, exc, delay,
                        )
                        time.sleep(delay)
                if last_exc is not None:
                    raise RuntimeError(f"Microsoft Edge TTS failed for chunk {idx}/{len(chunks)} voice={voice}: {last_exc}") from last_exc
                if sr != 24000:
                    raise RuntimeError(f"Unexpected Edge TTS sample rate after conversion: {sr}")
                if wav.size:
                    rendered.append(wav)
        if not rendered:
            return np.zeros(1, dtype=np.float32), 24000, {
                "tts_text": tts_text,
                "tts_chunks": len(chunks),
                "tts_speaker_requested": speaker,
                "tts_speaker_used": speaker,
                "tts_engine": "edge",
                "tts_voice": voice,
                "tts_speed": speed,
                "edge_rate": rate,
                "edge_pitch": self.pitch,
                "edge_volume": self.volume,
            }
        wav = np.concatenate(rendered).astype(np.float32, copy=False)
        final_wav, final_duration = passthrough_tts_audio(wav, 24000)
        return final_wav, 24000, {
            "tts_text": tts_text,
            "tts_chunks": len(chunks),
            "tts_duration": final_duration,
            "tts_speaker_requested": speaker,
            "tts_speaker_used": speaker,
            "tts_engine": "edge",
            "tts_voice": voice,
            "tts_speed": speed,
            "edge_rate": rate,
            "edge_pitch": self.pitch,
            "edge_volume": self.volume,
            "speaker_voice_class": profile.get("voice_class"),
            "speaker_pitch_hz": profile.get("pitch_hz"),
        }


def stretch_audio_to_duration(input_wav: Path, output_wav: Path, target_duration: float) -> Tuple[Path, float, Dict[str, Any]]:
    """Time-stretch audio only when the required tempo change stays within safe limits."""
    if target_duration <= 0:
        LOG.warning("Target duration is <= 0, skipping stretch for %s", input_wav.name)
        return input_wav, ffprobe_audio_duration(input_wav), {
            "stretched": False,
            "stretch_reason": "invalid_target_duration",
        }

    current_duration = ffprobe_audio_duration(input_wav)
    if current_duration <= 0:
        raise RuntimeError(f"Invalid audio duration for {input_wav}")

    ratio = current_duration / target_duration
    max_compress = float(os.environ.get("MAX_TTS_COMPRESS_RATIO", "1.15"))
    max_expand = float(os.environ.get("MAX_TTS_EXPAND_RATIO", "1.20"))

    if 0.95 <= ratio <= 1.05:
        output_wav.unlink(missing_ok=True)
        LOG.info("Duration difference negligible, using original TTS audio: %s", input_wav.name)
        return input_wav, current_duration, {
            "stretched": False,
            "stretch_reason": "near_target_duration",
            "stretch_ratio": ratio,
        }

    if ratio > max_compress:
        output_wav.unlink(missing_ok=True)
        LOG.warning(
            "Skipping unsafe TTS compression for %s: current=%.2fs target=%.2fs ratio=%.2f max=%.2f",
            input_wav.name, current_duration, target_duration, ratio, max_compress,
        )
        return input_wav, current_duration, {
            "stretched": False,
            "stretch_reason": "compression_ratio_too_high",
            "stretch_ratio": ratio,
        }

    if ratio < (1.0 / max_expand):
        output_wav.unlink(missing_ok=True)
        LOG.warning(
            "Skipping unsafe TTS expansion for %s: current=%.2fs target=%.2fs ratio=%.2f max_expand=%.2f",
            input_wav.name, current_duration, target_duration, ratio, max_expand,
        )
        return input_wav, current_duration, {
            "stretched": False,
            "stretch_reason": "expansion_ratio_too_high",
            "stretch_ratio": ratio,
        }

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_wav),
        "-filter:a", f"atempo={ratio:.4f}",
        "-ar", "24000",
        str(output_wav)
    ]
    run(cmd)
    stretched_duration = ffprobe_audio_duration(output_wav)
    LOG.info("Stretched %s from %.2fs to %.2fs (ratio=%.2f)", input_wav.name, current_duration, stretched_duration, ratio)
    return output_wav, stretched_duration, {
        "stretched": True,
        "stretch_reason": "within_safe_ratio",
        "stretch_ratio": ratio,
        "stretch_target_duration": target_duration,
    }


def save_tts_manifest(manifest_path: Path, config: Dict[str, Any], items: List[Dict]):
    save_json(manifest_path, {
        "schema_version": 2,
        "config": config,
        "items": items,
    })


def assemble_timeline(
    translated: List[Dict],
    cloner: XTTSCloner,
    total_duration: float,
    out_wav: Path,
    manifest_path: Path,
    tts_config: Dict[str, Any],
):
    segments_dir = out_wav.parent / f"tts_segments_{safe_filename(str(tts_config.get('tts_engine', 'tts')))}"
    segments_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    max_end = total_duration
    segment_config_hash = sha1_json({
        **tts_config,
        "xtts_split_limit": cloner.max_chars,
        "tts_pacing_version": 8,
    })

    existing_manifest = {}
    if manifest_path.exists():
        try:
            manifest_payload = load_json(manifest_path)
            manifest_items = normalized_payload_items(manifest_payload)
            for item in manifest_items:
                existing_manifest[int(item["index"])] = item
            LOG.info("Loaded existing TTS manifest: %d segments", len(existing_manifest))
        except Exception as exc:
            LOG.warning("Existing manifest is not readable and will be rebuilt: %s", exc)

    ordered_translated = sorted(translated, key=lambda x: float(x.get("start", 0.0)))
    next_utterance_starts = {
        id(utt): float(ordered_translated[pos + 1].get("start", 0.0))
        for pos, utt in enumerate(ordered_translated[:-1])
    }

    for idx, utt in enumerate(translated, start=1):
        text = translation_text(utt)
        if not text:
            continue
        seg_path = segments_dir / f"seg_{idx:05d}.wav"
        tts_text = sanitize_tts_text(text)
        text_sha1 = hashlib.sha1(tts_text.encode("utf-8")).hexdigest()
        expected_chunks = int(cloner.expected_chunks(tts_text)) if hasattr(cloner, "expected_chunks") else len(split_text_for_tts(tts_text, cloner.max_chars))
        utt_start = float(utt.get("start", 0.0))
        utt_end = float(utt.get("end", 0.0))
        target_duration = max(0.0, utt_end - utt_start)
        available_duration = target_duration
        next_start = next_utterance_starts.get(id(utt))
        if next_start is not None and next_start > utt_end:
            available_duration = max(target_duration, next_start - utt_start)
        previous = existing_manifest.get(idx)
        item = None
        if seg_path.exists() and seg_path.stat().st_size > 0:
            compatible = bool(
                previous
                and previous.get("text_sha1") == text_sha1
                and int(previous.get("tts_chunks", 0) or 0) == expected_chunks
                and int(previous.get("tts_split_limit", 0) or 0) == cloner.max_chars
                and int(previous.get("tts_pacing_version", 0) or 0) == 8
                and previous.get("tts_config_hash") == segment_config_hash
            )
            if compatible:
                try:
                    dur = ffprobe_audio_duration(seg_path)
                    item = {
                        **utt,
                        "index": idx,
                        "tts_path": str(seg_path),
                        "tts_sr": int(previous.get("tts_sr", 24000) or 24000),
                        "tts_duration": dur,
                        "tts_chunks": expected_chunks,
                        "tts_split_limit": cloner.max_chars,
                        "tts_pacing_version": 8,
                        "tts_config_hash": segment_config_hash,
                        "tts_text": tts_text,
                        "tts_speaker_requested": previous.get("tts_speaker_requested", utt["speaker"]),
                        "tts_speaker_used": previous.get("tts_speaker_used", utt["speaker"]),
                        "tts_engine": previous.get("tts_engine"),
                        "tts_voice": previous.get("tts_voice"),
                        "tts_speed": previous.get("tts_speed"),
                        "tts_pitch_shift_steps": previous.get("tts_pitch_shift_steps"),
                        "edge_rate": previous.get("edge_rate"),
                        "edge_pitch": previous.get("edge_pitch"),
                        "edge_volume": previous.get("edge_volume"),
                        "speaker_voice_class": previous.get("speaker_voice_class"),
                        "speaker_pitch_hz": previous.get("speaker_pitch_hz"),
                        "text_sha1": text_sha1,
                    }
                    LOG.info("TTS %d/%d already exists: reusing %s", idx, len(translated), seg_path.name)
                except Exception as exc:
                    LOG.warning("Existing TTS segment is invalid and will be regenerated (%s): %s", seg_path.name, exc)
                    try:
                        seg_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                LOG.info("TTS %d/%d exists but was created with an older or incompatible TTS policy: regenerating %s", idx, len(translated), seg_path.name)
                try:
                    seg_path.unlink(missing_ok=True)
                except Exception:
                    pass
        if item is None:
            wav, sr, synth_meta = cloner.synthesize(text, utt["speaker"])
            sf.write(seg_path, wav, sr, subtype="PCM_16")
            dur = len(wav) / sr
            item = {
                **utt,
                "index": idx,
                "tts_path": str(seg_path),
                "tts_sr": sr,
                "tts_duration": dur,
                "tts_chunks": int(synth_meta.get("tts_chunks", expected_chunks) or expected_chunks),
                "tts_split_limit": cloner.max_chars,
                "tts_pacing_version": 8,
                "tts_config_hash": segment_config_hash,
                "tts_text": synth_meta.get("tts_text", tts_text),
                "tts_speaker_requested": synth_meta.get("tts_speaker_requested", utt["speaker"]),
                "tts_speaker_used": synth_meta.get("tts_speaker_used", utt["speaker"]),
                "tts_engine": synth_meta.get("tts_engine"),
                "tts_voice": synth_meta.get("tts_voice"),
                "tts_speed": synth_meta.get("tts_speed"),
                "tts_pitch_shift_steps": synth_meta.get("tts_pitch_shift_steps"),
                "edge_rate": synth_meta.get("edge_rate"),
                "edge_pitch": synth_meta.get("edge_pitch"),
                "edge_volume": synth_meta.get("edge_volume"),
                "speaker_voice_class": synth_meta.get("speaker_voice_class"),
                "speaker_pitch_hz": synth_meta.get("speaker_pitch_hz"),
                "text_sha1": text_sha1,
            }
            if idx % 10 == 0 or idx == len(translated):
                LOG.info("Generated TTS %d/%d", idx, len(translated))
        
        stretched_seg_path = segments_dir / f"seg_{idx:05d}_stretched.wav"
        if target_duration > 0:
            current_tts_dur = item["tts_duration"]
            stretch_target_duration = target_duration
            if current_tts_dur > target_duration and available_duration > target_duration:
                stretch_target_duration = min(current_tts_dur, available_duration)
                if stretch_target_duration > target_duration + 0.05:
                    LOG.info(
                        "Using %.2fs of post-utterance silence for segment %d timing: nominal=%.2fs available=%.2fs tts=%.2fs",
                        stretch_target_duration - target_duration, idx, target_duration, available_duration, current_tts_dur,
                    )
            stretched_path, stretched_dur, stretch_meta = stretch_audio_to_duration(Path(item["tts_path"]), stretched_seg_path, stretch_target_duration)
            item = {
                **item,
                "tts_path": str(stretched_path),
                "tts_duration": stretched_dur,
                "tts_nominal_duration": target_duration,
                "tts_available_duration": available_duration,
                "tts_target_duration": stretch_target_duration,
                **stretch_meta,
            }
            if stretch_meta.get("stretched"):
                LOG.info("Using stretched TTS for segment %d: %.2fs -> %.2fs", idx, current_tts_dur, stretch_target_duration)
        else:
            item = {**item, "stretched": False, "stretch_reason": "no_target_duration"}
        
        rendered.append(item)
        max_end = max(max_end, utt["start"] + item["tts_duration"])
        save_tts_manifest(manifest_path, tts_config, rendered)

    save_tts_manifest(manifest_path, tts_config, rendered)

    final_sr = 24000
    total_samples = int(math.ceil(max_end * final_sr)) + final_sr
    tmp_mix = out_wav.parent / (out_wav.stem + ".mix.f32")
    mix = np.memmap(tmp_mix, dtype="float32", mode="w+", shape=(total_samples,))
    mix[:] = 0.0

    overlap_warnings = 0
    rendered_sorted = sorted(rendered, key=lambda x: x["start"])
    next_starts = [rendered_sorted[i + 1]["start"] if i + 1 < len(rendered_sorted) else None for i in range(len(rendered_sorted))]

    for item, next_start in zip(rendered_sorted, next_starts):
        audio, sr = sf.read(item["tts_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != final_sr:
            raise RuntimeError(f"Unexpected sample rate {sr}, expected {final_sr}")
        start_idx = max(0, int(round(item["start"] * final_sr)))
        end_idx = start_idx + len(audio)
        if next_start is not None and (item["start"] + len(audio) / final_sr) > next_start + 0.6:
            overlap_warnings += 1
            LOG.warning(
                "TTS segment extends beyond the next utterance: speaker=%s start=%.2f tts_dur=%.2f next_start=%.2f text=%r",
                item["speaker"], item["start"], len(audio) / final_sr, next_start, translation_text(item)[:120]
            )
        if end_idx > len(mix):
            extra = end_idx - len(mix)
            LOG.warning("Extending the mix by %d samples", extra)
            old = np.asarray(mix).copy()
            del mix
            new_len = end_idx + final_sr
            mix = np.memmap(tmp_mix, dtype="float32", mode="w+", shape=(new_len,))
            mix[:] = 0.0
            mix[:len(old)] = old
        mix[start_idx:end_idx] += audio

    peak = float(np.max(np.abs(mix))) if len(mix) else 1.0
    norm = 0.95 / peak if peak > 0.99 else 1.0
    LOG.info("Peak mix=%.4f norm=%.4f overlap_warnings=%d", peak, norm, overlap_warnings)
    sf.write(out_wav, np.asarray(mix) * norm, final_sr, subtype="PCM_16")
    del mix
    try:
        tmp_mix.unlink(missing_ok=True)
    except Exception:
        pass
    save_tts_manifest(manifest_path, tts_config, rendered_sorted)


def mux_video(original_video: Path, dubbed_wav: Path, output_video: Path):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    copy_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(original_video),
        "-i", str(dubbed_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", os.environ.get("AAC_BITRATE", "192k"),
        "-movflags", "+faststart",
        "-shortest",
        str(output_video),
    ]
    try:
        run(copy_cmd)
        return
    except subprocess.CalledProcessError as exc:
        LOG.warning("Stream-copy mux failed, retrying with H.264 video re-encoding: %s", exc)

    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(original_video),
        "-i", str(dubbed_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", os.environ.get("AAC_BITRATE", "192k"),
        "-movflags", "+faststart",
        "-shortest",
        str(output_video),
    ])


def normalize_transcript_payload(payload):
    if isinstance(payload, dict) and "items" in payload:
        detected_lang = payload.get("detected_whisper_lang", os.environ.get("SOURCE_LANG", "auto"))
        return payload["items"], detected_lang
    if isinstance(payload, list):
        return payload, os.environ.get("SOURCE_LANG", "auto")
    raise RuntimeError("Unrecognized transcript JSON format")


def translated_fingerprint(translated: List[Dict]) -> str:
    slim = [
        {
            "start": round(float(u.get("start", 0.0)), 3),
            "end": round(float(u.get("end", 0.0)), 3),
            "speaker": str(u.get("speaker", "")),
            "text": translation_text(u),
        }
        for u in translated
    ]
    return sha1_json(slim)


def print_tts_voice_preview_links(tts_engine: str):
    print()
    if tts_engine == "edge":
        print("Preview links:")
        print("  Microsoft Voice Gallery: https://speech.microsoft.com/portal/voicegallery")
        print("  Unofficial Edge TTS web preview: https://edge-tts.com/")
    elif tts_engine == "kokoro":
        print("Preview links:")
        print("  Kokoro demo: https://huggingface.co/spaces/hexgrad/Kokoro-TTS")
        print("  Kokoro voice list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md")


def list_tts_voices_action(tts_engine: str, target_lang: str):
    if tts_engine == "xtts":
        print("XTTS uses voice cloning from the input speaker reference clips and has no fixed voice catalog.")
        return
    locale, catalog = tts_voice_catalog(tts_engine, target_lang)
    print(f"TTS engine: {tts_engine}")
    print(f"Target language: {target_lang}")
    print(f"Resolved TTS locale: {locale}")
    print(f"Voices: {len(catalog)}")
    print()
    print(f"{'Voice':45} {'Gender':8} {'Locale':10} Friendly name")
    print(f"{'-' * 45} {'-' * 8} {'-' * 10} {'-' * 40}")
    for voice in catalog:
        print(
            f"{voice.get('name',''):45} "
            f"{voice.get('gender',''):8} "
            f"{voice.get('locale',''):10} "
            f"{voice.get('friendly_name','')}"
        )
    print_tts_voice_preview_links(tts_engine)


async def save_edge_sample(text: str, voice: str, rate: str, output_path: Path):
    edge_tts = _get_edge_tts_module()
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=os.environ.get("EDGE_TTS_VOLUME", "+0%"),
        pitch=os.environ.get("EDGE_TTS_PITCH", "+0Hz"),
        connect_timeout=int(os.environ.get("EDGE_TTS_CONNECT_TIMEOUT", "20")),
        receive_timeout=int(os.environ.get("EDGE_TTS_RECEIVE_TIMEOUT", "120")),
    )
    await communicate.save(str(output_path))


def sample_edge_voices(text: str, locale: str, catalog: List[Dict[str, Any]], out_dir: Path):
    rate = speed_to_edge_rate(float(os.environ.get("TTS_SPEED", "1.0")))
    for voice in catalog:
        name = voice["name"]
        gender = str(voice.get("gender", "unknown")).lower()
        filename = f"edge_{safe_filename(locale)}__{safe_filename(name)}__{gender}__rate_{rate.replace('+', 'plus').replace('-', 'minus').replace('%', 'pct')}.mp3"
        output_path = out_dir / filename
        asyncio.run(save_edge_sample(text, name, rate, output_path))
        print(output_path)


def sample_kokoro_voices(text: str, target_lang: str, locale: str, catalog: List[Dict[str, Any]], out_dir: Path):
    lang_code = KOKORO_LANG_MAP[target_lang]
    KPipeline = _get_kokoro_pipeline()
    pipeline = KPipeline(lang_code=lang_code, device=detect_torch_device())
    speed = float(os.environ.get("TTS_SPEED", "1.0"))
    for voice in catalog:
        name = voice["name"]
        gender = str(voice.get("gender", "unknown")).lower()
        filename = f"kokoro_{safe_filename(locale)}__{safe_filename(name)}__{gender}__speed_{speed:.3f}.wav"
        output_path = out_dir / filename
        chunks = []
        for item in pipeline(sanitize_tts_text(text), voice=name, speed=speed):
            audio = getattr(item, "audio", None)
            if audio is None and isinstance(item, tuple):
                audio = item[-1]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            wav = np.asarray(audio, dtype=np.float32).reshape(-1)
            if wav.size:
                chunks.append(wav)
        if not chunks:
            LOG.warning("Kokoro produced no sample audio for voice %s", name)
            continue
        sf.write(output_path, np.concatenate(chunks), 24000, subtype="PCM_16")
        print(output_path)


def sample_tts_voices_action(tts_engine: str, target_lang: str):
    if tts_engine == "xtts":
        raise RuntimeError("Voice sampling is not available for XTTS because it clones speakers from the input media.")
    locale, catalog = tts_voice_catalog(tts_engine, target_lang)
    manual_map = parse_tts_voice_map(os.environ.get("TTS_VOICE_MAP", ""))
    if manual_map:
        selected = set(manual_map.values())
        catalog = [voice for voice in catalog if voice["name"] in selected]
    if not catalog:
        raise RuntimeError("No TTS voices are available to sample.")
    text = os.environ.get("SAMPLE_TEXT", "").strip() or "This is a text-to-speech voice sample."
    out_env = os.environ.get("SAMPLE_OUTPUT_DIR", "").strip()
    if out_env:
        out_dir = Path(out_env).expanduser().resolve()
    else:
        out_dir = Path(os.environ["WORK_DIR"]) / "samples" / f"{tts_engine}_{safe_filename(locale)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(catalog)} sample files to: {out_dir}")
    if tts_engine == "edge":
        sample_edge_voices(text, locale, catalog, out_dir)
    else:
        sample_kokoro_voices(text, target_lang, locale, catalog, out_dir)
    print_tts_voice_preview_links(tts_engine)


def main():
    work_dir = Path(os.environ["WORK_DIR"])
    target_lang = os.environ["TARGET_LANG"].strip().lower()
    tts_engine = os.environ["TTS_ENGINE"].strip().lower()

    if os.environ.get("LIST_TTS_VOICES", "0") == "1":
        list_tts_voices_action(tts_engine, target_lang)
        return
    if os.environ.get("SAMPLE_TTS_VOICES", "0") == "1":
        sample_tts_voices_action(tts_engine, target_lang)
        return

    video = Path(os.environ["INPUT_FILE"]).expanduser().resolve()
    output_env = os.environ.get("OUTPUT_FILE", "").strip()
    source_lang = os.environ["SOURCE_LANG"].strip().lower()
    nllb_src_lang = os.environ.get("NLLB_SRC_LANG", "auto").strip()
    nllb_tgt_lang = os.environ.get("NLLB_TGT_LANG", "auto").strip()
    whisper_model = os.environ.get("WHISPER_MODEL", "medium").strip()
    num_speakers_raw = os.environ.get("NUM_SPEAKERS", "auto").strip().lower()
    num_speakers = None if num_speakers_raw == "auto" else int(num_speakers_raw)
    min_speakers_raw = os.environ.get("MIN_SPEAKERS", "").strip()
    max_speakers_raw = os.environ.get("MAX_SPEAKERS", "").strip()
    min_speakers = int(min_speakers_raw) if min_speakers_raw else None
    max_speakers = int(max_speakers_raw) if max_speakers_raw else None
    translation_method = os.environ["TRANSLATION_METHOD"].strip().lower()
    tts_engine = os.environ["TTS_ENGINE"].strip().lower()
    asr_backend = os.environ.get("ASR_BACKEND", "local").strip().lower()
    diarization_backend = os.environ.get("DIARIZATION_BACKEND", "local").strip().lower()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if diarization_backend == "local" and not hf_token:
        raise RuntimeError("HF_TOKEN is missing")

    if tts_engine == "xtts" and target_lang not in XTTS_LANG_MAP:
        supported = ", ".join(sorted(XTTS_LANG_MAP))
        raise RuntimeError(f"Target language '{target_lang}' is not supported by XTTS. Supported targets: {supported}")
    if tts_engine == "kokoro" and target_lang not in KOKORO_LANG_MAP:
        supported = ", ".join(sorted(KOKORO_LANG_MAP))
        raise RuntimeError(f"Target language '{target_lang}' is not supported by Kokoro TTS. Supported targets: {supported}")
    tts_locale, tts_catalog, _ = validate_tts_voice_options(tts_engine, target_lang, num_speakers_raw)

    stem = video.stem
    video_work = work_dir / stem
    video_work.mkdir(parents=True, exist_ok=True)

    mono_wav = video_work / f"{stem}.mono16k.wav"
    mono_meta_json = video_work / f"{stem}.mono16k.meta.json"
    dubbed_wav = video_work / f"{stem}.{target_lang}.{tts_engine}.wav"
    transcript_json = video_work / f"{stem}.transcript.json"
    diar_json = video_work / f"{stem}.diarization.json"
    utterances_json = video_work / f"{stem}.utterances.json"
    translated_json = video_work / f"{stem}.translated.{target_lang}.json"
    speaker_profiles_json = video_work / f"{stem}.speakers.{target_lang}.{tts_engine}.json"
    manifest_json = video_work / f"{stem}.{target_lang}.{tts_engine}.manifest.json"
    output_video = Path(output_env).expanduser().resolve() if output_env else video.parent / f"{stem}_{target_lang.upper()}_{tts_engine}_dub.mp4"

    LOG.info("=== Start %s ===", video.name)

    mono_config = {
        "schema_version": 2,
        "stage": "extract_audio",
        "input": file_fingerprint(video),
        "sample_rate": 16000,
        "channels": 1,
    }
    if mono_wav.exists() and mono_wav.stat().st_size > 0 and mono_meta_json.exists():
        mono_payload = load_json(mono_meta_json)
        mono_reusable = cache_config_matches(mono_payload, mono_config)
    else:
        mono_reusable = False
    if mono_reusable:
        LOG.info("Mono audio already extracted: %s", mono_wav.name)
    else:
        LOG.info("Extracting mono 16 kHz audio...")
        extract_audio(video, mono_wav)
        save_json(mono_meta_json, {"schema_version": 2, "config": mono_config})

    transcript_config = {
        "schema_version": 2,
        "stage": "transcript",
        "mono_audio": file_fingerprint(mono_wav),
        "asr_backend": asr_backend,
        "source_lang": source_lang,
        "whisper_model": whisper_model,
        "groq_whisper_model": os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3") if asr_backend == "groq" else None,
        "groq_prompt_sha1": hashlib.sha1(os.environ.get("GROQ_PROMPT", "").encode("utf-8")).hexdigest() if asr_backend == "groq" else None,
        "groq_chunk_seconds": float(os.environ.get("GROQ_CHUNK_SECONDS", "120")) if asr_backend == "groq" else None,
        "groq_overlap_seconds": float(os.environ.get("GROQ_OVERLAP_SECONDS", "1.0")) if asr_backend == "groq" else None,
        "assemblyai_speech_model": os.environ.get("ASSEMBLYAI_SPEECH_MODEL", "universal-3-5-pro,universal-2") if asr_backend == "assemblyai" else None,
        "asr_beam": int(os.environ.get("ASR_BEAM", "5")),
        "asr_vad": os.environ.get("ASR_VAD", "true").lower() == "true",
        "asr_compute_gpu": os.environ.get("ASR_COMPUTE_GPU", "int8_float16"),
        "asr_compute_cpu": os.environ.get("ASR_COMPUTE_CPU", "int8"),
    }
    cloud_diar_segments = None
    transcript_payload = load_json(transcript_json) if transcript_json.exists() and transcript_json.stat().st_size > 0 else None
    if transcript_payload is not None and cache_config_matches(transcript_payload, transcript_config):
        LOG.info("Transcript cache is compatible. Reusing it.")
        asr_segments, detected_lang = normalize_transcript_payload(transcript_payload)
    else:
        if asr_backend == "local":
            asr_segments, detected_lang = transcribe_audio(mono_wav, source_lang, whisper_model)
        elif asr_backend == "groq":
            asr_segments, detected_lang = transcribe_audio_groq(mono_wav, source_lang)
        elif asr_backend == "assemblyai":
            asr_segments, cloud_diar_segments, detected_lang = transcribe_and_diarize_assemblyai(
                mono_wav, source_lang, num_speakers, min_speakers, max_speakers
            )
        else:
            raise RuntimeError(f"Unsupported ASR backend: {asr_backend}")
        save_json(transcript_json, {
            "schema_version": 2,
            "config": transcript_config,
            "detected_whisper_lang": detected_lang,
            "items": serialize_transcript_segments(asr_segments),
        })

    diar_config = {
        "schema_version": 2,
        "stage": "diarization",
        "diarization_backend": diarization_backend,
        "mono_audio": file_fingerprint(mono_wav),
        "transcript_sha1": sha1_json(normalized_payload_items(load_json(transcript_json))) if diarization_backend == "assemblyai" else None,
        "num_speakers": num_speakers_raw,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "model": "pyannote/speaker-diarization-3.1" if diarization_backend == "local" else "assemblyai-speaker-labels",
    }
    diar_payload = load_json(diar_json) if diar_json.exists() and diar_json.stat().st_size > 0 else None
    if diar_payload is not None and cache_config_matches(diar_payload, diar_config):
        LOG.info("Diarization cache is compatible. Reusing it.")
        diar_segments = normalized_payload_items(diar_payload)
    else:
        if diarization_backend == "local":
            diar_segments = diarize_audio(mono_wav, hf_token, num_speakers, min_speakers, max_speakers)
        elif diarization_backend == "assemblyai":
            diar_segments = cloud_diar_segments if cloud_diar_segments is not None else transcript_segments_to_diarization(asr_segments)
            if not diar_segments:
                raise RuntimeError("AssemblyAI diarization produced no speaker segments.")
        else:
            raise RuntimeError(f"Unsupported diarization backend: {diarization_backend}")
        save_json(diar_json, {
            "schema_version": 2,
            "config": diar_config,
            "items": diar_segments,
        })

    utterance_config = {
        "schema_version": 2,
        "stage": "utterances",
        "utterance_boundary_version": UTTERANCE_BOUNDARY_VERSION,
        "transcript_sha1": sha1_json(normalized_payload_items(load_json(transcript_json))),
        "diarization_sha1": sha1_json(diar_segments),
        "utterance_max_gap": float(os.environ.get("UTTERANCE_MAX_GAP", "0.9")),
        "utterance_max_duration": float(os.environ.get("UTTERANCE_MAX_DURATION", "18.0")),
        "utterance_max_chars": int(os.environ.get("UTTERANCE_MAX_CHARS", "420")),
        "utterance_repair_max_gap": float(os.environ.get("UTTERANCE_REPAIR_MAX_GAP", "2.2")),
        "utterance_repair_max_duration": float(os.environ.get("UTTERANCE_REPAIR_MAX_DURATION", "24.0")),
        "utterance_repair_max_chars": int(os.environ.get("UTTERANCE_REPAIR_MAX_CHARS", "620")),
        "llm_segment": os.environ.get("LLM_SEGMENT", "auto"),
        "llm_segment_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_segment_model": os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b") if os.environ.get("LLM_PROVIDER", "ollama") == "groq" else os.environ.get("LLM_MODEL", "qwen3:8b-q4_K_M"),
        "llm_segment_prompt_version": LLM_SEGMENT_PROMPT_VERSION,
    }
    utterance_payload = load_json(utterances_json) if utterances_json.exists() and utterances_json.stat().st_size > 0 else None
    if utterance_payload is not None and cache_config_matches(utterance_payload, utterance_config):
        LOG.info("Utterance cache is compatible. Reusing it.")
        utterances = normalized_payload_items(utterance_payload)
    else:
        utterances = build_utterances(asr_segments, diar_segments)
        save_json(utterances_json, {
            "schema_version": 2,
            "config": utterance_config,
            "items": utterances,
        })

    used_src_code, used_tgt_code = resolve_translation_codes(
        translation_method,
        source_lang,
        detected_lang,
        target_lang,
        nllb_src_lang,
        nllb_tgt_lang,
    )
    translation_config = {
        "schema_version": 2,
        "stage": "translation",
        "translation_method": translation_method,
        "source_lang": source_lang,
        "detected_whisper_lang": detected_lang,
        "target_lang": target_lang,
        "used_src_code": used_src_code,
        "used_tgt_code": used_tgt_code,
        "utterances_sha1": utterances_fingerprint(utterances),
        "nllb_translation_unit_version": NLLB_TRANSLATION_UNIT_VERSION if translation_method == "local" else None,
        "google_translation_unit_version": GOOGLE_TRANSLATION_UNIT_VERSION if translation_method == "google" else None,
        "llm_adapt": os.environ.get("LLM_ADAPT", "auto"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_model": os.environ.get("LLM_MODEL", "qwen3:8b-q4_K_M"),
        "groq_llm_model": os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b"),
        "llm_prompt_version": LLM_PROMPT_VERSION,
        "llm_budget_version": LLM_BUDGET_VERSION,
        "llm_speaker_pacing_version": TTS_SPEAKER_PACING_VERSION,
        "llm_chars_per_second": os.environ.get("LLM_CHARS_PER_SECOND", ""),
        "llm_max_retries": int(os.environ.get("LLM_MAX_RETRIES", "3")),
        "llm_temperature": float(os.environ.get("LLM_TEMPERATURE", "0.1")),
        "llm_num_predict": int(os.environ.get("LLM_NUM_PREDICT", "256")),
    }

    translation_was_just_created = False
    translated_payload = load_json(translated_json) if translated_json.exists() and translated_json.stat().st_size > 0 else None
    if translated_payload is not None and cache_config_matches(translated_payload, translation_config):
        LOG.info("Translation cache is compatible. Reusing it.")
        translated = normalized_payload_items(translated_payload)
    else:
        translated, used_src_code = translate_utterances(
            utterances, source_lang, detected_lang, target_lang,
            translation_method=translation_method,
            nllb_src_code=nllb_src_lang,
            nllb_tgt_code=nllb_tgt_lang
        )
        translated = adapt_utterances_for_dubbing(translated, target_lang)
        save_json(translated_json, {
            "schema_version": 2,
            "config": translation_config,
            "detected_whisper_lang": detected_lang,
            "used_src_code": used_src_code,
            "used_tgt_code": used_tgt_code,
            "translation_method": translation_method,
            "target_tts_lang": target_lang,
            "items": translated,
        })
        translation_was_just_created = True
        LOG.info("Translation JSON created: %s (method: %s)", translated_json, translation_method)

    if os.environ.get("STOP_AFTER_TRANSLATION", "0") == "1":
        LOG.info("Stopping after translation as requested.")
        return

    if (
        translation_was_just_created
        and os.environ.get("REVIEW_TRANSLATION", "0") == "1"
        and not prompt_before_dubbing_if_translation_was_just_created(translated_json)
    ):
        LOG.info("Stopping before dubbing as requested.")
        return

    speaker_profiles = {}
    if tts_engine in {"kokoro", "edge"}:
        speaker_profile_config = {
            "schema_version": 2,
            "stage": "speaker_profiles",
            "tts_engine": tts_engine,
            "target_lang": target_lang,
            "tts_locale": tts_locale,
            "diarization_sha1": sha1_json(diar_segments),
            "utterances_sha1": utterances_fingerprint(utterances),
            "tts_voice_map": os.environ.get("TTS_VOICE_MAP", ""),
            "tts_voice_map_strict": os.environ.get("TTS_VOICE_MAP_STRICT", "0"),
            "tts_voice_female": os.environ.get("TTS_VOICE_FEMALE", ""),
            "tts_voice_male": os.environ.get("TTS_VOICE_MALE", ""),
            "tts_voice_child": os.environ.get("TTS_VOICE_CHILD", ""),
            "tts_profile_version": TTS_PROFILE_VERSION,
            "tts_speaker_pacing_version": TTS_SPEAKER_PACING_VERSION,
            "tts_voice_catalog_sha1": sha1_json(tts_catalog),
        }
        speaker_profile_payload = load_json(speaker_profiles_json) if speaker_profiles_json.exists() and speaker_profiles_json.stat().st_size > 0 else None
        if speaker_profile_payload is not None and cache_config_matches(speaker_profile_payload, speaker_profile_config):
            LOG.info("Speaker profile cache is compatible. Reusing it.")
            speaker_profiles = speaker_profile_payload["items"]
        else:
            speaker_profiles = build_speaker_profiles(mono_wav, diar_segments, target_lang, utterances, tts_engine)
            save_json(speaker_profiles_json, {
                "schema_version": 2,
                "config": speaker_profile_config,
                "items": speaker_profiles,
            })

    tts_config = {
        "schema_version": 2,
        "stage": "tts_mix",
        "tts_engine": tts_engine,
        "target_lang": target_lang,
        "tts_locale": tts_locale,
        "translated_sha1": translated_fingerprint(translated),
        "speaker_profiles_sha1": sha1_json(speaker_profiles) if speaker_profiles else "",
        "max_ref_clips": int(os.environ.get("MAX_REF_CLIPS", "3")),
        "xtts_max_chars": int(os.environ.get("XTTS_MAX_CHARS", "180")),
        "xtts_char_limit_margin": int(os.environ.get("XTTS_CHAR_LIMIT_MARGIN", "20")),
        "xtts_speed": float(os.environ.get("XTTS_SPEED", "1.0")),
        "xtts_temperature": float(os.environ.get("XTTS_TEMPERATURE", "0.65")),
        "xtts_repetition_penalty": float(os.environ.get("XTTS_REPETITION_PENALTY", "2.0")),
        "xtts_inter_chunk_silence_ms": int(os.environ.get("XTTS_INTER_CHUNK_SILENCE_MS", "120")),
        "tts_synthesis_version": TTS_SYNTHESIS_VERSION,
        "tts_speed": float(os.environ.get("TTS_SPEED", "1.0")),
        "tts_max_chars": int(os.environ.get("TTS_MAX_CHARS", "5000")),
        "edge_pitch": os.environ.get("EDGE_TTS_PITCH", "+0Hz"),
        "edge_volume": os.environ.get("EDGE_TTS_VOLUME", "+0%"),
        "edge_connect_timeout": int(os.environ.get("EDGE_TTS_CONNECT_TIMEOUT", "20")),
        "edge_receive_timeout": int(os.environ.get("EDGE_TTS_RECEIVE_TIMEOUT", "120")),
        "edge_max_retries": int(os.environ.get("EDGE_TTS_MAX_RETRIES", "3")),
        "edge_retry_delay": float(os.environ.get("EDGE_TTS_RETRY_DELAY", "5")),
        "max_tts_compress_ratio": float(os.environ.get("MAX_TTS_COMPRESS_RATIO", "1.15")),
        "max_tts_expand_ratio": float(os.environ.get("MAX_TTS_EXPAND_RATIO", "1.20")),
    }
    manifest_payload = load_json(manifest_json) if manifest_json.exists() and manifest_json.stat().st_size > 0 else None
    if dubbed_wav.exists() and dubbed_wav.stat().st_size > 0 and manifest_payload is not None and cache_config_matches(manifest_payload, tts_config):
        LOG.info("Dubbed audio cache is compatible. Reusing it: %s", dubbed_wav.name)
    else:
        if tts_engine == "xtts":
            refs = extract_reference_clips(mono_wav, diar_segments, video_work, max_clips=int(os.environ.get("MAX_REF_CLIPS", "3")))
            cloner = XTTSCloner(target_lang=target_lang)
            cloner.build_voice_cache(refs)
        elif tts_engine == "edge":
            cloner = EdgeTTS(target_lang=target_lang, speaker_profiles=speaker_profiles)
        else:
            cloner = KokoroTTS(target_lang=target_lang, speaker_profiles=speaker_profiles)
        total_duration = ffprobe_duration(video)
        assemble_timeline(translated, cloner, total_duration, dubbed_wav, manifest_json, tts_config)

    LOG.info("Final MP4 mux...")
    mux_video(video, dubbed_wav, output_video)
    LOG.info("Created: %s", output_video)
    LOG.info("=== End %s ===", video.name)
    LOG.info("Processing completed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        main()
        logging.shutdown()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)
    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        sys.exit(1)
PY
chmod +x "$PY_SCRIPT"

export SCRIPT_DIR WORK_DIR LOG_FILE INPUT_FILE OUTPUT_FILE
export SOURCE_LANG TARGET_LANG TRANSLATION_METHOD TTS_ENGINE NLLB_SRC_LANG NLLB_TGT_LANG
export ONLY_CLOUD ASR_BACKEND DIARIZATION_BACKEND
export WHISPER_MODEL GROQ_WHISPER_MODEL GROQ_PROMPT GROQ_CHUNK_SECONDS GROQ_OVERLAP_SECONDS GROQ_TIMEOUT GROQ_MAX_RETRIES GROQ_RATE_LIMIT
export ASSEMBLYAI_SPEECH_MODEL ASSEMBLYAI_POLL_INTERVAL ASSEMBLYAI_TIMEOUT
export GROQ_API_KEY ASSEMBLYAI_API_KEY DEEPGRAM_API_KEY GEMINI_API_KEY
export NUM_SPEAKERS MIN_SPEAKERS MAX_SPEAKERS ASR_BEAM ASR_VAD ASR_COMPUTE_GPU ASR_COMPUTE_CPU
export MAX_REF_CLIPS XTTS_MAX_CHARS XTTS_CHAR_LIMIT_MARGIN XTTS_SPEED XTTS_TEMPERATURE
export XTTS_REPETITION_PENALTY AAC_BITRATE XTTS_INTER_CHUNK_SILENCE_MS
export MAX_TTS_COMPRESS_RATIO MAX_TTS_EXPAND_RATIO LOG_LEVEL TRANSLATE_BATCH TRANSLATE_ON_GPU NO_GPU AUTODUB_NO_GPU
export UTTERANCE_MAX_GAP UTTERANCE_MAX_DURATION UTTERANCE_MAX_CHARS
export UTTERANCE_REPAIR_MAX_GAP UTTERANCE_REPAIR_MAX_DURATION UTTERANCE_REPAIR_MAX_CHARS
export REVIEW_TRANSLATION STOP_AFTER_TRANSLATION LLM_ADAPT LLM_SEGMENT LLM_PROVIDER LLM_MODEL GROQ_LLM_MODEL LLM_CHARS_PER_SECOND
export LLM_MAX_RETRIES LLM_TEMPERATURE LLM_TIMEOUT LLM_NUM_PREDICT
export TTS_LOCALE TTS_VOICE_MAP TTS_VOICE_MAP_STRICT TTS_VOICE_FEMALE TTS_VOICE_MALE TTS_VOICE_CHILD
export TTS_SPEED TTS_MAX_CHARS LIST_TTS_VOICES SAMPLE_TTS_VOICES SAMPLE_TEXT SAMPLE_OUTPUT_DIR
export EDGE_TTS_PITCH EDGE_TTS_VOLUME EDGE_TTS_CONNECT_TIMEOUT EDGE_TTS_RECEIVE_TIMEOUT EDGE_TTS_MAX_RETRIES EDGE_TTS_RETRY_DELAY
# Work around OpenMP duplicate-library failures in some ML dependency stacks.
export KMP_DUPLICATE_LIB_OK="TRUE"

info "Starting the local dubbing pipeline with checkpoint/resume support..."
python "$PY_SCRIPT"
info "Done. Full log: ${LOG_FILE}"
