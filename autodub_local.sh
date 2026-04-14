#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

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

info() { echo "[$(date +'%F %T')] [INFO] $*"; }
warn() { echo "[$(date +'%F %T')] [WARN] $*"; }

auto_apt_install() {
  local missing=()
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
  sudo apt-get update
  sudo apt-get install -y ffmpeg git-lfs python3-venv python3-pip
}

python_imports_ok() {
  python - <<'PY' >/dev/null 2>&1
mods = [
    'torch', 'torchaudio', 'faster_whisper', 'pyannote.audio', 'transformers',
    'sentencepiece', 'accelerate', 'huggingface_hub', 'soundfile', 'numpy', 'tqdm', 'TTS', 'spacy'
]
for m in mods:
    __import__(m)
print('ok')
PY
}

info "Log: ${LOG_FILE}"
auto_apt_install ffmpeg git-lfs python3

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

if [[ -f "${SCRIPT_DIR}/.hf_token" && -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(<"${SCRIPT_DIR}/.hf_token")"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo
  echo "A Hugging Face READ token is required for local pyannote diarization."
  echo "Before continuing, accept the model terms once at:"
  echo "  1) https://huggingface.co/pyannote/speaker-diarization-community-1"
  echo "  2) https://huggingface.co/settings/tokens  (create a READ token)"
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

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating the local Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade pip setuptools wheel

GPU_HINT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_HINT=1
fi

if python_imports_ok; then
  info "Python packages are already available in the virtual environment. Reinstallation skipped."
else
  if [[ "$GPU_HINT" -eq 1 ]]; then
    info "NVIDIA GPU detected. Installing CUDA-enabled PyTorch and CUDA runtime packages for faster-whisper."
    pip install --index-url https://download.pytorch.org/whl/cu128 "torch<2.9" "torchaudio<2.9"
    pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
  else
    info "No NVIDIA GPU detected. Installing CPU-only PyTorch."
    pip install --index-url https://download.pytorch.org/whl/cpu "torch<2.9" "torchaudio<2.9"
  fi

  info "Installing or updating local Python packages..."
  pip install \
    "faster-whisper==1.2.1" \
    "pyannote-audio>=4.0.4" \
    "transformers>=4.57.5,<5.0" \
    "sentencepiece>=0.2.0" \
    "accelerate>=1.0.0" \
    "huggingface-hub>=0.34.0" \
    "coqui-tts==0.27.5" \
    "spacy>=3.8,<4" \
    "soundfile>=0.12.1" \
    "numpy>=1.26" \
    "tqdm>=4.66"
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
import math
import logging
import hashlib
import subprocess
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import soundfile as sf
import librosa
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as PyannotePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

LOG = logging.getLogger("autodub")

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

DEFAULT_INPUTS = [
    "AllaVoronkova-DigiunoSeccoACascata-Parte1.mp4",
    "AllaVoronkova-DigiunoSeccoACascata-Parte2.mp4",
]


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


def choose_existing_inputs(base_dir: Path) -> List[Path]:
    explicit = [base_dir / name for name in DEFAULT_INPUTS if (base_dir / name).exists()]
    if explicit:
        return explicit
    return sorted(base_dir.glob("*.mp4"))


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


def torch_cuda_usable(min_major: int = 7) -> bool:
    try:
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
    return "cuda" if torch_cuda_usable() else "cpu"


def preload_audio_dict(audio_path: Path) -> Dict[str, Any]:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    return {"waveform": waveform, "sample_rate": int(sr)}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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


def transcribe_audio(audio_path: Path, source_lang: str, model_name: str):
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


def diarize_audio(audio_path: Path, hf_token: str, num_speakers: int):
    # pyannote 4.x can use an internal audio decoder (torchcodec) that may fail in some environments;
    # by passing preloaded audio in memory, that code path is avoided entirely.
    audio_dict = preload_audio_dict(audio_path)
    preferred = detect_torch_device()
    attempts = [preferred] if preferred == "cpu" else ["cuda", "cpu"]
    last_exc = None
    for device in attempts:
        try:
            LOG.info("pyannote diarization: device=%s num_speakers=%s", device, num_speakers)
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token,
            )
            if device == "cuda":
                pipeline.to(torch.device("cuda"))
                diar_input = {"waveform": audio_dict["waveform"].to(torch.device("cuda")), "sample_rate": audio_dict["sample_rate"]}
            else:
                diar_input = audio_dict
            diar = pipeline(diar_input, num_speakers=num_speakers)
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
            "https://huggingface.co/pyannote/speaker-diarization-community-1 and then use a valid READ token."
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

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = " ".join(current["text"].split())
            utterances.append(current)
        current = None

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
                if current is None:
                    current = {"start": ws, "end": we, "speaker": spk, "text": text}
                else:
                    gap = ws - current["end"]
                    if spk == current["speaker"] and gap <= 1.2:
                        current["end"] = max(current["end"], we)
                        current["text"] += " " + text
                    else:
                        flush()
                        current = {"start": ws, "end": we, "speaker": spk, "text": text}
        else:
            text = seg_text(seg).strip()
            if not text:
                continue
            s0, s1 = seg_start(seg), seg_end(seg)
            spk = speaker_for_time((s0 + s1) / 2.0, diar_segments)
            if current is None:
                current = {"start": s0, "end": s1, "speaker": spk, "text": text}
            else:
                gap = s0 - current["end"]
                if spk == current["speaker"] and gap <= 1.2:
                    current["end"] = max(current["end"], s1)
                    current["text"] += " " + text
                else:
                    flush()
                    current = {"start": s0, "end": s1, "speaker": spk, "text": text}
    flush()
    LOG.info("Reconstructed utterances: %d", len(utterances))
    return utterances


def extract_reference_clips(audio_path: Path, diar_segments: List[Dict], work_dir: Path, max_clips: int = 3) -> Dict[str, List[str]]:
    refs_dir = work_dir / "reference_clips"
    refs_dir.mkdir(parents=True, exist_ok=True)
    by_spk = defaultdict(list)
    for seg in diar_segments:
        dur = seg["end"] - seg["start"]
        if dur >= 4.0:
            by_spk[seg["speaker"]].append(seg)
    result = {}
    for spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda x: (x["end"] - x["start"]), reverse=True)[:max_clips]
        clips = []
        for idx, seg in enumerate(segs, start=1):
            out = refs_dir / f"{spk}_{idx}.wav"
            if not out.exists() or out.stat().st_size == 0:
                start = seg["start"]
                dur = min(seg["end"] - seg["start"], 12.0)
                extract_audio_segment(audio_path, start, dur, out)
            clips.append(str(out))
        if clips:
            result[spk] = clips
    if not result:
        raise RuntimeError("Could not derive reference voice clips")
    LOG.info("Reference clips available for %d speakers", len(result))
    return result


class Translator:
    def __init__(self, src_code: str, tgt_code: str, model_name: str = "facebook/nllb-200-distilled-600M"):
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
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos,
                max_length=512,
                num_beams=4,
                repetition_penalty=1.05,
            )
        txt = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in txt]


def translate_utterances(utterances: List[Dict], src_lang: str, detected_lang: str, target_lang: str) -> Tuple[List[Dict], str]:
    src_code = src_lang if src_lang != "auto" else LANG_MAP.get(detected_lang)
    if src_code is None:
        raise RuntimeError(
            f"Detected source language '{detected_lang}' is not mapped to NLLB. "
            f"Set NLLB_SRC_LANG explicitly, for example rus_Cyrl."
        )
    translator = Translator(src_code=src_code, tgt_code=target_lang)
    batch_size = int(os.environ.get("TRANSLATE_BATCH", "12"))
    translated = []
    for i in range(0, len(utterances), batch_size):
        chunk = utterances[i:i + batch_size]
        texts = [u["text"] for u in chunk]
        t_texts = translator.translate_batch(texts)
        for u, t in zip(chunk, t_texts):
            item = dict(u)
            item["text_it"] = t
            translated.append(item)
        LOG.info("Translation %d/%d", min(i + batch_size, len(utterances)), len(utterances))
    if torch_cuda_usable():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return translated, src_code




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


def time_stretch_to_target(wav: np.ndarray, sr: int, target_duration: Optional[float]) -> Tuple[np.ndarray, float]:
    if target_duration is None or target_duration <= 0:
        return wav, len(wav) / sr if sr else 0.0
    current_duration = len(wav) / sr if sr else 0.0
    if current_duration <= 0.0:
        return wav, current_duration

    min_fill = float(os.environ.get("XTTS_TARGET_MIN_FILL", "0.92"))
    max_slowdown = float(os.environ.get("XTTS_MAX_SLOWDOWN", "2.2"))
    max_speedup = float(os.environ.get("XTTS_MAX_SPEEDUP", "1.15"))

    desired_duration = target_duration * min_fill
    stretch_factor = None

    if current_duration < desired_duration:
        stretch_factor = min(max_slowdown, desired_duration / current_duration)
    elif current_duration > target_duration * 1.08:
        stretch_factor = max(1.0 / max_speedup, desired_duration / current_duration)

    if stretch_factor is None or abs(stretch_factor - 1.0) < 0.04:
        return wav, current_duration

    rate = 1.0 / stretch_factor
    try:
        stretched = librosa.effects.time_stretch(wav.astype(np.float32, copy=False), rate=rate)
        stretched = np.asarray(stretched, dtype=np.float32)
        return stretched, len(stretched) / sr
    except Exception as exc:
        LOG.warning("Time-stretch failed; using the original XTTS audio: %s", exc)
        return wav, current_duration


class XTTSCloner:
    def __init__(self, target_lang: str):
        if target_lang not in XTTS_LANG_MAP:
            raise RuntimeError(f"The target language '{target_lang}' is not directly supported by XTTS.")
        self.tts_lang = XTTS_LANG_MAP[target_lang]
        preferred = detect_torch_device()
        self.device = preferred
        try:
            LOG.info("Loading XTTS v2 on %s", self.device)
            self.api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self.api.to(self.device)
        except Exception as exc:
            LOG.exception("XTTS on %s failed, retrying on CPU: %s", self.device, exc)
            self.device = "cpu"
            self.api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
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

    def build_voice_cache(self, speaker_refs: Dict[str, List[str]]):
        for speaker, refs in speaker_refs.items():
            LOG.info("Computing voice embedding for %s using %d clips", speaker, len(refs))
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=refs)
            self.latents[speaker] = (gpt_cond_latent, speaker_embedding)

    def synthesize(self, text: str, speaker: str, target_duration: Optional[float] = None) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        if speaker not in self.latents:
            raise RuntimeError(f"Speaker '{speaker}' is not present in the XTTS cache")
        gpt_cond_latent, speaker_embedding = self.latents[speaker]
        tts_text = sanitize_tts_text(text)
        chunks = split_text_for_tts(tts_text, self.max_chars)
        if not chunks:
            return np.zeros(1, dtype=np.float32), 24000, {"tts_text": tts_text, "tts_chunks": 0, "tts_target_duration": target_duration, "tts_stretch_factor": 1.0}
        rendered = []
        silence_ms = int(os.environ.get("XTTS_INTER_CHUNK_SILENCE_MS", "120"))
        silence = np.zeros(int(24000 * silence_ms / 1000.0), dtype=np.float32)
        for idx, chunk in enumerate(chunks, start=1):
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
            return np.zeros(1, dtype=np.float32), 24000, {"tts_text": tts_text, "tts_chunks": len(chunks), "tts_target_duration": target_duration, "tts_stretch_factor": 1.0}
        wav = np.concatenate(rendered).astype(np.float32, copy=False)
        original_duration = len(wav) / 24000.0
        stretched, stretched_duration = time_stretch_to_target(wav, 24000, target_duration)
        stretch_factor = (stretched_duration / original_duration) if original_duration > 0 else 1.0
        return stretched, 24000, {
            "tts_text": tts_text,
            "tts_chunks": len(chunks),
            "tts_target_duration": target_duration,
            "tts_stretch_factor": stretch_factor,
        }


def assemble_timeline(translated: List[Dict], cloner: XTTSCloner, total_duration: float, out_wav: Path, manifest_path: Path):
    segments_dir = out_wav.parent / "tts_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    max_end = total_duration

    existing_manifest = {}
    if manifest_path.exists():
        try:
            manifest_payload = load_json(manifest_path)
            manifest_items = manifest_payload.get("items", manifest_payload) if isinstance(manifest_payload, dict) else manifest_payload
            for item in manifest_items:
                existing_manifest[int(item["index"])] = item
            LOG.info("Loaded existing TTS manifest: %d segments", len(existing_manifest))
        except Exception as exc:
            LOG.warning("Existing manifest is not readable and will be rebuilt: %s", exc)

    for idx, utt in enumerate(translated, start=1):
        text = utt.get("text_it", "").strip()
        if not text:
            continue
        seg_path = segments_dir / f"seg_{idx:05d}.wav"
        tts_text = sanitize_tts_text(text)
        text_sha1 = hashlib.sha1(tts_text.encode("utf-8")).hexdigest()
        expected_chunks = len(split_text_for_tts(tts_text, cloner.max_chars))
        target_duration = max(0.0, float(utt.get("end", 0.0)) - float(utt.get("start", 0.0)))
        previous = existing_manifest.get(idx)
        item = None
        if seg_path.exists() and seg_path.stat().st_size > 0:
            compatible = bool(
                previous
                and previous.get("text_sha1") == text_sha1
                and int(previous.get("tts_chunks", 0) or 0) == expected_chunks
                and int(previous.get("tts_split_limit", 0) or 0) == cloner.max_chars
                and int(previous.get("tts_pacing_version", 0) or 0) == 2
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
                        "tts_pacing_version": 2,
                        "tts_target_duration": target_duration,
                        "tts_text": tts_text,
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
            wav, sr, synth_meta = cloner.synthesize(text, utt["speaker"], target_duration=target_duration)
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
                "tts_pacing_version": 2,
                "tts_target_duration": target_duration,
                "tts_stretch_factor": float(synth_meta.get("tts_stretch_factor", 1.0) or 1.0),
                "tts_text": synth_meta.get("tts_text", tts_text),
                "text_sha1": text_sha1,
            }
            if idx % 10 == 0 or idx == len(translated):
                LOG.info("Generated TTS %d/%d", idx, len(translated))
        rendered.append(item)
        max_end = max(max_end, utt["start"] + item["tts_duration"])
        if idx % 20 == 0 or idx == len(translated):
            save_json(manifest_path, rendered)

    save_json(manifest_path, rendered)

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
                item["speaker"], item["start"], len(audio) / final_sr, next_start, item["text_it"][:120]
            )
        if end_idx > len(mix):
            extra = end_idx - len(mix)
            LOG.warning("Extending the mix by %d samples", extra)
            old = np.asarray(mix)
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
    save_json(manifest_path, rendered_sorted)


def mux_video(original_video: Path, dubbed_wav: Path, output_video: Path):
    run([
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
    ])


def normalize_transcript_payload(payload):
    if isinstance(payload, dict) and "items" in payload:
        detected_lang = payload.get("detected_whisper_lang", os.environ.get("SOURCE_LANG", "ru"))
        return payload["items"], detected_lang
    if isinstance(payload, list):
        return payload, os.environ.get("SOURCE_LANG", "ru")
    raise RuntimeError("Unrecognized transcript JSON format")


def main():
    script_dir = Path(os.environ["SCRIPT_DIR"])
    work_dir = Path(os.environ["WORK_DIR"])
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing")

    source_lang = os.environ.get("SOURCE_LANG", "ru").strip()
    target_lang = os.environ.get("TARGET_LANG", "it").strip()
    nllb_src_lang = os.environ.get("NLLB_SRC_LANG", "auto").strip()
    nllb_tgt_lang = os.environ.get("NLLB_TGT_LANG", "ita_Latn").strip()
    whisper_model = os.environ.get("WHISPER_MODEL", "medium").strip()
    num_speakers = int(os.environ.get("NUM_SPEAKERS", "2"))

    inputs = choose_existing_inputs(script_dir)
    if not inputs:
        raise RuntimeError("No MP4 files were found in the script directory")

    LOG.info("Files to process: %s", ", ".join(p.name for p in inputs))
    for video in inputs:
        stem = video.stem
        video_work = work_dir / stem
        video_work.mkdir(parents=True, exist_ok=True)

        mono_wav = video_work / f"{stem}.mono16k.wav"
        dubbed_wav = video_work / f"{stem}.it.wav"
        transcript_json = video_work / f"{stem}.transcript.json"
        diar_json = video_work / f"{stem}.diarization.json"
        utterances_json = video_work / f"{stem}.utterances.json"
        translated_json = video_work / f"{stem}.translated.json"
        manifest_json = video_work / f"{stem}.manifest.json"
        output_video = script_dir / f"{stem}_{target_lang.upper()}_dub.mp4"

        LOG.info("=== Start %s ===", video.name)

        if output_video.exists() and output_video.stat().st_size > 0:
            LOG.info("Final output already exists, skipping video: %s", output_video.name)
            LOG.info("=== End %s ===", video.name)
            continue

        if mono_wav.exists() and mono_wav.stat().st_size > 0:
            LOG.info("Mono audio already extracted: %s", mono_wav.name)
        else:
            LOG.info("Extracting mono 16 kHz audio...")
            extract_audio(video, mono_wav)

        if transcript_json.exists() and transcript_json.stat().st_size > 0:
            LOG.info("Transcript already exists. Reusing it.")
            asr_segments, detected_lang = normalize_transcript_payload(load_json(transcript_json))
        else:
            asr_segments, detected_lang = transcribe_audio(mono_wav, source_lang, whisper_model)
            save_json(transcript_json, {
                "detected_whisper_lang": detected_lang,
                "items": [
                    {
                        "start": float(seg_start(s)),
                        "end": float(seg_end(s)),
                        "text": seg_text(s),
                        "words": [
                            {
                                "start": float((w.get("start", 0.0) if isinstance(w, dict) else getattr(w, "start", 0.0)) or 0.0),
                                "end": float((w.get("end", 0.0) if isinstance(w, dict) else getattr(w, "end", 0.0)) or 0.0),
                                "word": (w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")) or "",
                            }
                            for w in seg_words(s)
                        ],
                    }
                    for s in asr_segments
                ],
            })

        if diar_json.exists() and diar_json.stat().st_size > 0:
            LOG.info("Diarization already exists. Reusing it.")
            diar_segments = load_json(diar_json)
        else:
            diar_segments = diarize_audio(mono_wav, hf_token, num_speakers)
            save_json(diar_json, diar_segments)

        if utterances_json.exists() and utterances_json.stat().st_size > 0:
            LOG.info("Utterances already exist. Reusing them.")
            utterances = load_json(utterances_json)
        else:
            utterances = build_utterances(asr_segments, diar_segments)
            save_json(utterances_json, utterances)

        if translated_json.exists() and translated_json.stat().st_size > 0:
            LOG.info("Translation already exists. Reusing it.")
            translated_payload = load_json(translated_json)
            translated = translated_payload["items"]
            detected_lang = translated_payload.get("detected_whisper_lang", detected_lang)
            used_src_code = translated_payload.get("used_nllb_src", nllb_src_lang)
            LOG.info("Loaded translation: src=%s tgt=%s items=%d", used_src_code, translated_payload.get("used_nllb_tgt", nllb_tgt_lang), len(translated))
        else:
            translated, used_src_code = translate_utterances(utterances, nllb_src_lang, detected_lang, nllb_tgt_lang)
            save_json(translated_json, {
                "detected_whisper_lang": detected_lang,
                "used_nllb_src": used_src_code,
                "used_nllb_tgt": nllb_tgt_lang,
                "target_tts_lang": target_lang,
                "items": translated,
            })

        if dubbed_wav.exists() and dubbed_wav.stat().st_size > 0:
            LOG.info("Dubbed audio already exists. Skipping TTS/mix: %s", dubbed_wav.name)
        else:
            refs = extract_reference_clips(mono_wav, diar_segments, video_work, max_clips=int(os.environ.get("MAX_REF_CLIPS", "3")))
            cloner = XTTSCloner(target_lang=target_lang)
            cloner.build_voice_cache(refs)
            total_duration = ffprobe_duration(video)
            assemble_timeline(translated, cloner, total_duration, dubbed_wav, manifest_json)

        LOG.info("Final MP4 mux...")
        mux_video(video, dubbed_wav, output_video)
        LOG.info("Created: %s", output_video)
        LOG.info("=== End %s ===", video.name)

    LOG.info("All processing completed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        main()
    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        sys.exit(1)
PY
chmod +x "$PY_SCRIPT"

export SCRIPT_DIR WORK_DIR LOG_FILE
export SOURCE_LANG="${SOURCE_LANG:-ru}"
export TARGET_LANG="${TARGET_LANG:-it}"
export NLLB_SRC_LANG="${NLLB_SRC_LANG:-rus_Cyrl}"
export NLLB_TGT_LANG="${NLLB_TGT_LANG:-ita_Latn}"
export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export NUM_SPEAKERS="${NUM_SPEAKERS:-2}"
export ASR_BEAM="${ASR_BEAM:-5}"
export ASR_VAD="${ASR_VAD:-true}"
export ASR_COMPUTE_GPU="${ASR_COMPUTE_GPU:-int8_float16}"
export ASR_COMPUTE_CPU="${ASR_COMPUTE_CPU:-int8}"
export MAX_REF_CLIPS="${MAX_REF_CLIPS:-3}"
export XTTS_MAX_CHARS="${XTTS_MAX_CHARS:-180}"
export XTTS_CHAR_LIMIT_MARGIN="${XTTS_CHAR_LIMIT_MARGIN:-20}"
export XTTS_SPEED="${XTTS_SPEED:-1.0}"
export XTTS_TEMPERATURE="${XTTS_TEMPERATURE:-0.65}"
export XTTS_REPETITION_PENALTY="${XTTS_REPETITION_PENALTY:-2.0}"
export AAC_BITRATE="${AAC_BITRATE:-192k}"
export XTTS_INTER_CHUNK_SILENCE_MS="${XTTS_INTER_CHUNK_SILENCE_MS:-120}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

info "Starting the local dubbing pipeline with checkpoint/resume support..."
python "$PY_SCRIPT"
info "Done. Full log: ${LOG_FILE}"
