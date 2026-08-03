### About the test video

- **Title**: "Dr. Richard Stallman: Free/Libre Software And Our Freedom: Our shield against many digital injustices"
- **Source**: https://nojs.us/
- **License**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
- **Duration**: ~3 minutes (ideal for quick testing)

To test with the sample video:

```bash
# Using Google Translate (recommended for quick testing)
TRANSLATION_METHOD=google \
SOURCE_LANG=en \
TARGET_LANG=it \
NUM_SPEAKERS=2 \
./autodub_local.sh
```

Or with local NLLB translation:

```bash
# Using local NLLB (offline, requires model download)
TRANSLATION_METHOD=local \
SOURCE_LANG=en \
TARGET_LANG=it \
NLLB_SRC_LANG=eng_Latn \
NLLB_TGT_LANG=ita_Latn \
NUM_SPEAKERS=2 \
./autodub_local.sh
```
