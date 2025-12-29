# Datasets

Name | Sample rate | Size | Speakers | Description
-----|-------------|------|----------|------------
[VAIS-1000](https://huggingface.co/datasets/doof-ferb/vais1000) | 16kHz | 1.6hr, 1k samples | 1 | Transcription has no formatting.
[ViVoice](https://huggingface.co/datasets/capleaf/viVoice) | 24kHz | 1k hrs, 888k samples | Diverse (186 YouTube channels) | Sourced from YouTube

## ViVoice

- Based on the current tokenization scheme, 20.5 (+-3.8) text tokens per speech second -> 10 seconds ~ 200 text tokens.
- Using 24kHz audio
  - stride = 128 (5.3ms frame) -> 188 audio tokens per speech second -> 10 seconds ~ 1880 audio tokens
  - stride = 256 (10.7ms frame) -> 93.8 audio tokens per speech second -> 10 seconds ~ 938 audio tokens
  - stride = 512 (21.3ms frame) -> 46.9 audio tokens per speech second -> 10 seconds ~ 469 audio tokens
- Audio duration is 4.2s on average, and can go up to 20s (long tail).
- May want to filter out samples that have either too short text (<10 tokens) or too short audio (<1s).
- Look at a few samples, the text transcription looks good. No number/case normalization i.e. up to the user. Names are correctly capitalized.
