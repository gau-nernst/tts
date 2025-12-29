# TTS

Goal: build a lightweight Vietnamese TTS

Plan:
- [x] Custom tokenizer that extracts tones as separate tokens and put them at the end of a word.
- [x] Gather some datasets. See [DATASETS.md](DATASETS.md).
  - Might want to make sure audio has high fidelity e.g. check for high-frequency components
- [ ] Gather some ASR+LLM systems to validate transcription + text formatting e.g. [Whisper](https://huggingface.co/openai/whisper-large-v3) (v2 might have less hallucination?), [Parakeet-vi](https://build.nvidia.com/nvidia/parakeet-ctc-0_6b-vi)
- [ ] Find a good vocoder
- [ ] Decide on an architecture + modelling: probably flow matching with an arch similar to Z-Image?
