# TTS

Goal: build a lightweight Vietnamese TTS

Plan:
- [x] Custom tokenizer that extracts tones as separate tokens and put them at the end of a word.
- [ ] Gather some datasets e.g. [ViVoice](https://huggingface.co/datasets/capleaf/viVoice), [VAIS-1000](https://huggingface.co/datasets/doof-ferb/vais1000), [Common Voice](https://datacollective.mozillafoundation.org/datasets/cmj8u3q0300ttnxxbzedg83wq)
  - Might want to make sure audio has high fidelity e.g. check for high-frequency components
- [ ] Gather some ASR+LLM systems to validate transcription + text formatting e.g. [Whisper](https://huggingface.co/openai/whisper-large-v3) (v2 might have less hallucination?), [Parakeet-vi](https://build.nvidia.com/nvidia/parakeet-ctc-0_6b-vi)
- [ ] Find a good vocoder
- [ ] Decide on an architecture + modelling: probably flow matching with an arch similar to Z-Image?
