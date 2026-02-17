# ComfyUI-kaola-soulX-Singer

ComfyUI nodes for [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), a high-quality singing voice synthesis system.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: You might need to install `ffmpeg` and other system dependencies required by `SoulX-Singer`.

## Model Setup

You need to download the models from Hugging Face and place them in the `ComfyUI/models` directory.

1. **SoulX-Singer Model**:
   Download [SoulX-Singer](https://huggingface.co/Soul-AILab/SoulX-Singer) and place it in `models/soulx-singer`.
   For example:
   ```
   ComfyUI/models/soulx-singer/model.pt
   ```
   Note: The config file `soulxsinger.yaml` is expected to be found automatically relative to the repo or model.

2. **Preprocessing Models**:
   Download [SoulX-Singer-Preprocess](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess) and place it in `models/soulx-singer/SoulX-Singer-Preprocess`.
   Structure should look like:
   ```
   ComfyUI/models/soulx-singer/SoulX-Singer-Preprocess/
       ├── rmvpe/
       ├── mel-band-roformer-karaoke/
       ├── dereverb_mel_band_roformer/
       ├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
       ├── parakeet-tdt-0.6b-v2/
       └── rosvot/
   ```

## Nodes

### SoulX-Singer Loader
Loads the SoulX-Singer model.
- **model_path**: Select the model checkpoint (e.g., `model.pt`).

### SoulX-Singer Preprocess
Preprocesses audio for use as prompt or target.
- **audio**: Audio to preprocess.
- **mode**: `prompt` (reference audio) or `target` (melody/lyrics source).
- **vocal_separation**: Whether to separate vocals from background music.
- **language**: Lyric language (Mandarin, Cantonese, English).
- **model_dirs**: Optional path override for preprocess models.

Returns:
- **metadata_path**: Path to the generated metadata JSON.
- **audio_path**: Path to the preprocessed audio WAV.

### SoulX-Singer Generate
Generates singing voice.
- **soulx_model**: The loaded model.
- **prompt_audio_path**: Path to prompt audio (from Preprocess).
- **prompt_metadata_path**: Path to prompt metadata (from Preprocess).
- **target_metadata_path**: Path to target metadata (from Preprocess).
- **control**: `score-controlled` (use midi pitch) or `melody-controlled` (use f0).
- **pitch_shift**: Pitch shift in semitones.
- **auto_shift**: Auto pitch shift.
- **seed**: Random seed.

## Usage

1. Load Model with `SoulX-Singer Loader`.
2. Load Prompt Audio and Target Audio with `Load Audio` nodes.
3. Pass Prompt Audio to `SoulX-Singer Preprocess` (mode="prompt").
4. Pass Target Audio to `SoulX-Singer Preprocess` (mode="target").
5. Connect outputs to `SoulX-Singer Generate`.
6. Output is the generated singing voice.
