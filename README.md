# ComfyUI-kaola-soulX-Singer

ComfyUI nodes for [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), a high-quality singing voice synthesis system.

## Installation

1.  **Clone the repository**:
    Navigate to your ComfyUI's `custom_nodes` directory and run:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/kana112233/ComfyUI-kaola-soulX-Singer.git
    cd ComfyUI-kaola-soulX-Singer
    ```

2.  **Install dependencies**:
    Install the required Python packages. Note that we have resolved compatibility issues for Mac users (MPS support).
    ```bash
    pip install -r requirements.txt
    ```

## Model Download & Setup

You need to download two sets of models: the main **SoulX-Singer model** and the **Preprocessing models**.

### 1. SoulX-Singer Model (Main Model)
Download from Hugging Face: [SoulX-Singer](https://huggingface.co/Soul-AILab/SoulX-Singer)

Place the files in `ComfyUI/models/soulx-singer`.
Your directory structure should look like this:
```
ComfyUI/models/soulx-singer/
├── model.pt
├── config.yaml
├── ... (other assets)
```

### 2. Preprocessing Models (Required for Audio Input)
Download from Hugging Face: [SoulX-Singer-Preprocess](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess)

Place the **contents** of this repo into `ComfyUI/models/soulx-singer/SoulX-Singer-Preprocess`.
Alternatively, you can place them in `ComfyUI/models/SoulX-Singer-Preprocess`.

**Directory Structure Checklist:**
Ensure you have these subfolders inside `ComfyUI/models/soulx-singer/SoulX-Singer-Preprocess/`:
```
ComfyUI/models/soulx-singer/SoulX-Singer-Preprocess/
├── rmvpe/
├── mel-band-roformer-karaoke/
├── dereverb_mel_band_roformer/
├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/  (for Mandarin ASR)
├── parakeet-tdt-0.6b-v2/                                                      (for English ASR)
└── rosvot/
```
**Tip:** You can use `huggingface-cli` or `hfd` to download effectively.
```bash
# Example using hfd script inside the target directory
hfd Soul-AILab/SoulX-Singer-Preprocess
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

## Usage & Workflows

An example workflow is provided in `workflow_example.json`. You can drag and drop this file into ComfyUI to load the pipeline.

1.  **SoulX-Singer Loader**: Selects the model.
2.  **Load Audio**: Two instances, one for "Prompt" (Reference) and one for "Target" (Melody/Lyrics).
3.  **SoulX-Singer Preprocess**:
    *   One set to `mode="prompt"`: Extracts timbre/style from prompt audio.
    *   One set to `mode="target"`: Extracts lyrics/melody from target audio.
4.  **SoulX-Singer Generate**: combines the model, prompt style, and target content to generate new singing voice.
5.  **Save Audio**: Saves the output.

### Basic Steps
1.  Load Model with `SoulX-Singer Loader`.

