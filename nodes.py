import os
import sys
import torch
import folder_paths
import logging
import torchaudio
import tempfile
import soundfile as sf
import shutil
import json
from pathlib import Path
import types
import unittest.mock as mock

# --- NeMo 2.6.1 + Mac Import Fix ---
# NeMo 2.6.1 has a hard dependency on 'nv_one_logger' which is internal/missing.
# We mock it here to prevent ImportError when importing nemo.collections.asr
def _mock_nemo_logger():
    if "nv_one_logger" in sys.modules:
        return

    def mock_module(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    m_main = mock_module("nv_one_logger")
    m_api = mock_module("nv_one_logger.api")
    m_api_cfg = mock_module("nv_one_logger.api.config")
    m_tt = mock_module("nv_one_logger.training_telemetry")
    m_tt_api = mock_module("nv_one_logger.training_telemetry.api")
    m_tt_api_cb = mock_module("nv_one_logger.training_telemetry.api.callbacks")
    m_tt_api_cfg = mock_module("nv_one_logger.training_telemetry.api.config")
    m_tt_api_prov = mock_module("nv_one_logger.training_telemetry.api.training_telemetry_provider")
    m_tt_int = mock_module("nv_one_logger.training_telemetry.integration")
    m_tt_int_pl = mock_module("nv_one_logger.training_telemetry.integration.pytorch_lightning")

    # Populate attributes
    m_api_cfg.OneLoggerConfig = mock.MagicMock()
    m_tt_api_cb.on_app_start = mock.MagicMock()
    m_tt_api_cfg.TrainingTelemetryConfig = mock.MagicMock()
    m_tt_api_prov.TrainingTelemetryProvider = mock.MagicMock()

    class MockTimeEventCallback:
        def __init__(self, *args, **kwargs):
            pass
    m_tt_int_pl.TimeEventCallback = MockTimeEventCallback

_mock_nemo_logger()
# -----------------------------------

# Add the soulx_singer_repo directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
soulx_repo_path = os.path.join(current_dir, "soulx_singer_repo")
if soulx_repo_path not in sys.path:
    sys.path.insert(0, soulx_repo_path)

# Import from soulx_singer_repo
try:
    from cli.inference import build_model as build_svs_model, process as svs_process
    from soulxsinger.utils.file_utils import load_config
    
    # Import preprocess tools
    # We import PreprocessPipeline from pipeline but we will override it
    from preprocess.pipeline import PreprocessPipeline
    from preprocess.tools import (
        F0Extractor,
        VocalDetector,
        VocalSeparator,
        NoteTranscriber,
        LyricTranscriber,
    )
    from preprocess.utils import convert_metadata, merge_short_segments
    from preprocess.tools.midi_parser import MidiParser

except ImportError as e:
    logging.error(f"Failed to import soulx_singer_repo modules: {e}")
    # We might fail here if dependencies are missing, but ComfyUI will catch it.
    pass

# Define constants
# Define constants
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Register model path
# Register model path
# Ensure 'soulx-singer' is looked for in ComfyUI/models/soulx-singer
if "soulx-singer" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("soulx-singer", os.path.join(folder_paths.models_dir, "soulx-singer"))

class SoulXSingerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (folder_paths.get_filename_list("soulx-singer"),),
            },
        }

    RETURN_TYPES = ("SOULX_MODEL",)
    RETURN_NAMES = ("soulx_model",)
    FUNCTION = "load_model"
    CATEGORY = "SoulXSinger"

    def load_model(self, model_path):
        model_full_path = folder_paths.get_full_path("soulx-singer", model_path)
        if not model_full_path:
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Config location assumptions
        config_path = os.path.join(soulx_repo_path, "soulxsinger", "config", "soulxsinger.yaml")
        if not os.path.exists(config_path):
             # Try next to model
             config_path = os.path.join(os.path.dirname(model_full_path), "soulxsinger.yaml")
        
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config not found. Expected at standard repo location or next to model.")

        config = load_config(config_path)
        
        # Load model
        model = build_svs_model(
            model_path=model_full_path,
            config=config,
            device=DEVICE,
        )
        
        soulx_model = {
            "model": model,
            "config": config,
            "device": DEVICE,
            "phoneset_path": os.path.join(soulx_repo_path, "soulxsinger", "utils", "phoneme", "phone_set.json"),
            "model_path": model_full_path,
        }
        return (soulx_model,)

class CustomPreprocessPipeline(PreprocessPipeline):
    def __init__(self, device: str, language: str, save_dir: str, models_root: str, vocal_sep: bool = True, max_merge_duration: int = 60000):
        self.device = device
        self.language = language
        self.save_dir = save_dir
        self.vocal_sep = vocal_sep
        self.max_merge_duration = max_merge_duration
        
        # Construct paths
        rmvpe_path = os.path.join(models_root, "rmvpe", "rmvpe.pt")
        
        if vocal_sep:
            sep_model = os.path.join(models_root, "mel-band-roformer-karaoke", "mel_band_roformer_karaoke_becruily.ckpt")
            sep_config = os.path.join(models_root, "mel-band-roformer-karaoke", "config_karaoke_becruily.yaml")
            der_model = os.path.join(models_root, "dereverb_mel_band_roformer", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
            der_config = os.path.join(models_root, "dereverb_mel_band_roformer", "dereverb_mel_band_roformer_anvuew.yaml")
            
            if not all(os.path.exists(p) for p in [sep_model, sep_config, der_model, der_config]):
                logging.warning(f"Vocal separation models missing in {models_root}. Make sure they are downloaded.")

            self.vocal_separator = VocalSeparator(
                sep_model_path=sep_model,
                sep_config_path=sep_config,
                der_model_path=der_model,
                der_config_path=der_config,
                device=device
            )
        else:
            self.vocal_separator = None

        if not os.path.exists(rmvpe_path):
             logging.warning(f"RMVPE model missing at {rmvpe_path}")

        self.f0_extractor = F0Extractor(
            model_path=rmvpe_path,
            device=device,
        )
        
        self.vocal_detector = VocalDetector(
            cut_wavs_output_dir=f"{save_dir}/cut_wavs",
        )
        
        zh_model = os.path.join(models_root, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        en_model = os.path.join(models_root, "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")
        
        self.lyric_transcriber = LyricTranscriber(
            zh_model_path=zh_model,
            en_model_path=en_model,
            device=device
        )
        
        rosvot_model = os.path.join(models_root, "rosvot", "rosvot", "model.pt")
        rwbd_model = os.path.join(models_root, "rosvot", "rwbd", "model.pt")

        self.note_transcriber = NoteTranscriber(
            rosvot_model_path=rosvot_model,
            rwbd_model_path=rwbd_model,
            device=device
        )
        
    # We reuse parent's run method as it uses self.* components we just initialized

class SoulXSingerPreprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["prompt", "target"],),
                "language": (["Mandarin", "Cantonese", "English"],),
                "vocal_separation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                 "model_dirs": ("STRING", {"default": "models/soulx-singer/SoulX-Singer-Preprocess"}), 
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata_path", "audio_path")
    FUNCTION = "preprocess"
    CATEGORY = "SoulXSinger"

    def preprocess(self, audio, mode, language, vocal_separation, model_dirs=""):
        # Determine paths
        # Search priority: 
        # 1. ComfyUI/models/soulx-singer/SoulX-Singer-Preprocess
        # 2. ComfyUI/models/SoulX-Singer-Preprocess
        # 3. repo/pretrained_models/SoulX-Singer-Preprocess
        
        possible_roots = []
        if folder_paths.models_dir:
             possible_roots.append(os.path.join(folder_paths.models_dir, "soulx-singer", "SoulX-Singer-Preprocess"))
             possible_roots.append(os.path.join(folder_paths.models_dir, "SoulX-Singer-Preprocess"))
        
        # User specified path (Removed for portability)
        # user_model_path = "/Users/xiohu/work/comfyUI/ComfyUI-source/models/SoulX-Singer"
        # possible_roots.append(os.path.join(user_model_path, "SoulX-Singer-Preprocess"))

        possible_roots.append(os.path.join(soulx_repo_path, "pretrained_models", "SoulX-Singer-Preprocess"))
        
        models_root = None
        for p in possible_roots:
            if os.path.exists(p):
                models_root = p
                break
        
        if not models_root:
            raise FileNotFoundError(f"Could not find SoulX-Singer-Preprocess models. Checked: {possible_roots}. Please download them.")
        
        output_dir = folder_paths.get_output_directory()
        temp_dir = os.path.join(output_dir, "soulx_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = os.urandom(4).hex()
        audio_name = f"{mode}_{timestamp}.wav"
        save_path = os.path.join(temp_dir, audio_name)

        # Save audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3:
            waveform = waveform[0] # (channels, samples)
            
        if waveform.size(0) > waveform.size(1):
             waveform = waveform.t()

        # torchaudio.save expects (channels, samples)
        torchaudio.save(save_path, waveform, sample_rate)
        
        preprocess_save_dir = os.path.join(temp_dir, f"{mode}_preprocess_{timestamp}")
        # Clean up previous run if same timestamp collision (unlikely)
        if os.path.exists(preprocess_save_dir):
            shutil.rmtree(preprocess_save_dir)
        os.makedirs(preprocess_save_dir, exist_ok=True)
        
        pipeline = CustomPreprocessPipeline(
            device=DEVICE,
            language=language,
            save_dir=preprocess_save_dir,
            models_root=models_root,
            vocal_sep=vocal_separation,
            max_merge_duration=60000,
        )
        
        try:
            pipeline.run(
                audio_path=save_path,
                vocal_sep=vocal_separation,
                max_merge_duration=60000,
                language=language,
            )
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")

        metadata_path = os.path.join(preprocess_save_dir, "metadata.json")
        if not os.path.exists(metadata_path):
             raise RuntimeError(f"Preprocessing failed to generate metadata at {metadata_path}")
        
        return (metadata_path, save_path)


class SoulXSingerGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "soulx_model": ("SOULX_MODEL",),
                "prompt_audio_path": ("STRING", {"default": "", "multiline": False}),
                "prompt_metadata_path": ("STRING", {"default": "", "multiline": False}),
                "target_metadata_path": ("STRING", {"default": "", "multiline": False}),
                "control": (["score-controlled", "melody-controlled"], {"default": "score-controlled"}),
                "pitch_shift": ("FLOAT", {"default": 0, "min": -36, "max": 36, "step": 1}),
                "auto_shift": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "SoulXSinger"

    def generate(self, soulx_model, prompt_audio_path, prompt_metadata_path, target_metadata_path, control, pitch_shift, auto_shift, seed):
        torch.manual_seed(seed)
        
        output_dir = folder_paths.get_output_directory()
        save_dir = os.path.join(output_dir, "soulx_gen")
        os.makedirs(save_dir, exist_ok=True)
        
        class Args:
            pass
        args = Args()
        args.device = soulx_model["device"]
        args.model_path = soulx_model["model_path"]
        args.config = "" 
        args.prompt_wav_path = prompt_audio_path
        args.prompt_metadata_path = prompt_metadata_path
        args.target_metadata_path = target_metadata_path
        args.phoneset_path = soulx_model["phoneset_path"]
        args.save_dir = save_dir
        args.auto_shift = auto_shift
        args.pitch_shift = int(pitch_shift)
        args.control = "melody" if control == "melody-controlled" else "score"

        try:
             svs_process(args, soulx_model["config"], soulx_model["model"])
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        generated_path = os.path.join(save_dir, "generated.wav")
        if not os.path.exists(generated_path):
            raise RuntimeError("Generation finished but output file not found.")

        # Load as (channels, samples)
        waveform, sample_rate = torchaudio.load(generated_path)
        
        # Add batch dim: (1, channels, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            
        return ({"waveform": waveform, "sample_rate": sample_rate},)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SoulXSingerLoader": SoulXSingerLoader,
    "SoulXSingerPreprocess": SoulXSingerPreprocess,
    "SoulXSingerGenerate": SoulXSingerGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoulXSingerLoader": "SoulX-Singer Loader",
    "SoulXSingerPreprocess": "SoulX-Singer Preprocess",
    "SoulXSingerGenerate": "SoulX-Singer Generate",
}
