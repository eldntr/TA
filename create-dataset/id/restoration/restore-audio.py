#!/usr/bin/env python3
import os
import sys
import glob
import torch
import torchaudio
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Iterable

# Simplified padding for Sidon
def _pad_batch(features: list[torch.Tensor], padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(feature.shape[0] for feature in features)
    target_length = ((target_length + 1) // 2) * 2

    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device

    padded = torch.full(
        (batch_size, target_length, feature_dim),
        padding_value,
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)

    for index, feature in enumerate(features):
        seq_len = feature.shape[0]
        padded[index, :seq_len] = feature
        attention_mask[index, :seq_len] = 1

    return padded, attention_mask

def extract_seamless_m4t_features(
    raw_speech: list[torch.Tensor],
    *,
    sampling_rate: int = 16_000,
    stride: int = 2,
    padding_value: float = 1.0,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    features: list[torch.Tensor] = []
    for waveform in raw_speech:
        if waveform.ndim > 1:
            waveform = waveform[0]
        feature = torchaudio.compliance.kaldi.fbank(
            waveform=waveform.unsqueeze(0),
            sample_frequency=sampling_rate,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feature.mean(0, keepdim=True)
        var = feature.var(0, keepdim=True)
        features.append((feature - mean) / torch.sqrt(var + 1e-5))

    input_features, attention_mask = _pad_batch(features, padding_value=padding_value)
    batch_size, num_frames, num_channels = input_features.shape
    new_num_frames = (num_frames // stride) * stride
    input_features = input_features[:, :new_num_frames, :]
    attention_mask = attention_mask[:, :new_num_frames]
    input_features = input_features.reshape(batch_size, new_num_frames // stride, num_channels * stride)

    return {
        "input_features": input_features.to(device),
        "attention_mask": attention_mask[:, 1::stride].to(device),
    }

class SidonRestorer:
    def __init__(
        self,
        *,
        feature_extractor_path: Path,
        decoder_path: Path,
        device: str,
        input_sample_rate: int = 16_000,
        target_sample_rate: int = 48_000,
        chunk_seconds: float = 30.0,
    ) -> None:
        self.device = torch.device(device)
        self.feature_extractor = torch.jit.load(str(feature_extractor_path), map_location=self.device)
        self.decoder = torch.jit.load(str(decoder_path), map_location=self.device)
        self.feature_extractor.eval()
        self.decoder.eval()
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_samples = max(int(round(chunk_seconds * input_sample_rate)), 1_600)

    def _prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.to(dtype=torch.float32)
        if sample_rate != self.input_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.input_sample_rate)
        return waveform.contiguous()

    @torch.inference_mode()
    def restore_chunks(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        padded = torch.zeros((len(chunks), self.chunk_samples), dtype=torch.float32)
        expected_lengths: list[int] = []

        for index, chunk in enumerate(chunks):
            truncated = chunk[: self.chunk_samples]
            padded[index, : truncated.shape[-1]] = truncated
            duration_seconds = truncated.shape[-1] / float(self.input_sample_rate)
            expected_lengths.append(int(round(duration_seconds * self.target_sample_rate)))

        features = extract_seamless_m4t_features(
            [sample for sample in padded],
            device=self.device,
        )
        hidden = self.feature_extractor(features["input_features"])["last_hidden_state"]
        restored = self.decoder(hidden.transpose(1, 2)).cpu()

        outputs: list[torch.Tensor] = []
        for index, sample in enumerate(restored):
            flat = sample.reshape(-1)
            target_length = expected_lengths[index]
            current_length = flat.shape[-1]
            if current_length < target_length:
                flat = torch.nn.functional.pad(flat, (0, target_length - current_length))
            elif current_length > target_length:
                flat = flat[:target_length]
            outputs.append(flat.contiguous())
        return outputs

    def restore_waveform(self, waveform: torch.Tensor, sample_rate: int, batch_size: int) -> torch.Tensor:
        prepared = self._prepare_waveform(waveform, sample_rate)
        chunks = [
            prepared[start : start + self.chunk_samples]
            for start in range(0, prepared.shape[-1], self.chunk_samples)
        ]
        restored_parts: list[torch.Tensor] = []
        for start in range(0, len(chunks), batch_size):
            restored_parts.extend(self.restore_chunks(chunks[start : start + batch_size]))
        return torch.cat(restored_parts, dim=0)

def download_model_files(repo_id: str, model_dir: Path, device: str) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "cuda" if device.startswith("cuda") else "cpu"
    feature_name = f"feature_extractor_{suffix}.pt"
    decoder_name = f"decoder_{suffix}.pt"
    feature_local = model_dir / feature_name
    decoder_local = model_dir / decoder_name

    if feature_local.exists() and decoder_local.exists():
        return feature_local, decoder_local

    print(f"Downloading Sidon models from {repo_id}...")
    feature_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=feature_name,
            local_dir=str(model_dir),
        )
    )
    decoder_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=decoder_name,
            local_dir=str(model_dir),
        )
    )
    return feature_path, decoder_path

def restore_audio_directory(input_dir, output_dir, device="auto", batch_size=4):
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist.")
        return

    # Resolve device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Model settings
    repo_id = "sarulab-speech/sidon-v0.1"
    # Go up 3 levels from create-dataset/id/restoration/ to TA/ and then into models/
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir.parents[2] / "models" / "sidon-v0.1"
    
    feature_path, decoder_path = download_model_files(repo_id, model_dir, device)
    
    restorer = SidonRestorer(
        feature_extractor_path=feature_path,
        decoder_path=decoder_path,
        device=device
    )
    
    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"):
        audio_files.extend(list(input_root.rglob(ext)))
    
    audio_files.sort()
    total = len(audio_files)
    print(f"Found {total} audio files to restore.")
    
    for i, audio_path in enumerate(audio_files, 1):
        rel_path = audio_path.relative_to(input_root)
        out_path = output_root / rel_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if out_path.exists():
            print(f"[{i}/{total}] Skipping (already exists): {rel_path}")
            continue
            
        print(f"[{i}/{total}] Restoring: {rel_path}")
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            restored = restorer.restore_waveform(waveform, sample_rate, batch_size)
            
            # Save at 16kHz mono (matching your pipeline's needs)
            save_waveform = restored.unsqueeze(0)
            target_sr = 16000
            if target_sr != restorer.target_sample_rate:
                save_waveform = torchaudio.functional.resample(
                    save_waveform,
                    restorer.target_sample_rate,
                    target_sr,
                )
            
            torchaudio.save(str(out_path), save_waveform.cpu(), target_sr)
        except Exception as e:
            print(f"[{i}/{total}] Failed to restore {rel_path}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir: TA/create-dataset/id/restoration
    # parent: TA/create-dataset/id
    base_dir = os.path.dirname(script_dir)
    
    input_directory = os.path.join(base_dir, "dataset_audio")
    output_directory = os.path.join(base_dir, "dataset_audio_restored")
    
    print(f"Source: {input_directory}")
    print(f"Destination: {output_directory}")
    
    restore_audio_directory(input_directory, output_directory)
    print("Restoration process completed.")
