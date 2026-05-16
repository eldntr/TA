#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from huggingface_hub import hf_hub_download


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


def iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for pattern in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"):
        yield from sorted(input_dir.rglob(pattern))


def copy_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for name in (
        "metadata.csv",
        "train_list.txt",
        "val_list.txt",
        "test_list.txt",
        "speaker_map.txt",
        "judul_map.txt",
        "total_duration.txt",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a speech dataset with Sidon without overwriting the source audio.")
    parser.add_argument("--input-dir", default="dataset", help="Dataset root that contains wavs/ and metadata files.")
    parser.add_argument("--output-dir", default="dataset_sidon", help="Output dataset root for restored audio.")
    parser.add_argument("--repo-id", default="sarulab-speech/sidon-v0.1", help="Hugging Face repository id for Sidon checkpoints.")
    parser.add_argument("--model-dir", default="models/sidon-v0.1", help="Local directory where Sidon checkpoints are stored.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk size passed through Sidon.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of chunks processed together.")
    parser.add_argument("--save-sample-rate", type=int, default=16_000, help="Sample rate for saved audio files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for dry validation runs.")
    parser.add_argument("--include-file", action="append", default=None, help="Only process the given relative audio filename(s), e.g. 011_128_0871.wav.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist in the output directory.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def download_model_files(repo_id: str, model_dir: Path, device: str) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = "cuda" if device.startswith("cuda") else "cpu"
    feature_name = f"feature_extractor_{suffix}.pt"
    decoder_name = f"decoder_{suffix}.pt"
    feature_local = model_dir / feature_name
    decoder_local = model_dir / decoder_name

    if feature_local.exists() and decoder_local.exists():
        return feature_local, decoder_local

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


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    input_wavs = input_root / "wavs"
    output_wavs = output_root / "wavs"

    if not input_wavs.exists():
        raise FileNotFoundError(f"Input audio directory not found: {input_wavs}")

    device = resolve_device(args.device)
    feature_extractor_path, decoder_path = download_model_files(
        args.repo_id,
        Path(args.model_dir).resolve(),
        device,
    )

    output_wavs.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(input_root, output_root)

    restorer = SidonRestorer(
        feature_extractor_path=feature_extractor_path,
        decoder_path=decoder_path,
        device=device,
        chunk_seconds=args.chunk_seconds,
    )

    audio_files = list(iter_audio_files(input_wavs))
    if args.include_file:
        include = set(args.include_file)
        audio_files = [path for path in audio_files if path.name in include]
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    total = len(audio_files)

    for index, audio_path in enumerate(audio_files, start=1):
        rel_path = audio_path.relative_to(input_wavs)
        output_path = output_wavs / rel_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and output_path.exists():
            print(f"[{index}/{total}] skip {rel_path}")
            continue

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            restored = restorer.restore_waveform(waveform, sample_rate, args.batch_size)
            save_waveform = restored.unsqueeze(0)
            if args.save_sample_rate != restorer.target_sample_rate:
                save_waveform = torchaudio.functional.resample(
                    save_waveform,
                    restorer.target_sample_rate,
                    args.save_sample_rate,
                )
            torchaudio.save(str(output_path), save_waveform.cpu(), args.save_sample_rate)
            print(f"[{index}/{total}] ok   {rel_path}")
        except Exception as exc:  # noqa: BLE001
            failures.append((rel_path, str(exc)))
            print(f"[{index}/{total}] fail {rel_path}: {exc}", file=sys.stderr)

    if failures:
        failure_log = output_root / "sidon_failures.txt"
        with failure_log.open("w", encoding="utf-8") as handle:
            for rel_path, message in failures:
                handle.write(f"{rel_path}\t{message}\n")
        print(f"Completed with {len(failures)} failures. See {failure_log}.", file=sys.stderr)
        return 2

    print(f"Restored {total} files into {output_wavs}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
