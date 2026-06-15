from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import Trainer, Wav2Vec2FeatureExtractor

from pitch import extract_pitch_features


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phone_tokenize(text: str) -> List[str]:
    return [token for token in text.split() if token]


def encode_phones(tokens: Sequence[str], token_to_id: Dict[str, int], row_id: str) -> torch.Tensor:
    missing = [token for token in tokens if token not in token_to_id]
    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise ValueError(f"Unknown phone(s) in row {row_id}: {unique_missing}")
    return torch.tensor([token_to_id[token] for token in tokens], dtype=torch.long)


def load_phone(csv_path: Path | str, fill_missing_transcript: bool = True) -> List[dict]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "path", "canonical"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        has_transcript_column = "transcript" in fieldnames
        rows = []
        for row in reader:
            has_transcript = bool(has_transcript_column and row.get("transcript"))
            row["_has_transcript"] = has_transcript
            if not has_transcript:
                if not fill_missing_transcript:
                    raise ValueError(f"{path} row {row.get('id')} has no transcript")
                row["transcript"] = row["canonical"]
            rows.append(row)
        return rows


def split_rows(rows: List[dict], validation_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if validation_ratio <= 0:
        return rows, []
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(round(len(rows) * validation_ratio)))
    val_indices = set(indices[:val_size])
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train_rows, val_rows


class PhoneDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        audio_root: Path | str,
        token_to_id: Dict[str, int],
        sampling_rate: int,
        speed_perturb_prob: float,
        speed_perturb_range: Tuple[float, float],
        speed_perturb_bins: int,
        pitch_perturb_prob: float,
        pitch_perturb_range: Tuple[float, float],
        pitch_perturb_bins: int,
        pitch_perturb_n_fft: int,
    ):
        self.rows = rows
        self.audio_root = Path(audio_root)
        self.token_to_id = token_to_id
        self.sampling_rate = int(sampling_rate)
        self.speed_perturb_prob = float(speed_perturb_prob)
        self.speed_perturb_values = tuple(
            np.linspace(*speed_perturb_range, max(1, int(speed_perturb_bins)))
        )
        self._resamplers = {}
        self.pitch_perturb_prob = float(pitch_perturb_prob)
        self.pitch_perturb_values = tuple(
            np.linspace(*pitch_perturb_range, max(1, int(pitch_perturb_bins)))
        )
        self.pitch_perturb_n_fft = int(pitch_perturb_n_fft)
        self._pitch_shifters = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_audio_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.audio_root / path

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio_path = self._resolve_audio_path(row["path"])
        wav, sample_rate = torchaudio.load(str(audio_path))

        with torch.no_grad():
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            speed = (
                random.choice(self.speed_perturb_values)
                if random.random() < self.speed_perturb_prob
                else 1.0
            )
            source_rate = max(1, int(round(sample_rate * speed)))
            if source_rate != self.sampling_rate:
                key = (source_rate, self.sampling_rate)
                if key not in self._resamplers:
                    self._resamplers[key] = torchaudio.transforms.Resample(
                        source_rate, self.sampling_rate
                    )
                wav = self._resamplers[key](wav)

            pitch = (
                random.choice(self.pitch_perturb_values)
                if random.random() < self.pitch_perturb_prob
                else 0.0
            )
            if abs(pitch) > 1e-6:
                key = round(float(pitch), 4)
                if key not in self._pitch_shifters:
                    self._pitch_shifters[key] = torchaudio.transforms.PitchShift(
                        self.sampling_rate,
                        key,
                        n_fft=self.pitch_perturb_n_fft,
                    )
                wav = self._pitch_shifters[key](wav)

        labels = encode_phones(phone_tokenize(row["transcript"]), self.token_to_id, row["id"])
        canonical_labels = encode_phones(phone_tokenize(row["canonical"]), self.token_to_id, row["id"])
        return {
            "input_values": wav.squeeze(0).detach().numpy(),
            "labels": labels,
            "canonical_labels": canonical_labels,
        }


@dataclass
class LingWav2Vec2DataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    pad_token_id: int
    sampling_rate: int
    pitch_frame_shift_ms: float
    pitch_freq_low: float
    pitch_freq_high: float
    pitch_nccf_smoothing_frames: int
    label_pad_id: int = -100

    def __call__(self, features: List[dict]) -> dict:
        audio = [feature["input_values"] for feature in features]
        batch = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        pitch_features = [
            extract_pitch_features(
                feature["input_values"],
                sampling_rate=self.sampling_rate,
                frame_shift_ms=self.pitch_frame_shift_ms,
                freq_low=self.pitch_freq_low,
                freq_high=self.pitch_freq_high,
                nccf_smoothing_frames=self.pitch_nccf_smoothing_frames,
            )
            for feature in features
        ]
        pitch_lengths = torch.tensor([pitch.size(0) for pitch in pitch_features], dtype=torch.long)
        batch["pitch_features"] = pad_sequence(pitch_features, batch_first=True, padding_value=0.0)
        frame_idx = torch.arange(batch["pitch_features"].size(1)).unsqueeze(0)
        batch["pitch_attention_mask"] = frame_idx.lt(pitch_lengths.unsqueeze(1)).long()

        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        batch["labels"] = labels.masked_fill(labels.eq(self.pad_token_id), self.label_pad_id)

        canonical_labels = pad_sequence(
            [feature["canonical_labels"] for feature in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        batch["canonical_labels"] = canonical_labels
        batch["canonical_attention_mask"] = canonical_labels.ne(self.pad_token_id).long()
        return batch


def build_sample_weights(
    rows: Sequence[dict],
    normal_weight: float,
    error_weight: float,
) -> torch.Tensor:
    return torch.tensor(
        [
            error_weight
            if phone_tokenize(row["canonical"]) != phone_tokenize(row["transcript"])
            else normal_weight
            for row in rows
        ],
        dtype=torch.double,
    )


class WeightedTrainer(Trainer):
    def __init__(self, *args, train_sample_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sample_weights = train_sample_weights

    def _get_train_sampler(self, train_dataset=None):
        if self.train_sample_weights is None:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if len(self.train_sample_weights) != len(dataset):
            raise ValueError(
                f"train_sample_weights length ({len(self.train_sample_weights)}) must match "
                f"train_dataset length ({len(dataset)})."
            )
        return WeightedRandomSampler(
            self.train_sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
