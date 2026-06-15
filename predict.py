from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import TrainingArguments, Wav2Vec2FeatureExtractor

from config import expand_path, resolve_fp16
from dataset import LingWav2Vec2DataCollator, PhoneDataset, WeightedTrainer, load_phone, seed_everything
from metrics import build_compute_metrics, ctc_decode_ids
from model import LingWav2Vec2ForCTC
from train import _model_kwargs
from vocab import load_vocab


def _feature_extractor_source(checkpoint_path: Path, base_model: str) -> str:
    if (checkpoint_path / "preprocessor_config.json").exists():
        return str(checkpoint_path)
    return base_model


def _prediction_args_kwargs(cfg: dict) -> dict:
    kwargs = {
        "output_dir": str(expand_path(cfg["output_dir"])),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "fp16": resolve_fp16(cfg["fp16"]),
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": "none",
        "dataloader_num_workers": int(cfg["dataloader_num_workers"]),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_key] = "no"
    return kwargs


def _trainer_kwargs(feature_extractor, **kwargs) -> dict:
    signature = inspect.signature(WeightedTrainer.__mro__[1].__init__)
    result = dict(kwargs)
    if "processing_class" in signature.parameters:
        result["processing_class"] = feature_extractor
    else:
        result["tokenizer"] = feature_extractor
    return result


def write_submission(rows_out: list[dict], output_csv: Path | str) -> None:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "predict"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "predict": row["predict"]} for row in rows_out)


def predict(
    cfg: dict,
    data_path: str | Path,
    checkpoint_path: str | Path,
    output_csv: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
):
    seed_everything(int(cfg["seed"]))
    checkpoint_path = expand_path(checkpoint_path)

    token_to_id, id_to_token = load_vocab(expand_path(cfg["vocab_file"]))
    if "<pad>" not in token_to_id:
        raise ValueError('The vocab must contain "<pad>" as the CTC blank/pad token.')
    pad_token_id = token_to_id["<pad>"]

    rows = load_phone(expand_path(data_path))
    if max_samples is not None:
        rows = rows[:max_samples]

    dataset = PhoneDataset(
        rows=rows,
        audio_root=expand_path(cfg["audio_root"]),
        token_to_id=token_to_id,
        sampling_rate=int(cfg["sampling_rate"]),
        speed_perturb_prob=float(cfg["eval_speed_perturb_prob"]),
        speed_perturb_range=tuple(cfg["eval_speed_perturb_range"]),
        speed_perturb_bins=int(cfg["eval_speed_perturb_bins"]),
        pitch_perturb_prob=float(cfg["eval_pitch_perturb_prob"]),
        pitch_perturb_range=tuple(cfg["eval_pitch_perturb_range"]),
        pitch_perturb_bins=int(cfg["eval_pitch_perturb_bins"]),
        pitch_perturb_n_fft=int(cfg["eval_pitch_perturb_n_fft"]),
    )

    feature_source = _feature_extractor_source(checkpoint_path, cfg["base_model"])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_source)
    data_collator = LingWav2Vec2DataCollator(
        feature_extractor=feature_extractor,
        pad_token_id=pad_token_id,
        sampling_rate=int(cfg["sampling_rate"]),
        pitch_frame_shift_ms=float(cfg["pitch_frame_shift_ms"]),
        pitch_freq_low=float(cfg["pitch_freq_low"]),
        pitch_freq_high=float(cfg["pitch_freq_high"]),
        pitch_nccf_smoothing_frames=int(cfg["pitch_nccf_smoothing_frames"]),
    )

    model = LingWav2Vec2ForCTC.from_pretrained(
        str(checkpoint_path),
        **_model_kwargs(cfg, len(id_to_token), pad_token_id),
    )
    model.config.id2label = {idx: token for idx, token in enumerate(id_to_token)}
    model.config.label2id = {token: idx for idx, token in enumerate(id_to_token)}
    model.eval()

    can_compute_metrics = all(row.get("_has_transcript", False) for row in rows)
    if not can_compute_metrics:
        print("Skip metrics because this CSV has no transcript column.")

    trainer = WeightedTrainer(
        **_trainer_kwargs(
            feature_extractor,
            model=model,
            args=TrainingArguments(**_prediction_args_kwargs(cfg)),
            data_collator=data_collator,
            eval_dataset=dataset,
            compute_metrics=(
                build_compute_metrics(
                    id_to_token,
                    pad_token_id,
                    rows,
                )
                if can_compute_metrics
                else None
            ),
        )
    )

    with torch.inference_mode():
        pred_output = trainer.predict(dataset)

    logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    pred_ids = np.argmax(logits, axis=-1)

    rows_out = []
    for row, ids in zip(rows, pred_ids):
        prediction = " ".join(ctc_decode_ids(ids, id_to_token, pad_token_id))
        rows_out.append(
            {
                "id": row["id"],
                "path": row["path"],
                "canonical": row["canonical"],
                "transcript": row["transcript"],
                "predict": prediction,
            }
        )

    print("=== Metrics ===")
    for key, value in pred_output.metrics.items():
        print(f"{key}: {value}")

    if output_csv is not None:
        write_submission(rows_out, output_csv)
        print(f"Saved predictions to: {output_csv}")

    return rows_out, pred_output.metrics
