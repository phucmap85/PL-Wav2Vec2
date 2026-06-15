from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, Wav2Vec2FeatureExtractor
from transformers.trainer_utils import get_last_checkpoint

from config import expand_path, resolve_fp16
from dataset import (
    LingWav2Vec2DataCollator,
    PhoneDataset,
    WeightedTrainer,
    build_sample_weights,
    load_phone,
    phone_tokenize,
    seed_everything,
    split_rows,
)
from metrics import build_compute_metrics
from model import LingWav2Vec2ForCTC
from optim import build_optimizer, choose_resume_checkpoint_for_trainer
from vocab import load_vocab


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")


def _training_args_kwargs(cfg: dict, has_eval: bool) -> dict:
    fp16 = resolve_fp16(cfg["fp16"])
    kwargs = {
        "output_dir": str(expand_path(cfg["output_dir"])),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "fp16": fp16,
        "learning_rate": float(cfg["adam_learning_rate"]),
        "weight_decay": float(cfg["adam_weight_decay"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "logging_steps": int(cfg["logging_steps"]),
        "save_steps": int(cfg["save_steps"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "report_to": cfg["report_to"],
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "save_strategy": "steps",
        "dataloader_num_workers": int(cfg["dataloader_num_workers"]),
    }

    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"

    if has_eval:
        kwargs.update(
            {
                "eval_steps": int(cfg["eval_steps"]),
                "load_best_model_at_end": True,
                "metric_for_best_model": "score",
                "greater_is_better": True,
            }
        )
    return kwargs


def _trainer_kwargs(feature_extractor, **kwargs) -> dict:
    signature = inspect.signature(WeightedTrainer.__init__)
    hf_signature = inspect.signature(WeightedTrainer.__mro__[1].__init__)
    result = dict(kwargs)
    if "processing_class" in hf_signature.parameters:
        result["processing_class"] = feature_extractor
    else:
        result["tokenizer"] = feature_extractor
    return result


def _model_kwargs(cfg: dict, vocab_size: int, pad_token_id: int) -> dict:
    return {
        "ignore_mismatched_sizes": False,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "ctc_loss_reduction": "mean",
        "ctc_zero_infinity": True,
        "max_canonical_positions": int(cfg["max_canonical_positions"]),
        "pitch_input_dim": int(cfg["pitch_input_dim"]),
        "pitch_encoder_hidden_size": cfg["pitch_encoder_hidden_size"],
        "pitch_dropout": cfg["pitch_dropout"],
        "apply_spec_augment": bool(cfg["apply_spec_augment"]),
        "mask_time_prob": float(cfg["mask_time_prob"]),
        "mask_time_length": int(cfg["mask_time_length"]),
        "mask_feature_prob": float(cfg["mask_feature_prob"]),
        "mask_feature_length": int(cfg["mask_feature_length"]),
    }

def run_training(cfg: dict):
    seed_everything(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    token_to_id, id_to_token = load_vocab(expand_path(cfg["vocab_file"]))
    if "<pad>" not in token_to_id:
        raise ValueError('The vocab must contain "<pad>" as the CTC blank/pad token.')
    pad_token_id = token_to_id["<pad>"]

    train_rows = load_phone(expand_path(cfg["train_csv"]))
    eval_csv = expand_path(cfg["eval_csv"])
    if eval_csv is not None:
        eval_rows = load_phone(eval_csv)
    else:
        if cfg["validation_ratio"] is None:
            raise ValueError("validation_ratio must be set when eval_csv is null.")
        train_rows, eval_rows = split_rows(
            train_rows,
            float(cfg["validation_ratio"]),
            int(cfg["seed"]),
        )

    train_dataset = PhoneDataset(
        rows=train_rows,
        audio_root=expand_path(cfg["audio_root"]),
        token_to_id=token_to_id,
        sampling_rate=int(cfg["sampling_rate"]),
        speed_perturb_prob=float(cfg["speed_perturb_prob"]),
        speed_perturb_range=tuple(cfg["speed_perturb_range"]),
        speed_perturb_bins=int(cfg["speed_perturb_bins"]),
        pitch_perturb_prob=float(cfg["pitch_perturb_prob"]),
        pitch_perturb_range=tuple(cfg["pitch_perturb_range"]),
        pitch_perturb_bins=int(cfg["pitch_perturb_bins"]),
        pitch_perturb_n_fft=int(cfg["pitch_perturb_n_fft"]),
    )
    train_sample_weights = None
    if cfg["use_weighted_sampler"]:
        train_sample_weights = build_sample_weights(
            train_rows,
            normal_weight=float(cfg["normal_sample_weight"]),
            error_weight=float(cfg["error_sample_weight"]),
        )

    eval_dataset = (
        PhoneDataset(
            rows=eval_rows,
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
        if eval_rows
        else None
    )

    output_dir = expand_path(cfg["output_dir"])
    resume_value = cfg["resume_from_checkpoint"]
    checkpoint_path = None
    if resume_value is True:
        last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        if last_checkpoint is None:
            raise ValueError(f"resume_from_checkpoint=True but no checkpoint was found in {output_dir}.")
        checkpoint_path = Path(last_checkpoint)
    elif resume_value:
        checkpoint_path = expand_path(resume_value)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_source = str(checkpoint_path) if checkpoint_path is not None else cfg["base_model"]
    if checkpoint_path is not None:
        print(f"Loading model weights from checkpoint: {checkpoint_path}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg["base_model"])
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
        model_source,
        init_head=True,
        **_model_kwargs(cfg, len(id_to_token), pad_token_id),
    )
    model.config.id2label = {idx: token for idx, token in enumerate(id_to_token)}
    model.config.label2id = {token: idx for idx, token in enumerate(id_to_token)}

    if cfg["freeze_wav2vec2"]:
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
    elif cfg["freeze_feature_encoder"]:
        model.freeze_feature_encoder()

    optimizer = build_optimizer(model, cfg)
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    training_args = TrainingArguments(**_training_args_kwargs(cfg, has_eval=has_eval))

    trainer = WeightedTrainer(
        **_trainer_kwargs(
            feature_extractor,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=(
                build_compute_metrics(
                    id_to_token,
                    pad_token_id,
                    eval_rows,
                )
                if has_eval
                else None
            ),
            optimizers=(optimizer, None),
            train_sample_weights=train_sample_weights,
        )
    )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Vocab size: {len(id_to_token)}")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params} ({trainable_params / total_params:.2%})")
    if train_sample_weights is not None:
        n_error = sum(
            phone_tokenize(row["canonical"]) != phone_tokenize(row["transcript"])
            for row in train_rows
        )
        print(f"Weighted sampler: {n_error}/{len(train_rows)} error samples")

    trainer_resume_checkpoint = choose_resume_checkpoint_for_trainer(
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
    )
    train_result = trainer.train(resume_from_checkpoint=trainer_resume_checkpoint)
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_out = {token: idx for idx, token in enumerate(id_to_token)}
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval_metrics = trainer.evaluate() if has_eval else None
    if eval_metrics is not None:
        print(eval_metrics)
    return trainer, train_result, eval_metrics
