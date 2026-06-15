from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio


def normalize_pitch_sequence(
    pitch_hz: torch.Tensor,
    voicing: Optional[torch.Tensor] = None,
    log_floor: float = 1.0,
) -> torch.Tensor:
    pitch_hz = pitch_hz.float().flatten().nan_to_num(0.0).clamp_min(0.0)
    voiced = pitch_hz > log_floor

    log_pitch = torch.zeros_like(pitch_hz)
    if voiced.any():
        voiced_log_pitch = torch.log(pitch_hz[voiced].clamp_min(log_floor))
        log_pitch[voiced] = (
            voiced_log_pitch - voiced_log_pitch.mean()
        ) / voiced_log_pitch.std(unbiased=False).clamp_min(1e-5)

    if voicing is None:
        voicing = voiced.float()
    else:
        voicing = voicing.float().flatten().nan_to_num(0.0)
        if voicing.numel() != pitch_hz.numel():
            voicing = F.interpolate(
                voicing.view(1, 1, -1),
                size=pitch_hz.numel(),
                mode="linear",
                align_corners=False,
            ).view(-1)
        voicing = voicing.clamp_min(0.0).clamp_max(1.0)

    return torch.stack([log_pitch, voicing], dim=-1)


def extract_pitch_features(
    input_values,
    sampling_rate: int,
    frame_shift_ms: float,
    freq_low: float,
    freq_high: float,
    nccf_smoothing_frames: int,
) -> torch.Tensor:
    waveform = torch.as_tensor(input_values, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.numel() == 0:
        return torch.zeros(1, 2, dtype=torch.float32)

    pitch_hz = torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate=int(sampling_rate),
        frame_time=float(frame_shift_ms) / 1000.0,
        win_length=int(nccf_smoothing_frames),
        freq_low=int(freq_low),
        freq_high=int(freq_high),
    )
    if pitch_hz.dim() > 1:
        pitch_hz = pitch_hz.squeeze(0)

    features = normalize_pitch_sequence(pitch_hz)
    if features.numel() == 0:
        return torch.zeros(1, 2, dtype=torch.float32)
    return features.float()
