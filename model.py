from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_position: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_position = max_position
        self.register_buffer("pe", self._build(max_position, d_model), persistent=True)

    @staticmethod
    def _build(max_position: int, d_model: int) -> torch.Tensor:
        positions = torch.arange(max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_position, d_model)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term[: pe[0, :, 1::2].shape[-1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.max_position:
            raise ValueError(f"Sequence length {length} exceeds max_position={self.max_position}.")
        return x + self.pe[:, :length].to(dtype=x.dtype, device=x.device)


class PitchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_hidden_size = int(encoder_hidden_size or hidden_size)
        conv_hidden_size = max(32, min(encoder_hidden_size, hidden_size))
        rnn_hidden_size = max(1, encoder_hidden_size // 2)

        self.input_norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_hidden_size, conv_hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            input_size=conv_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
            nn.RMSNorm(hidden_size, eps=1e-6),
        )

    def forward(
        self,
        pitch_features: torch.Tensor,
        target_length: int,
        pitch_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        module_dtype = self.input_norm.weight.dtype
        module_device = self.input_norm.weight.device
        pitch_features = pitch_features.to(dtype=module_dtype, device=module_device)
        if pitch_attention_mask is not None:
            pitch_attention_mask = pitch_attention_mask.to(device=module_device)

        if pitch_features.size(1) != target_length:
            pitch_features = F.interpolate(
                pitch_features.transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            if pitch_attention_mask is not None:
                pitch_attention_mask = F.interpolate(
                    pitch_attention_mask.float().unsqueeze(1),
                    size=target_length,
                    mode="nearest",
                ).squeeze(1).long()

        x = self.input_norm(pitch_features)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.proj(x)

        if pitch_attention_mask is not None:
            x = x * pitch_attention_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        return x


class LingWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(
        self,
        config,
        init_head: bool = False,
        max_canonical_positions: int = 1024,
        pitch_input_dim: int = 2,
        pitch_encoder_hidden_size: Optional[int] = None,
        pitch_dropout: Optional[float] = None,
    ):
        super().__init__(config)
        self.init_head = init_head
        self.wav2vec2 = Wav2Vec2Model(config)

        if config.vocab_size is None:
            raise ValueError("config.vocab_size must be set for LingWav2Vec2ForCTC.")

        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )

        self.config.pitch_input_dim = int(pitch_input_dim)
        self.config.pitch_encoder_hidden_size = pitch_encoder_hidden_size
        self.config.pitch_dropout = pitch_dropout

        self.pitch_encoder = PitchEncoder(
            input_dim=int(pitch_input_dim),
            hidden_size=output_hidden_size,
            encoder_hidden_size=pitch_encoder_hidden_size,
            dropout=config.final_dropout if pitch_dropout is None else float(pitch_dropout),
        )
        self.pitch_query_proj = nn.Sequential(
            nn.RMSNorm(output_hidden_size * 2, eps=1e-6),
            nn.Linear(output_hidden_size * 2, output_hidden_size),
            nn.Dropout(config.final_dropout),
        )

        self.ling_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=output_hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.ling_pos_enc = SinusoidalPositionalEncoding(
            d_model=output_hidden_size,
            max_position=int(max_canonical_positions),
        )
        self.ling_norm = nn.RMSNorm(output_hidden_size, eps=1e-6)
        self.ling_cross_attn = nn.MultiheadAttention(
            embed_dim=output_hidden_size,
            num_heads=max(1, output_hidden_size // 64),
            dropout=getattr(config, "attention_dropout", 0.0),
            batch_first=True,
        )
        self.ling_ffn_layer = nn.Sequential(
            nn.RMSNorm(output_hidden_size, eps=1e-6),
            nn.Linear(output_hidden_size, output_hidden_size * 2),
            SwiGLU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(output_hidden_size, output_hidden_size),
        )
        self.ling_out_norm = nn.RMSNorm(output_hidden_size, eps=1e-6)
        self.ling_lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        self.post_init()

    def freeze_feature_encoder(self) -> None:
        if hasattr(self.wav2vec2, "freeze_feature_encoder"):
            self.wav2vec2.freeze_feature_encoder()
        else:
            self.wav2vec2.feature_extractor._freeze_parameters()

    def _init_weights(self, module):
        if not self.init_head:
            return

        super()._init_weights(module)
        init_range = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.zero_()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        canonical_labels: Optional[torch.Tensor] = None,
        canonical_attention_mask: Optional[torch.Tensor] = None,
        pitch_features: Optional[torch.Tensor] = None,
        pitch_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        if canonical_labels is None:
            raise ValueError("canonical_labels are required for LingWav2Vec2.")
        if pitch_features is None:
            raise ValueError("pitch_features are required for LingWav2Vec2 pitch encoder.")

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        enc_states = outputs[0]
        pitch_states = self.pitch_encoder(
            pitch_features.to(device=enc_states.device),
            target_length=enc_states.size(1),
            pitch_attention_mask=pitch_attention_mask,
        )
        head_dtype = enc_states.dtype
        pitch_states = pitch_states.to(dtype=head_dtype, device=enc_states.device)

        for module in (
            self.pitch_query_proj,
            self.ling_norm,
            self.ling_cross_attn,
            self.ling_ffn_layer,
            self.ling_out_norm,
            self.ling_lm_head,
        ):
            module.to(dtype=head_dtype, device=enc_states.device)

        query_states = self.pitch_query_proj(torch.cat([enc_states, pitch_states], dim=-1))
        ling_emb_states = self.ling_pos_enc(self.ling_emb(canonical_labels)).to(dtype=head_dtype)
        ling_states = self.ling_norm(ling_emb_states)

        key_padding_mask = None
        if canonical_attention_mask is not None:
            key_padding_mask = canonical_attention_mask.eq(0)

        attn_states = self.ling_cross_attn(
            query_states,
            ling_states,
            ling_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        hidden_states = query_states + attn_states
        hidden_states = self.ling_ffn_layer(hidden_states) + hidden_states
        hidden_states = self.ling_out_norm(hidden_states)
        logits = self.ling_lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be smaller than vocab_size={self.config.vocab_size}.")

            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
