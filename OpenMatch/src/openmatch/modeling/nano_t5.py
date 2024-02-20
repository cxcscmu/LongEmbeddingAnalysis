# From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import copy
import math
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5DenseGatedActDense,
    T5DenseActDense
)


@dataclass
class EncoderOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    decoder_hidden_states: object = None
    logits: torch.FloatTensor = None
    encoder_outputs: EncoderOutput = None


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states).type_as(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None
    
    return q_embed, k_embed

class T5Attention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.head_dim = self.d_model // self.n_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,
            base=8000,
        )
            
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = self.q(hidden_states)
        if key_value_states is None:
            key_states, value_states = self.k(hidden_states), self.v(hidden_states)
        else:
            key_states, value_states = self.k(key_value_states), self.v(key_value_states)
        query_states, key_states, value_states = shape(query_states), shape(key_states), shape(value_states)
        
        if key_states.shape[-2] == query_states.shape[-2]:
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            position_ids = torch.arange(
                0, kv_seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        else:
            key_len = key_states.shape[-2]
            k_cos, k_sin = self.rotary_emb(value_states, seq_len=key_len)
            q_len = query_states.shape[-2]
            q_cos, q_sin = self.rotary_emb(value_states, seq_len=q_len)

            key_position_ids = torch.arange(
                0, key_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

            query_position_ids = torch.arange(
                0, q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

            query_states, _ = apply_rotary_pos_emb(query_states, None, q_cos, q_sin, query_position_ids)
            _, key_states = apply_rotary_pos_emb(None, key_states, k_cos, k_sin, key_position_ids)


        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    
        if mask is not None:
            scores = scores + mask

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return (attn_output, position_bias)


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.SelfAttention = T5Attention(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states).type_as(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states,)
        outputs = outputs + attention_outputs

        return outputs  # hidden-states, (self-attention position bias), (cross-attention position bias)


class T5Stack(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, embed_tokens):
        super().__init__()
        assert embed_tokens is not None

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config) for i in range(config.num_layers)]
        )

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ) -> EncoderOutput:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self.config, 'is_bf16') and self.config.is_bf16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # Masking
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for _, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )
            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[2]
        
        hidden_states = self.final_layer_norm(hidden_states).type_as(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return EncoderOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )


class MyT5RoPE(nn.Module):
    _keys_to_ignore_on_load_unexpected = [
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]
    def __init__(self, config: T5Config):
        super().__init__()

        config.is_encoder_decoder = False
        #assert not config.tie_word_embeddings

        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.generation_config = None

        self.apply(self._init_weights)
        self.lm_head.weight = self.shared.weight

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_length = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            
            Generation:
                Starts with 0, ends with 1, padding is 0

            # For 20 input/outputs, the diff between my implementation and HF is 9.8s vs 11.4s
        """
        B, _ = input_ids.size()
        labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
        encoder_outputs = None

        for _ in range(max_length):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=labels,
                encoder_outputs=encoder_outputs,
            )
            encoder_outputs = out.encoder_outputs
            top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
            labels = torch.cat([labels, top_labels], dim=-1)

            if (labels == 1).sum(-1).clamp(min=0, max=1).sum().item() == B:
                break
        
        labels[:, -1] = 1

        # Mask out the padding, i.e., all positions after the first 1 with 0
        B, L = labels.size()
        mask = torch.arange(L, device=labels.device).unsqueeze(0) <= (labels == 1).long().argmax(-1).unsqueeze(-1)
        labels = labels.masked_fill(~mask, 0)

        return labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_outputs = None,
        return_dict = None
    ) -> Seq2SeqLMOutput:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            labels: B x L_decoder, int64
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = encoder_outputs.hidden_states

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        if not return_dict:
            return encoder_outputs + decoder_outputs

        return Seq2SeqLMOutput(
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_outputs=encoder_outputs,
        )

    def _init_weights(self, module):
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (MyT5RoPE)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseGatedActDense):
            d_ff, d_model = module.wi_0.weight.data.size()
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None and pad_token_id is not None
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

