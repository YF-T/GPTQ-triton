import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, repeat_kv, apply_rotary_pos_emb, \
    LlamaRotaryEmbedding, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN


logger = logging.get_logger(__name__)


class ReorderLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.register_buffer('q_order_row', torch.zeros(self.num_heads * self.head_dim, dtype=torch.int32))
        self.register_buffer('k_order_row', torch.zeros(self.num_key_value_heads * self.head_dim, dtype=torch.int32))
        self.register_buffer('v_order_row', torch.zeros(self.num_key_value_heads * self.head_dim, dtype=torch.int32))
        self.register_buffer('o_order_row', torch.zeros(self.hidden_size, dtype=torch.int32))

        self.register_buffer('q_order_col', torch.zeros(self.hidden_size, dtype=torch.int32))
        self.register_buffer('k_order_col', torch.zeros(self.hidden_size, dtype=torch.int32))
        self.register_buffer('v_order_col', torch.zeros(self.hidden_size, dtype=torch.int32))
        self.register_buffer('o_order_col', torch.zeros(self.num_heads * self.head_dim, dtype=torch.int32))

        self.register_buffer('q_block_size', torch.zeros(2, dtype=torch.int32))
        self.register_buffer('k_block_size', torch.zeros(2, dtype=torch.int32))
        self.register_buffer('v_block_size', torch.zeros(2, dtype=torch.int32))
        self.register_buffer('o_block_size', torch.zeros(2, dtype=torch.int32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # if self.config.pretraining_tp > 1:
        #     key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        #     query_slices = self.q_proj.weight.split(
        #         (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        #     )
        #     key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        #     value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        #     query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        #     query_states = torch.cat(query_states, dim=-1)

        #     key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        #     key_states = torch.cat(key_states, dim=-1)

        #     value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        #     value_states = torch.cat(value_states, dim=-1)

        # else:
        #     query_states = self.q_proj(hidden_states)
        #     key_states = self.k_proj(hidden_states)
        #     value_states = self.v_proj(hidden_states)
        query_states = self.q_proj(hidden_states[:, :, self.q_order_col])[:, :, self.q_order_row]
        key_states = self.k_proj(hidden_states[:, :, self.k_order_col])[:, :, self.k_order_row]
        value_states = self.v_proj(hidden_states[:, :, self.v_order_col])[:, :, self.v_order_row]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        # if self.config.pretraining_tp > 1:
        #     attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        #     o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        #     attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        # else:
        #     attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj(attn_output[:, :, self.o_order_col])[:, :, self.o_order_row]
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ReorderLlamaFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

        self.register_buffer('gate_order_row', torch.zeros(self.intermediate_size, dtype=torch.int32))
        self.register_buffer('up_order_row', torch.zeros(self.intermediate_size, dtype=torch.int32))
        self.register_buffer('down_order_row', torch.zeros(self.hidden_size, dtype=torch.int32))

        self.register_buffer('gate_order_col', torch.zeros(self.hidden_size, dtype=torch.int32))
        self.register_buffer('up_order_col', torch.zeros(self.hidden_size, dtype=torch.int32))
        self.register_buffer('down_order_col', torch.zeros(self.intermediate_size, dtype=torch.int32))

        self.register_buffer('gate_block_size', torch.zeros(2, dtype=torch.int32))
        self.register_buffer('up_block_size', torch.zeros(2, dtype=torch.int32))
        self.register_buffer('down_block_size', torch.zeros(2, dtype=torch.int32))

    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #     slice = self.intermediate_size // self.config.pretraining_tp
        #     gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        #     up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        #     down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        #     gate_proj = torch.cat(
        #         [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        #     )
        #     up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        #     intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        #     down_proj = [
        #         F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        #     ]
        #     down_proj = sum(down_proj)
        # else:
        #     down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x1 = self.gate_proj(x[:, :, self.gate_order_col])[:, :, self.gate_order_row]
        x1 = self.act_fn(x1)
        x2 = self.up_proj(x[:, :, self.up_order_col])[:, :, self.up_order_row]
        y = x1 * x2
        y = self.down_proj(y[:, :, self.down_order_col])[:, :, self.down_order_row]

        return y


LLAMA_ATTENTION_CLASSES = {
    "eager": ReorderLlamaAttention,
    # "flash_attention_2": LlamaFlashAttention2,
    # "sdpa": LlamaSdpaAttention,
}


class ReorderLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Rewirte the LlamaDecoderLayer
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = ReorderLlamaFFN(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class ReorderLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ReorderLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ReorderLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class ReorderLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ReorderLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()