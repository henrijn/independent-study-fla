
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


class RoutedDeltaNetAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        use_gate: bool = False,
        use_norm: bool = True,
        layer_idx: int | None = None,
        topk: int = 4,
        router_noise: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode

        # Dimension and nb of heads
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        # Short-Range Convolution
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        # Router
        self.topk = topk
        self.router_noise = router_noise

        # Slot
        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        # Ouput Gate
        self.use_gate = use_gate
        self.use_norm = use_norm

        # Layer cache
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        # Learnable Mamba-style step size parameter
        self.dt = nn.Parameter(torch.empty(self.num_kv_heads))
        nn.init.uniform_(self.dt, math.log(0.001), math.log(0.1))

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        
        self.a_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False)
        self.r_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        if use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, elementwise_affine, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, elementwise_affine, eps=norm_eps, dtype=torch.float32)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding)."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        a = self.a_proj(hidden_states)
        rt = self.r_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        rt = rearrange(rt, '... (h m) -> ... h m', m=self.num_slots)

        a = -F.softplus(a) * torch.exp(self.dt)

        if self.training and self.router_noise:
            x = torch.empty_like(rt).exponential_(1)
            rt = rt - torch.log(x + 1e-6)

        m = F.sigmoid(rt)
        topk_weights, topk_indices = torch.topk(m, self.topk, dim=-1)
        mask = torch.zeros_like(m)
        mask.scatter_(dim=-1, index=topk_indices, src=topk_weights)
        rt = mask / (mask.sum(dim=-1, keepdim=True) + 1e-6)

        g = a.unsqueeze(-1) * rt
        
        k = k * (1 - g.exp()).to(k.dtype)

        if self.num_kv_groups > 1:
            k, v, g = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, g))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
            
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
