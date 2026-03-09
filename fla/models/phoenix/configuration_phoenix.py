
import warnings

from transformers.configuration_utils import PretrainedConfig


class PhoenixConfig(PretrainedConfig):

    model_type = 'phoenix'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        num_slots: Optional[int] = 256,
        feature_map: str = 'swish',
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_logit_normalizer: int = 8,
        elementwise_affine: Optional[bool] = True,
        use_norm: bool = True,
        norm_eps: float = 1e-6,
        topk: int = 128,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        attn: Optional[str] = None,
        log_decay_init: float = -3.0,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        rope_scaling: Optional[dict] = None,
        num_hidden_layers: int = 24,
        share_token_embeddings: bool = False,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_slots = num_slots
        self.topk = topk
        self.feature_map = feature_map
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.share_token_embeddings = share_token_embeddings
        self.gate_logit_normalizer = gate_logit_normalizer
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.attn = attn
        self.log_decay_init = log_decay_init

        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.rope_scaling = rope_scaling

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
