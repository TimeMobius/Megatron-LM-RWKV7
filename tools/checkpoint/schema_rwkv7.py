# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""RWKV7 model schemas."""

import typing as T

from schema_base import ModelSchema
from schema_core import CoreSchema, get_core_transformer_block_key

class RWKV7LayerSchema:
    """RWKV7层的Schema定义"""
    
    def __init__(self, use_bias=True):
        self.schema = {
            # TimeMix模块 (对应Transformer的Self-Attention)
            "self_attn_norm_weight": "input_layernorm.weight",
            "self_attn_norm_bias": "input_layernorm.bias",

            # TimeMix的Bottleneck组件
            "bottleneck_learn_weight_down": "self_attention.bottleneck_learn.weight_down",
            "bottleneck_learn_weight_up": "self_attention.bottleneck_learn.weight_up",
            "bottleneck_learn_bias": "self_attention.bottleneck_learn.bias" if use_bias else None,
            
            "bottleneck_gate_weight_down": "self_attention.bottleneck_gate.weight_down",
            "bottleneck_gate_weight_up": "self_attention.bottleneck_gate.weight_up",
            
            "bottleneck_decay_weight_down": "self_attention.bottleneck_decay.weight_down",
            "bottleneck_decay_weight_up": "self_attention.bottleneck_decay.weight_up",
            "bottleneck_decay_bias": "self_attention.bottleneck_decay.bias" if use_bias else None,
            
            "bottleneck_v_weight_down": "self_attention.bottleneck_v.weight_down",
            "bottleneck_v_weight_up": "self_attention.bottleneck_v.weight_up",
            "bottleneck_v_bias": "self_attention.bottleneck_v.bias" if use_bias else None,
            
            # TimeMix的特有参数
            "mix_coeff_input": "self_attention.mix_coeff_input",
            "rho": "self_attention.rho",
            "beta": "self_attention.beta",
            "alpha": "self_attention.alpha",
            
            # TimeMix的线性层
            "self_attn_r_weight": "self_attention.linear_receptance.weight",
            "self_attn_r_bias": "self_attention.linear_receptance.bias" if use_bias else None,
            "self_attn_k_weight": "self_attention.linear_key.weight",
            "self_attn_k_bias": "self_attention.linear_key.bias" if use_bias else None,
            "self_attn_v_weight": "self_attention.linear_value.weight",
            "self_attn_v_bias": "self_attention.linear_value.bias" if use_bias else None,
            
            # GroupNorm
            "group_norm_weight": "self_attention.group_norm.weight",
            "group_norm_bias": "self_attention.group_norm.bias" if use_bias else None,
            
            # TimeMix的输出投影
            "self_attn_proj_weight": "self_attention.linear_proj.weight",
            "self_attn_proj_bias": "self_attention.linear_proj.bias" if use_bias else None,
            
            # ChannelMix模块 (对应Transformer的MLP)
            "mlp_norm_weight": "pre_mlp_layernorm.weight",
            "mlp_norm_bias": "pre_mlp_layernorm.bias",
            
            # ChannelMix参数
            "mix_coeffs": "mlp.mix_coeffs",
            
            # ChannelMix的线性层
            "mlp_fc1_weight": "mlp.linear_up.weight",
            "mlp_fc1_bias": "mlp.linear_up.bias" if use_bias else None,
            "mlp_fc2_weight": "mlp.linear_down.weight",
            "mlp_fc2_bias": "mlp.linear_down.bias" if use_bias else None,
        }
        
        # 过滤掉None值
        self.schema = {k: v for k, v in self.schema.items() if v is not None}
    
    def __getitem__(self, key):
        return self.schema.get(key)


class RWKV7Schema(CoreSchema):
    """RWKV7模型的Schema定义"""
    
    def __init__(self, model_type, use_bias=True):
        block_key = get_core_transformer_block_key(model_type)
        layer_schema = RWKV7LayerSchema(use_bias).schema
        
        schema_dict = {
            "embeddings": {
                "pos": "embedding.position_embeddings.weight",
                "word": "embedding.word_embeddings.weight",
            },
            "layer_prefix": f"{block_key}.layers",
            "layer": layer_schema,
            "final_norm": {
                "weight": f"{block_key}.final_layernorm.weight",
                "bias": f"{block_key}.final_layernorm.bias",
            },
            "output_layer": {
                "weight": "output_layer.weight",
            },
        }
        
        super(CoreSchema, self).__init__(schema_dict)


def get_rwkv7_schema(
    model_type: T.Literal["GPT", "BERT"],
    use_bias: bool = True,
) -> RWKV7Schema:
    """获取RWKV7模型的Schema"""
    return RWKV7Schema(model_type, use_bias)