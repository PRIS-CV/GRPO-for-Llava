# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import PretrainedConfig

VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen2_vl", "qwen2_5_vl", "llava"}


def get_device_flops(unit="T"):

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type

    if "MI300X" in device_name:
        flops = 1336e12
    elif "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if not config.model_type in VALID_CONFIG_TYPE:
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. "
                  f"MFU will always be zero.")

        self.estimate_func = {
            'qwen2': self._estimate_qwen2_flops,
            'llama': self._estimate_qwen2_flops,
            'qwen2_vl': self._estimate_qwen2_flops,
            'qwen2_5_vl': self._estimate_qwen2_flops,
            "llava": self._estimate_llava_flops,
        }
        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0


    def _estimate_llava_flops(self, tokens_sum, batch_seqlens, delta_time):
        config = self.config
        batch_size = len(batch_seqlens)
        image_seq_length = config.image_seq_length

        vision_config = config.vision_config
        
        v_hidden = vision_config.hidden_size
        v_inter = vision_config.intermediate_size
        v_layers = vision_config.num_hidden_layers
        v_heads = vision_config.num_attention_heads
        patch_size = vision_config.patch_size
        img_size = vision_config.image_size
        channels = 3
        
        num_patches = (img_size // patch_size) ** 2
        
        patch_flops = 2 * (patch_size**2 * channels) * v_hidden * num_patches
        
        per_layer_flops = (
            8 * v_hidden**2 * num_patches +
            2 * v_hidden * num_patches**2 +
            4 * v_hidden * v_inter * num_patches
        )
        
        vision_flops_per = patch_flops + v_layers * per_layer_flops

        text_hidden = config.text_config.hidden_size
        
        proj_inter = text_hidden * 4
        projector_flops_per = 2 * image_seq_length * (
            v_hidden * proj_inter + 
            proj_inter * text_hidden 
        )

        adjusted_seqlens = [seqlen + image_seq_length for seqlen in batch_seqlens]
        seqlen_sq_sum = sum(sl**2 for sl in adjusted_seqlens)
        total_tokens = tokens_sum + batch_size * image_seq_length

        t_config = config.text_config
        t_hidden = t_config.hidden_size
        t_inter = t_config.intermediate_size
        t_layers = t_config.num_hidden_layers
        t_attn_heads = t_config.num_attention_heads
        t_kv_heads = getattr(t_config, 'num_key_value_heads', t_attn_heads)
        vocab_size = t_config.vocab_size

        head_dim = t_hidden // t_attn_heads
        q_size = t_attn_heads * head_dim
        k_size = t_kv_heads * head_dim
        v_size = k_size

        mlp_params = t_hidden * t_inter * 3
        attn_params = t_hidden * (q_size + k_size + v_size + t_attn_heads*head_dim)
        emb_params = vocab_size * t_hidden * 2
        
        dense_flops = 6 * total_tokens * (
            (mlp_params + attn_params) * t_layers + emb_params
        )

        attn_flops = 12 * seqlen_sq_sum * head_dim * t_attn_heads * t_layers

        total_flops = (
            batch_size * (vision_flops_per + projector_flops_per) +  # 视觉部分
            dense_flops + attn_flops  # 文本部分
        )

        return total_flops / delta_time / 1e12
    

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
