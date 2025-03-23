from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
import torch
from glob import glob
from collections import defaultdict
import argparse
import os


def main(ckpt_path, step, world_size, output_path):
    
    fsdp_checkpoint_path = os.path.join(ckpt_path, f"global_step_{step}/actor")
    huggingface_model_path = os.path.join(ckpt_path, f"global_step_{step}/actor/huggingface")
    output_path = os.path.join(output_path, f"checkpoint_global_step_{step}")
    state_dict = defaultdict(list)

    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = LlavaForConditionalGeneration._from_config(config)
    # model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.ckpt_path, args.step, args.world_size, args.output_path)
    # ckpt_path = "verl/examples/grpo_trainer/checkpoints/verl_grpo_llava_run/geo3k_tp8_n8_mb_4_gmu03_2k_offload"
    # step = 64
    # world_size = 8
    # output_path = "model_zoo/llava1.5-grpo"
    # main(ckpt_path, step, world_size, output_path)


