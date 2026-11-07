# convert_dcp_to_pt.py                                                                                                                                                                              
import torch                                                                                                                                                                                        
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save                                                                                                                           
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to PyTorch .pt format")
    parser.add_argument("--tag", type=str, required=True, default="cosmos_predict2_2b_480p_pusht_no_pred",help="Tag of the checkpoint to convert")
    parser.add_argument("--iter", type=int, default=5000, help="Iteration number of the checkpoint to convert")
    args = parser.parse_args()
    tag = args.tag
    iter_num = args.iter
    output_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
    base_dir = 'cosmos_v2_finetune' if ('2p5' in tag or '2b' in tag) else 'cosmos_v1_light'
    dcp_dir = f"{output_root}/cosmos_policy/{base_dir}/{tag}/checkpoints/iter_{iter_num:09d}/model"
    output_path = f"{output_root}/cosmos_policy/{base_dir}/{tag}/checkpoints/model_{iter_num:09d}.pt"

    dcp_to_torch_save(dcp_dir, output_path)
    print(f"Saved to {output_path}")