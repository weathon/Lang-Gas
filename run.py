import os
import pandas as pd
import argparse
import numpy as np
root_path = "../../sim"
videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))
videos.sort()
args = argparse.ArgumentParser()
args.add_argument("--skip_past", action="store_true")
args.add_argument("--method", type=str, default="owl")
# args.add_argument("--temporal_filter", action="store_true")
args.add_argument("--positive_prompt", type=str, default="white steam")
args.add_argument("--disable_wandb", action="store_true")
args = args.parse_args()

filename = args.positive_prompt.replace(" ", "_") + ".csv"

if args.disable_wandb:
    os.system("wandb disabled")
# args.positive_prompt = args.positive_prompt.replace(" ", r"\ ")
blacklist = ["vid26", "vid27", "vid28", "vid24"]

# for vlm_threashold in np.arange(0.01, 0.20, 0.02):
vlm_threashold = 0.12
for video_id in videos:
    if video_id in blacklist:
        continue
    print(f"python3 owl_notracking.py --video_id {video_id} --root_path {root_path} --log_file {filename} --vlm_threashold {vlm_threashold} --temporal_filter --positive_prompt \"{args.positive_prompt}\"")
    os.system(f"python3 owl_notracking.py --video_id {video_id} --root_path {root_path} --log_file {filename} --vlm_threashold {vlm_threashold} --temporal_filter --positive_prompt \"{args.positive_prompt}\"")