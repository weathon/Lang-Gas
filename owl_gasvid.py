# %% init

import warnings

import os
import cv2
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import pylab
import numpy as np
from IPython.display import display, clear_output, HTML
from torchvision.ops import box_convert
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoConfig
import torchvision
import bitsandbytes
import torch

import wandb
import argparse
from boxes import get_valid_boxes

parser = argparse.ArgumentParser()

warnings.filterwarnings("ignore", category=UserWarning)
parser.add_argument("--display", type=str, default="localhost:10.0")
parser.add_argument("--log_file", type=str, default="results.csv")
parser.add_argument("--video_id", type=str, default="MOV_1237")
args = parser.parse_args()
current_video_id = args.video_id
os.environ["DISPLAY"] = args.display

# %% Load model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_checkpoint = "../../.sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

texts = ["white steam", "white human, car, bird, bike, and other objects"]

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
config = AutoConfig.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble",
                                                quantization_config={"load_in_4bit": True, 
                                                                     "bnb_4bit_compute_dtype":torch.float16},
                                                config=config)#.to("cuda") make it not quantized actually hurt the performance


video = cv2.VideoCapture("/home/wg25r/Videos/" + current_video_id + "/frame_%05d.png")
history_state = []
bgsub = cv2.createBackgroundSubtractorMOG2(history=30)

prev_masks = None

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

past_boxes = [] 

if_init = False
out_obj_ids = None
index = 0
pylab.figure(figsize=(15, 10))
masks = torch.zeros((1, 1, 1, 1), device="cuda")
from eval import BinaryConfusion

confusion = BinaryConfusion()


os.makedirs(f"./gasvid_res_full/{current_video_id}", exist_ok=True)
# bar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))
# while True:
    # bar.update(1)
for frame_name in tqdm(sorted(os.listdir(f"/home/wg25r/Videos/{current_video_id}"))):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # ret, img = video.read()
        # if not ret:
        #     break
        # print(os.path.join(f"/home/wg25r/Videos/{current_video_id}", frame_name))
        img = cv2.imread(os.path.join(f"/home/wg25r/Videos/{current_video_id}", frame_name))
        bgsub.apply(img)
        if index % 60 != 0:
            index += 1
            continue
        bg = bgsub.getBackgroundImage().astype(float)
        diff = cv2.absdiff(img.astype(float), bg) * 15 #too large cannot see detial, 2 step and clip each time? target value same
        diff = np.clip(diff, 0, 128).astype(np.uint8)
        
        diff = Image.fromarray(diff)
        inputs = processor(text=texts, images=diff, return_tensors="pt", padding="longest").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([diff.size[::-1]])

        frame = np.array(diff)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.06)

        i = 0
        text = texts[i]
        boxes, logits, phrases = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        positive = boxes[phrases == 0]
        positive_logits = logits[phrases == 0]
        indices = torchvision.ops.nms(positive, positive_logits, 0.3)
        
        
        valid_boxes = get_valid_boxes(boxes[indices], past_boxes, img.shape[:2])

        mask = np.zeros_like(bg)
        
        if len(valid_boxes) > 0:        
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=valid_boxes, multimask_output=False)
            mask = masks.sum(0) > 0
            if len(mask.shape) == 3:
                mask = mask[0]
        
        
            frame[mask,0] = 255
                
        if positive[indices].shape[0] > 0:
            past_boxes.append(positive[indices])
            
        if len(past_boxes) > 10:
            past_boxes.pop(0)
        diff = np.array(diff)
        img = cv2.putText(img, f"Input Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        diff = cv2.putText(diff, f"MOG Difference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mask = (mask > 0).astype(np.uint8) * 255
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
            mask = np.concatenate([mask, mask, mask], axis=-1)
        mask = cv2.putText(mask, f"Predicted Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.hconcat([img, diff, mask])
        
        cv2.imwrite(f"gasvid_res_full/{current_video_id}/{frame_name}", frame)
        index += 1
        
print("Finished prediction on video:", current_video_id)
video.release()

# convert the png sequence into webp using ffmpeg

os.system(f"ffmpeg -i gasvid_res_full/{current_video_id}/frame_%05d.png -c:v libwebp -lossless 0 -q:v 50 -preset veryslow -loop 0 -an -vsync 0 gasvid_res_full/{current_video_id}.webp")
