# %% init
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import wandb
import argparse
from boxes import get_valid_boxes

parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=str, default="vid1_")
parser.add_argument("--root_path", type=str, default="../sim")
parser.add_argument("--display", type=str, default="localhost:10.0")
parser.add_argument("--log_file", type=str, default="results.csv")
parser.add_argument("--temporal_filter", action="store_true")
parser.add_argument("--vlm_threashold", type=float, default=0.12)
parser.add_argument("--positive_prompt", type=str, default="white steam")

args = parser.parse_args()
current_video_id = args.video_id
root_path = args.root_path
os.environ["DISPLAY"] = args.display


wandb.init("LangBGS2", name=str(current_video_id))


videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))
# print(videos)
frames = sorted([i for i in os.listdir(os.path.join(root_path, "in")) if i.startswith(current_video_id + "_0")])
gt_id = current_video_id.split("_")[0].replace("vid","smoke_only")+"_0"
gt_frames = sorted([i for i in os.listdir(os.path.join(root_path, "gt")) if i.startswith(gt_id)])
# if current_video_id == "vid11_":
#     frames = frames[300:]
assert len(frames) > 0, "No frames found, video should be one of " + str(videos)
assert len(gt_frames) > 0, "No gt frames found, video should be one of " + str(videos)
assert len(frames) == len(gt_frames),("Number of frames and gt frames should be equal ", len(frames), len(gt_frames), current_video_id)


# %% Load model
sam2_checkpoint = "../../.sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

texts = [args.positive_prompt, "white human, car, bird, bike, and other objects"]

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
config = AutoConfig.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble",
                                                quantization_config={"load_in_4bit": True, 
                                                                     "bnb_4bit_compute_dtype":torch.float16},
                                                config=config)#.to("cuda") make it not quantized actually hurt the performance

img0 = cv2.imread(os.path.join(root_path, "in", frames[0]))
history_state = []
bgsub = cv2.createBackgroundSubtractorMOG2(history=30)
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (img0.shape[1] * 2, img0.shape[0] * 2))

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


frame_level_pred = []
frame_level_gt = []

os.makedirs("result", exist_ok=True)
# todo: there could be places where boxes not detected, use pre and post frames to detect them
for frame in tqdm(frames):
    filename = f'{current_video_id}_{frame.split(".")[0]}'
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img = cv2.imread(os.path.join(root_path, "in", frame))
        bgsub.apply(img)
        if index % 5 != 0: # the fewer frames the lower recall 
            index += 1
            continue 
        bg = bgsub.getBackgroundImage().astype(float)
        diff = cv2.absdiff(img.astype(float), bg)#too large cannot see detial, 2 step and clip each time? target value same
        factor = 15
        high_end = diff.astype(float).mean() + 1 * diff.astype(float).std()
        if high_end * factor > 255:
            factor = 255.0 / high_end
            print("factor", factor)
        diff = diff.astype(float) * factor
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        diff = Image.fromarray(diff)
        inputs = processor(text=texts, images=diff, return_tensors="pt", padding="longest").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([diff.size[::-1]])

        frame = np.array(diff)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=args.vlm_threashold)

        i = 0 
        text = texts[i]
        boxes, logits, phrases = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        positive = boxes[phrases == 0]
        positive_logits = logits[phrases == 0]
        indices = torchvision.ops.nms(positive, positive_logits, 0.3)
        # draw all positive boxes
        all_boxes_canvas = frame.copy()
        for box in positive:
            cv2.rectangle(all_boxes_canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            
            
        if args.temporal_filter:
            valid_boxes = get_valid_boxes(positive[indices], past_boxes, img.shape[:2])
        else:
            valid_boxes = positive[indices]
                    
        # draw valid boxes
        valid_boxes_canvas = frame.copy()
        for box in valid_boxes:
            cv2.rectangle(valid_boxes_canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            
            
        gt = cv2.imread(os.path.join(root_path, "gt", gt_frames[index]))
        gt_binary = (gt.mean(-1) > 20).astype(np.uint8) #ohhhh cannot use 0, also maybe not 3? depends on what it looks like

        mask = np.zeros_like(gt_binary)
        if len(valid_boxes) > 0:        
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=valid_boxes, multimask_output=False)
            mask = masks.sum(0) > 0
            if len(mask.shape) == 3:
                mask = mask[0]
        
        
            frame[mask,0] = 255
                
        if positive[indices].shape[0] > 0:
            past_boxes.append(positive) # add all boxes before nms
            # past_boxes.append(positive[indices])
            
        if len(past_boxes) > 10:
            past_boxes.pop(0)
        # gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        # frame = cv2.hconcat([frame, img])
        # frame_bottom = cv2.hconcat([gt, np.array(diff)])
        # frame = cv2.vconcat([frame, frame_bottom])
        
        confusion.update(torch.tensor(gt_binary), torch.tensor(mask))
        if len(valid_boxes) > 0:
            #this cannot be inside, otherwise it cannot detect false negative
            print(confusion.get_f1(), confusion.get_iou(), confusion.get_precision(), confusion.get_recall())
            wandb.log({"f1": confusion.get_f1(),
                       "iou": confusion.get_iou(),
                        "precision": confusion.get_precision(),
                        "recall": confusion.get_recall(),
                        "frame": wandb.Image(frame),
                        "all_boxes": wandb.Image(all_boxes_canvas),
                        "valid_boxes": wandb.Image(valid_boxes_canvas),
                        "frame": wandb.Image(frame),
                        "diff": wandb.Image(np.array(diff)),
                        "gt": wandb.Image(gt_binary),
                        })
            
        cv2.imwrite(os.path.join("result", filename + ".png"), frame)
        if args.display:
            pylab.clf()
            pylab.imshow(frame)
            pylab.show(block=False)
            pylab.pause(0.0001)
            
        frame_level_pred.append(mask.any())
        frame_level_gt.append(gt_binary.any())
        video_writer.write(frame)
        index += 1
    
video_writer.release()
os.system("ffmpeg -i out.mp4 -c:v libx264 out_x264.mp4 -y > /dev/null 2>&1")


if not os.path.exists(args.log_file):
    with open(args.log_file, "w") as f:
        f.write("video_id,f1,iou,precision,recall,fla,vlm_threashold,temporal_filter,positive_prompt\n")
frame_level_acc = (np.array(frame_level_pred) == np.array(frame_level_gt)).mean()

with open(args.log_file, "a") as f:
    f.write(f"{current_video_id},{confusion.get_f1()},{confusion.get_iou()},{confusion.get_precision()},{confusion.get_recall()},{frame_level_acc},{args.vlm_threashold},{args.temporal_filter},{args.positive_prompt}\n")