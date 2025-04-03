import os
import cv2


os.makedirs("sim/in", exist_ok=True)
os.makedirs("sim/background", exist_ok=True)
os.makedirs("sim/gt", exist_ok=True)


videos = os.listdir("simulated_gas")
# videos = ["smoke_only17"]

from tqdm import tqdm
for video in tqdm(videos):
    if video.startswith("sim"):
        cap = cv2.VideoCapture("simulated_gas/"+video)
        bgsub = cv2.createBackgroundSubtractorMOG2(history=20)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bgsub.apply(frame)
            bg = bgsub.getBackgroundImage()
            fgmask = cv2.absdiff(frame, bg)
            cv2.imwrite(f"sim/in/{video.split('.')[0].replace('simgas','')}_{frame_id:04d}.png", frame)
            cv2.imwrite(f"sim/background/{video.split('.')[0].replace('simgas','')}_{frame_id:04d}.png", bg)
            frame_id += 1
    else:
        # print(video)
        cap = cv2.VideoCapture("simulated_gas/"+video)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"sim/gt/{video.split('.')[0].replace('smokeonly','')}_{frame_id:04d}.png", frame)
            frame_id += 1