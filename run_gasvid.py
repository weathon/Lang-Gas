import os

videos = os.listdir("/home/wg25r/Videos")
for video in videos:
    os.system(f"python3 owl_gasvid.py --video_id {video} --display ''")
