import os

videos = os.listdir("/home/wg25r/Videos")
for video in videos:
    os.system(f"python3 owl_gasvid.py --video_id {video} --display ''")

os.system("git add .")
os.system(f"git commit -m 'update results for {video}'")
os.system(f"git push")